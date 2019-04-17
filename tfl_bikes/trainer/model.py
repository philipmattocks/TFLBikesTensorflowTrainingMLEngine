#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = None  # set from task.py
PATTERN = '' # gets all files
#TRAIN_STEPS = 10000
CSV_COLUMNS =  'precipIntensity,temperature,badWeather,hour,dayofweek,bikesused,key,weekend'.split(',')
LABEL_COLUMN = 'bikesused'
KEY_COLUMN = 'key'
DEFAULTS = [[0.0],[15.0],["False"],[0],['Monday'],[0.5], ['nokey'],["False"]]

def read_dataset(prefix, pattern, batch_size=512):
    # use prefix to create filename
    filename = 'gs://{}/{}*'.format(BUCKET, prefix, pattern)   
    print(filename)
    if prefix == 'train':
        mode = tf.estimator.ModeKeys.TRAIN
        num_epochs = None # indefinitely
    else:
        mode = tf.estimator.ModeKeys.EVAL
        num_epochs = 1 # end-of-input after this
    
    # the actual input function passed to TensorFlow
    def _input_fn():
        # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
            input_file_names, shuffle=True, num_epochs=num_epochs)
 
        # read CSV
        reader = tf.TextLineReader(skip_header_lines=1)
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)
        if mode == tf.estimator.ModeKeys.TRAIN:
          value = tf.train.shuffle_batch([value], batch_size, capacity=10*batch_size, 
                                         min_after_dequeue=batch_size, enqueue_many=True, 
                                         allow_smaller_final_batch=False)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        features.pop(KEY_COLUMN)
        #features.pop('callTime')
        label = features.pop(LABEL_COLUMN)
        return features, label
  
    return _input_fn

#def get_wide_deep():
def get_features():
    # define column types
    #is_male,mother_age,plurality,gestation_weeks = \
    badWeather,precipIntensity,temperature,hour,weekend,dayofweek = \
        [\
            #tf.feature_column.numeric_column('badWeather'),
            tf.feature_column.categorical_column_with_vocabulary_list('badWeather',['True','False']),
            tf.feature_column.numeric_column('precipIntensity'),
            tf.feature_column.numeric_column('temperature'),
            tf.feature_column.numeric_column('hour'),
            #tf.feature_column.numeric_column('weekend'),
            tf.feature_column.categorical_column_with_vocabulary_list('weekend',['True','False']),
            tf.feature_column.categorical_column_with_vocabulary_list('dayofweek',['Sunday', 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
        ]

    # discretize
    hour_buckets = tf.feature_column.bucketized_column(hour, 
                        boundaries=np.arange(0,24,1).tolist())
      
    # sparse columns are wide 
    #wide = [badWeather,
    #        hour_buckets,
    #        weekend,
    #        dayofweek]
 
    
    # feature cross all the wide columns and embed into a lower dimension
    #Columns
    #crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
    #embed = tf.feature_column.embedding_column(crossed, 4)
    badWeather_indicator = tf.feature_column.indicator_column(badWeather)
    dayofweek_indicator = tf.feature_column.indicator_column(dayofweek)
    weekend_indicator = tf.feature_column.indicator_column(weekend)
    # continuous columns are deep
    #deep = [badWeather,
    #        embed]
    features = [precipIntensity,temperature,hour_buckets,badWeather_indicator,dayofweek_indicator,weekend_indicator]
    return features
    #deep = [precipIntensity,temperature,embed]
    #return wide, deep

def serving_input_fn():
    feature_placeholders = {
        'hour': tf.placeholder(tf.float32, [None]),
        'dayofweek': tf.placeholder(tf.string, [None]),
        'weekend': tf.placeholder(tf.string, [None]),
        #'badWeather': tf.placeholder(tf.bool, [None])
        'badWeather': tf.placeholder(tf.string, [None]),
        'temperature': tf.placeholder(tf.float32, [None]),
        'precipIntensity': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

#def experiment_fn(output_dir):
#    wide, deep = get_wide_deep()
#    return tf.contrib.learn.Experiment(
#        tf.estimator.DNNLinearCombinedRegressor(model_dir=output_dir,
#                                                    linear_feature_columns=wide,
#                                                    dnn_feature_columns=deep,
#                                                    dnn_hidden_units=[64, 32],
#        train_input_fn=read_dataset('train', PATTERN),
#        eval_input_fn=read_dataset('eval', PATTERN),
#        export_strategies=[saved_model_export_utils.make_export_strategy(
#            serving_input_fn,
#            default_output_alternative_key=None,
#            exports_to_keep=1
#        )],
#        train_steps=TRAIN_STEPS,
#        eval_steps=None
#    )
run_config = tf.estimator.RunConfig(save_checkpoints_secs = 150, 
                                    keep_checkpoint_max = 3)
def train_and_evaluate(output_dir):
    features = get_features()
    estimator = tf.estimator.DNNRegressor(
                         model_dir=output_dir,
                         #linear_feature_columns=wide,
                         #dnn_feature_columns=deep,
                         feature_columns=features,
                         #dnn_hidden_units=[64, 32])#,
                         hidden_units=[64, 32],
                         config = run_config)
                         #config = run_config)
    train_spec=tf.estimator.TrainSpec(
                         input_fn=read_dataset('train', PATTERN),
                         max_steps=TRAIN_STEPS)
    exporter = tf.estimator.LatestExporter('exporter',serving_input_fn)
    eval_spec=tf.estimator.EvalSpec(
                         input_fn=read_dataset('eval', PATTERN),
                         steps=None,
                         start_delay_secs = 1, # start evaluating after N seconds
                         throttle_secs = 10,  # evaluate every N seconds
                         exporters=exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


#def train_and_evaluate(output_dir):
#    wide, deep = get_wide_deep()
#    estimator = tf.estimator.DNNLinearCombinedRegressor(
#                         model_dir=output_dir,
#                         linear_feature_columns=wide,
#                         dnn_feature_columns=deep,
#                         dnn_hidden_units=[64, 32],
#                         config = run_config)
#    train_spec=tf.estimator.TrainSpec(
#                         input_fn=read_dataset('train', PATTERN),
#                         max_steps=TRAIN_STEPS)
#    exporter = tf.estimator.LatestExporter('exporter',serving_input_fn)
#    eval_spec=tf.estimator.EvalSpec(
#                         input_fn=read_dataset('eval', PATTERN),
#                         steps=None,
#                         start_delay_secs = 1, # start evaluating after N seconds#Doesn't seem to have effect when running remotely, set via run_config
#                         throttle_secs = 10,  # evaluate every N seconds#Doesn't seem to have effect when running remotely, set via run_config
#                         exporters=exporter)
#    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

