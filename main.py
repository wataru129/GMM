#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from src.main_function   import *

start = time.time()
# パラメータ設定ファイルの指定
parameter_file = './setting/parameter.yaml'
# 設定パラメータ(yaml)の読み込み
params = load_parameters(parameter_file)
#params = process_parameters(params)
dataset = eval(params['general']['development_dataset'])(data_path=params['path']['data'])
params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])
files = []
dataset_evaluation_mode = 'folds'
for fold in dataset.folds(mode=dataset_evaluation_mode):
    for item_id, item in enumerate(dataset.train(fold)):
        if item['file'] not in files:
            files.append(item['file'])
    for item_id, item in enumerate(dataset.test(fold)):
        if item['file'] not in files:
            files.append(item['file'])
files = sorted(files)
# ファイルを調べ、すべての特徴量が抽出されていることを確認する
print("feature_extract")
#if not os.path.exists("features"):
do_feature_extraction(files=files,
                          dataset=dataset,
                          feature_path=params['path']['features'],
                          params=params['features'],
                          overwrite=params['general']['overwrite'])
print("feature_normalize")
do_feature_normalization(dataset=dataset,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 overwrite=params['general']['overwrite'])
print("trainning")

do_system_training(dataset=dataset,
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           feature_params=params['features'],
#                           classifier_params=params['classifier']['parameters'],
                           classifier_params=a,
                           classifier_method=params['classifier']['method'],
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           clean_audio_errors=params['classifier']['audio_error_handling']['clean_data'],
                           overwrite=params['general']['overwrite']
                           )

print("create challenge dataset")
challenge_dataset = eval(params['general']['challenge_dataset'])(data_path=params['path']['data'])
result_path = params['path']['results']
if not params['general']['challenge_submission_mode']:
        # Extract feature if not running in challenge submission mode.
        # Collect test files
        files = []
        for fold in challenge_dataset.folds(mode=dataset_evaluation_mode):
            for item_id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)
            # Go through files and make sure all features are extracted
        do_feature_extraction(files=files,
                                  dataset=challenge_dataset,
                                  feature_path=params['path']['features'],
                                  params=params['features'],
                                  overwrite=params['general']['overwrite'])
print("testing")
do_system_testing(dataset=challenge_dataset,
                  feature_path=params['path']['features'],
                  result_path=result_path,
                  model_path=params['path']['models'],
                  feature_params=params['features'],
                  dataset_evaluation_mode=dataset_evaluation_mode,
                  classifier_method=params['classifier']['method'],
                  clean_audio_errors=params['recognizer']['audio_error_handling']['clean_data'],
                  overwrite=params['general']['overwrite'] or params['general']['challenge_submission_mode']
                  )
print(evaluation)
do_system_evaluation(dataset=challenge_dataset,
                     dataset_evaluation_mode=dataset_evaluation_mode,
                     result_path=result_path)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
