import os
from src.feature import *
from src.dataset import *
from src.files   import *
import numpy
import csv
import argparse
import textwrap
import copy

from sklearn import mixture

def get_feature_filename(audio_file, path, extension='cpickle'):
    audio_filename = os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)
# 特徴量抽出を実行するための関数
def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)

def get_model_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)

def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    for file_id, audio_filename in enumerate(files):
        # 特徴量のファイルを獲得
        current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)
        y, fs = load_audio(filename=dataset.relative_to_absolute_path(audio_filename), mono=True, fs=params['fs'])
        feature_data = feature_extraction(y=y,
                                              fs=fs,
                                              include_mfcc0=params['include_mfcc0'],
                                              include_delta=params['include_delta'],
                                              include_acceleration=params['include_acceleration'],
                                              mfcc_params=params['mfcc'],
                                              delta_params=params['mfcc_delta'],
                                              acceleration_params=params['mfcc_acceleration'])
        # オブジェクトの保存
        save_data(current_feature_file, feature_data)

def do_feature_normalization(dataset, feature_normalizer_path, feature_path, dataset_evaluation_mode='folds', overwrite=False):
    # Check that target path exists, create if not
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_normalizer_file = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)

        if not os.path.isfile(current_normalizer_file) or overwrite:
            # Initialize statistics
            file_count = len(dataset.train(fold))
            normalizer = FeatureNormalizer()

            for item_id, item in enumerate(dataset.train(fold)):                # Load features
                feature_data = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))['stat']
                # Accumulate statistics
                normalizer.accumulate(feature_data)

            # Calculate normalization factors
            normalizer.finalize()
            # Save
            save_data(current_normalizer_file, normalizer)

def process_parameters(params):
    # Convert feature extraction window and hop sizes seconds to samples
    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])
    # 現在の分類パラメータを保存
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]
    # Let's keep hashes backwards compatible after added parameters.
    classifier_params = copy.copy(params['classifier'])
    return params
def do_system_training(dataset, model_path, feature_normalizer_path, feature_path, feature_params, classifier_params,
                       dataset_evaluation_mode='folds', classifier_method='gmm', clean_audio_errors=False, overwrite=False):
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_model_file = get_model_filename(fold=fold, path=model_path)
        if not os.path.isfile(current_model_file) or overwrite:
            # Load normalizer
            feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
            normalizer = load_data(feature_normalizer_filename)
            # Initialize model container
            model_container = {'normalizer': normalizer, 'models': {}}
            # Collect training examples
            file_count = len(dataset.train(fold))
            data = {}
            for item_id, item in enumerate(dataset.train(fold)):
                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                feature_data = load_data(feature_filename)['feat']
                # Scale features
                feature_data = model_container['normalizer'].normalize(feature_data)
                # Store features per class label
                if item['scene_label'] not in data:
                    data[item['scene_label']] = feature_data
                else:
                    data[item['scene_label']] = numpy.vstack((data[item['scene_label']], feature_data))
            # Train models for each class
            for label in data:
##                model_container['models'][label] = mixture.GMM(classifier_params).fit(data[label])
                model_container['models'][label] = mixture.GMM(n_components=16, covariance_type='diag', random_state=0, min_covar=0.001).fit(data[label])
            # Save models
            save_data(current_model_file, model_container)
