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
#特徴量のファイル作成
def get_feature_filename(audio_file, path, extension='cpickle'):
    audio_filename = os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)
#正規化のファイル取得
def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)
#モデルのファイル取得
def get_model_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)
#結果のファイル取得
def get_result_filename(fold, path, extension='txt'):
    return os.path.join(path, 'results.' + extension)
#特徴量抽出を実行
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
# 特徴量の正規化実行
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

def do_system_testing(dataset, result_path, feature_path, model_path, feature_params,
                      dataset_evaluation_mode='folds', classifier_method='gmm', clean_audio_errors=False, overwrite=False):
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_result_file = get_result_filename(fold=fold, path=result_path)
        if not os.path.isfile(current_result_file) or overwrite:
            results = []
            # Load class model container
            model_filename = get_model_filename(fold=fold, path=model_path)
            model_container = load_data(model_filename)
            file_count = len(dataset.test(fold))
            for file_id, item in enumerate(dataset.test(fold)):
                progress(title_text='Testing',
                         fold=fold,
                         percentage=(float(file_id) / file_count),
                         note=os.path.split(item['file'])[1])
                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)

                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    # Load audio
                    y, fs = load_audio(filename=dataset.relative_to_absolute_path(item['file']), mono=True, fs=feature_params['fs'])
                    feature_data = feature_extraction(y=y,
                                                      fs=fs,
                                                      include_mfcc0=feature_params['include_mfcc0'],
                                                      include_delta=feature_params['include_delta'],
                                                      include_acceleration=feature_params['include_acceleration'],
                                                      mfcc_params=feature_params['mfcc'],
                                                      delta_params=feature_params['mfcc_delta'],
                                                      acceleration_params=feature_params['mfcc_acceleration'],
                                                      statistics=False)['feat']

                # Scale features
                feature_data = model_container['normalizer'].normalize(feature_data)

                if clean_audio_errors:
                    current_errors = dataset.file_error_meta(item['file'])
                    if current_errors:
                        removal_mask = numpy.ones((feature_data.shape[0]), dtype=bool)
                        for error_event in current_errors:
                            onset_frame = int(numpy.floor(error_event['event_onset'] / feature_params['hop_length_seconds']))
                            offset_frame = int(numpy.ceil(error_event['event_offset'] / feature_params['hop_length_seconds']))
                            if offset_frame > feature_data.shape[0]:
                                offset_frame = feature_data.shape[0]
                            removal_mask[onset_frame:offset_frame] = False
                        feature_data = feature_data[removal_mask, :]

                # Do classification for the block
                if classifier_method == 'gmm':
                    current_result = do_classification_gmm(feature_data, model_container)
                else:
                    raise ValueError("Unknown classifier method ["+classifier_method+"]")

                # Store the result
                results.append((dataset.absolute_to_relative(item['file']), current_result))

            # Save testing results
            with open(current_result_file, 'wt') as f:
                writer = csv.writer(f, delimiter='\t')
                for result_item in results:
                    writer.writerow(result_item)


def do_classification_gmm(feature_data, model_container):
    # Initialize log-likelihood matrix to -inf
    logls = numpy.empty(len(model_container['models']))
    logls.fill(-numpy.inf)

    for label_id, label in enumerate(model_container['models']):
        logls[label_id] = numpy.sum(model_container['models'][label].score(feature_data))

    classification_result_id = numpy.argmax(logls)
    return model_container['models'].keys()[classification_result_id]

def do_system_evaluation(dataset, result_path, dataset_evaluation_mode='folds'):
    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
    results_fold = []
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        dcase2016_scene_metric_fold = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        results = []
        result_filename = get_result_filename(fold=fold, path=result_path)
        with open(result_filename, 'rt') as f:
            for row in csv.reader(f, delimiter='\t'):
                results.append(row)
        y_true = []
        y_pred = []
        for result in results:
            y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        dcase2016_scene_metric_fold.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        results_fold.append(dcase2016_scene_metric_fold.results())
    results = dcase2016_scene_metric.results()
    fold_labels = ''
    separator = '     =====================+======+======+==========+  +'
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_labels += " {:8s} |".format('Fold'+str(fold))
            separator += "==========+"
    for label_id, label in enumerate(sorted(results['class_wise_accuracy'])):
        fold_values = ''
        if dataset.fold_count > 1:
            for fold in dataset.folds(mode=dataset_evaluation_mode):
                fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['class_wise_accuracy'][label] * 100)
    fold_values = ''
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['overall_accuracy'] * 100)
