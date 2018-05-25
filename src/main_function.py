from os import path
from src.feature import *
from src.my_dataset import *
from src.files   import *
from src.evaluation import *
import numpy as np
import csv
import copy
from sklearn import mixture
#特徴量のファイル作成
def get_feature_filename(audio_file, path, extension='cpickle'):
    audio_filename = path.split(audio_file)[1]
    return path.join(path, path.splitext(audio_filename)[0] + '.' + extension)
#正規化のファイル取得
def get_feature_normalizer_filename(path, extension='cpickle'):
    return path.join(path, 'scale' + '.' + extension)
#モデルのファイル取得
def get_model_filename(path, extension='cpickle'):
    return path.join(path, 'model'  + '.' + extension)
#結果のファイル取得
def get_result_filename(path, extension='txt'):
    return path.join(path, 'results.' + extension)
#特徴量抽出を実行
def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    for file_id, audio_filename in enumerate(files):
        # 特徴量のファイルを獲得
        current_feature_file = get_feature_filename(audio_file=path.split(audio_filename)[1], path=feature_path)
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
def do_feature_normalization(dataset, feature_normalizer_path, feature_path):
    current_normalizer_file = get_feature_normalizer_filename(path=feature_normalizer_path)
    normalizer = FeatureNormalizer()
    for item_id, item in enumerate(dataset.train()):# Load features
        feature_data = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))['stat']
        # Accumulate statistics
        normalizer.accumulate(feature_data)
    # Calculate normalization factors
    normalizer.finalize()
    save_data(current_normalizer_file, normalizer)
def do_system_training(dataset, model_path, feature_normalizer_path, feature_path, feature_params, classifier_params,
                       classifier_method='gmm', clean_audio_errors=False, overwrite=False):
    current_model_file = get_model_filename(path=model_path)
    feature_normalizer_filename = get_feature_normalizer_filename(path=feature_normalizer_path)
    normalizer = load_data(feature_normalizer_filename)
    # Initialize model container
    model_container = {'normalizer': normalizer, 'models': {}}
    # Collect training examples
    file_count = len(dataset.train())
    data = {}
    for item_id, item in enumerate(dataset.train()):
        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
        feature_data = load_data(feature_filename)['feat']
        feature_data = model_container['normalizer'].normalize(feature_data)
        if item['scene_label'] not in data:
            data[item['scene_label']] = feature_data
        else:
            data[item['scene_label']] = np.vstack((data[item['scene_label']], feature_data))
    # Train models for each class
    for label in data:
        ##model_container['models'][label] = mixture.GMM(classifier_params).fit(data[label])
        model_container['models'][label] = mixture.GMM(n_components=16, covariance_type='diag', random_state=0, min_covar=0.001).fit(data[label])
    # モデルの保存
    save_data(current_model_file, model_container)

def do_system_testing(dataset, result_path, feature_path, model_path, feature_params):
    current_result_file = get_result_filename(path=result_path)
    results = []
    model_filename = get_model_filename(path=model_path)
    model_container = load_data(model_filename)
    file_count = len(dataset.test())
    for file_id, item in enumerate(dataset.test()):
        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
        feature_data = load_data(feature_filename)['feat']
        feature_data = model_container['normalizer'].normalize(feature_data)
        current_result = do_classification_gmm(feature_data, model_container)
        results.append((dataset.absolute_to_relative(item['file']), current_result))
        with open(current_result_file, 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for result_item in results:
                writer.writerow(result_item)
def do_classification_gmm(feature_data, model_container):
    # マイナス無限大でlog-likelihood行列を初期化
    logls = np.empty(len(model_container['models']))
    logls.fill(-np.inf)
    for label_id, label in enumerate(model_container['models']):
        logls[label_id] = np.sum(model_container['models'][label].score(feature_data))
    classification_result_id = np.argmax(logls)
    return model_container['models'].keys()[classification_result_id]

def do_system_evaluation(dataset, result_path):
    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
    results = []
    result_filename = get_result_filename(path=result_path)
    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            results.append(row)
    y_true = []
    y_pred = []
    for result in results:
        y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
        y_pred.append(result[1])
    dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
    results = dcase2016_scene_metric.results()
