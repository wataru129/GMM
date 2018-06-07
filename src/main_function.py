import os
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
    audio_filename = os.path.split(audio_file)[1]
    os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)
#正規化のファイル取得
def get_feature_normalizer_filename(path, extension='cpickle'):
    return os.path.join(path, 'scale' + '.' + extension)
#モデルのファイル取得
def get_model_filename(path, extension='cpickle'):
    return os.path.join(path, 'model'  + '.' + extension)
#結果のファイル取得
def get_result_filename(path, extension='txt'):
    return os.path.join(path, 'results.' + extension)
#特徴量抽出を実行
def do_feature_extraction(files, dataset, feature_path, params):
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
def do_feature_normalization(dataset, feature_normalizer_path, feature_path):
    current_normalizer_file = get_feature_normalizer_filename(path=feature_normalizer_path)
    normalizer = FeatureNormalizer()
    for item_id, item in enumerate(dataset.train()):# Load features
        feature_data = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))['stat']
        # 統計量の蓄積
        normalizer.accumulate(feature_data)
    # 正規化要素の計算
    normalizer.finalize()
    save_data(current_normalizer_file, normalizer)
def do_system_training(dataset, model_path, feature_normalizer_path, feature_path, feature_params, classifier_params):
    #ファイルのパス取得
    current_model_file = get_model_filename(path=model_path)
    feature_normalizer_filename = get_feature_normalizer_filename(path=feature_normalizer_path)
    normalizer = load_data(feature_normalizer_filename)
    # モデルの初期化
    model_container = {'normalizer': normalizer, 'models': {}}
    data = {}
    for item_id, item in enumerate(dataset.train()):
        #　抽出した特徴量の取得
        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
        feature_data = load_data(feature_filename)['feat']
        #　平均などを計算し掛け合わせる
        feature_data = model_container['normalizer'].normalize(feature_data)
        # クラスラベル単位で特徴量保存
        if item['scene_label'] not in data:
            data[item['scene_label']] = feature_data
        else:
            data[item['scene_label']] = np.vstack((data[item['scene_label']], feature_data))
    # それぞれのクラスごとにモデルを学習させる
    for label in data:
        model_container['models'][label] = mixture.GMM(n_components=16, covariance_type='diag', random_state=0, min_covar=0.001).fit(data[label])
    # モデルの保存
    save_data(current_model_file, model_container)

def do_system_testing(dataset, result_path, feature_path, model_path, feature_params):
    # システムの評価
    current_result_file = get_result_filename(path=result_path)
    results = []
    model_filename = get_model_filename(path=model_path)
    model_container = load_data(model_filename)
    for file_id, item in enumerate(dataset.test()):
        #特徴量ファイルの読み込み
        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
        feature_data = load_data(feature_filename)['feat']
        #特量量ファイルの正規化
        feature_data = model_container['normalizer'].normalize(feature_data)
        #GMMによる分類
        current_result = do_classification_gmm(feature_data, model_container)
        results.append((dataset.absolute_to_relative(item['file']), current_result))
        #CSVファイルへの結果書き込み
        with open(current_result_file, 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for result_item in results:
                writer.writerow(result_item)
def do_classification_gmm(feature_data, model_container):
    # マイナス無限大で対数尤度行列を初期化
    logls = np.empty(len(model_container['models']))
    logls.fill(-np.inf)
    for label_id, label in enumerate(model_container['models']):
        #与えられたデータごとの平均対数尤度を計算
        logls[label_id] = np.sum(model_container['models'][label].score(feature_data))
    #最も確率が高いものを記録する
    classification_result_id = np.argmax(logls)
    return list(model_container['models'].keys())[classification_result_id]

def do_system_evaluation(dataset, result_path):
    #評価用インスタンス作成
    metric = Metrics(class_list=dataset.scene_labels)
    results = []
    #結果保存ファイル指定
    result_filename = get_result_filename(path=result_path)
    with open(result_filename, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            results.append(row)
    y_true = []
    y_pred = []
    for result in results:
        y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
        y_pred.append(result[1])
    metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
    results = metric.results()
    print(results)
#'''
    print ("  File-wise evaluation")
    fold_labels = ''
    separator = '     =====================+======+======+==========+  +'
    separator += "==========+"
    print ("     {:20s} | {:4s} : {:4s} | {:8s} |  |".format('Scene label', 'Nref', 'Nsys', 'Accuracy','RESULTS'))
    print (separator)
    for label_id, label in enumerate(sorted(results['class_wise_accuracy'])):
        fold_values = ''
        fold_values += " {:5.1f} %  |".format(results['class_wise_accuracy'][label] * 100)
        print ("     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format(label,
                                                                     results['class_wise_data'][label]['Nref'],
                                                                     results['class_wise_data'][label]['Nsys'],
                                                                     results['class_wise_accuracy'][label] * 100)+fold_values)
    print (separator)
    fold_values = ''
    fold_values += " {:5.1f} %  |".format(results['overall_accuracy'] * 100)
    print ("     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format('Overall accuracy',
                                                                 results['Nref'],
                                                                 results['Nsys'],
                                                                 results['overall_accuracy'] * 100)+fold_values)
#'''
