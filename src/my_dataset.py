#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from src.files import *
import csv
class Dataset(object):
    def __init__(self, data_path='data', name='dataset'):
        self.name = name
        #データセットの格納場所指定
        self.local_path = os.path.join(data_path, self.name)
        self.evaluation_setup_folder = 'evaluation_setup'
        self.evaluation_setup_path = os.path.join(self.local_path, self.evaluation_setup_folder)
        self.meta_file = os.path.join(self.local_path, self.evaluation_setup_folder, 'meta.txt')
        self.evaluation_folds = 1
        self.files = None
        self.evaluation_data_train = {}
        self.evaluation_data_test = {}
    @property
    def meta(self):
        self.meta_data = []
        meta_id = 0
        f = open(self.meta_file, 'rt')
        try:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                # Scene meta
                self.meta_data.append({'file': row[0], 'scene_label': row[1].rstrip()})
                meta_id += 1
        finally:
            f.close()
        return self.meta_data

    @property
    def scene_labels(self):
        labels = []
        for item in self.meta:
            if 'scene_label' in item and item['scene_label'] not in labels:
                labels.append(item['scene_label'])
        labels.sort()
        return labels
    def test(self, fold=0):
        with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
            for row in csv.reader(f, delimiter='\t'):
                self.evaluation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
        return self.evaluation_data_test[fold]
    def folds(self, mode='folds'):
        if mode == 'folds':
            return range(1, self.evaluation_folds + 1)
        elif mode == 'full':
            return [0]
    def relative_to_absolute_path(self, path):
        return os.path.abspath(os.path.join(self.local_path, path))
    def train(self):
        self.evaluation_data_train = []
        print(os.path.join(self.evaluation_setup_path,  'train.csv'))
        #ファイル名と正解ラベルのついたcsvファイルを読み込む
        csv_file = open(os.path.join(self.evaluation_setup_path,  'train.csv'), "r", encoding="ms932", errors="", newline="" )
        f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        next(f)
        for row in f:
            self.evaluation_data_train.append({
                'file': self.relative_to_absolute_path(row[1]),
                'scene_label': row[2]
            })
        return self.evaluation_data_train
class DevelopmentSet(Dataset):
    def __init__(self, data_path='data'):
        #トレーニング用データセットのインスタンス作成
        Dataset.__init__(self, data_path=data_path, name='development')
class EvaluationSet(Dataset):
    def __init__(self, data_path='data'):
        #評価用データセットのインスタンス作成
        Dataset.__init__(self, data_path=data_path, name='evaluation')
    def test(self):
        #評価用データの格納場所を記録する.
        print(os.path.join(self.evaluation_setup_path,  'test.csv'))
        self.evaluation_data_test = []
        csv_file = open(os.path.join(self.evaluation_setup_path,  'test.csv'), "r", encoding="ms932", errors="", newline="" )
        f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        next(f)
        for row in f:
            self.evaluation_data_test.append({
                'file': self.relative_to_absolute_path(row[1])
            })
        return self.evaluation_data_test
