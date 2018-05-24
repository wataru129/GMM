#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from src.files import *
import csv
class Dataset(object):
    def __init__(self, data_path='data', name='dataset'):
        self.name = name
        self.local_path = os.path.join(data_path, self.name)
        self.evaluation_setup_folder = 'evaluation_setup'
        self.evaluation_setup_path = os.path.join(self.local_path, self.evaluation_setup_folder)
        self.filelisthash_filename = 'filelist.python.hash'
        self.evaluation_folds = 1
        self.files = None
        self.evaluation_data_train = {}
        self.evaluation_data_test = {}
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
        Dataset.__init__(self, data_path=data_path, name='development')
class EvaluationSet(Dataset):
    def __init__(self, data_path='data'):
        Dataset.__init__(self, data_path=data_path, name='evaluation')
    def test(self, fold=0):
        if fold not in self.evaluation_data_test:
            self.evaluation_data_test[fold] = []
            with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    self.evaluation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
        return self.evaluation_data_test[fold]
