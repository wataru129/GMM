#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
#from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from src.files import *
class Dataset(object):
    def __init__(self, data_path='data', name='dataset'):
        self.name = name
        self.local_path = os.path.join(data_path, self.name)
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        self.evaluation_setup_folder = 'evaluation_setup'
        self.evaluation_setup_path = os.path.join(self.local_path, self.evaluation_setup_folder)
        self.filelisthash_filename = 'filelist.python.hash'
        self.evaluation_folds = 1
        self.files = None
        self.evaluation_data_train = {}
        self.evaluation_data_test = {}
        self.audio_extensions = {'wav'}
    def test(self, fold=0):
        if fold not in self.evaluation_data_test:
            self.evaluation_data_test[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.evaluation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
            else:
                data = []
                files = []
                for item in self.meta:
                    if self.relative_to_absolute_path(item['file']) not in files:
                        data.append({'file': self.relative_to_absolute_path(item['file'])})
                        files.append(self.relative_to_absolute_path(item['file']))
                self.evaluation_data_test[fold] = data
        return self.evaluation_data_test[fold]
    def folds(self, mode='folds'):
        if mode == 'folds':
            return range(1, self.evaluation_folds + 1)
        elif mode == 'full':
            return [0]

    def relative_to_absolute_path(self, path):
        return os.path.abspath(os.path.join(self.local_path, path))
    def train(self, fold=0):
        if fold not in self.evaluation_data_train:
            self.evaluation_data_train[fold] = []
            if fold > 0:
                print(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt'))
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        if len(row) == 2:
                            # Scene meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1]
                            })
                        elif len(row) == 4:
                            # Audio tagging meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1],
                                'tag_string': row[2],
                                'tags': row[3].split(';')
                            })
                        elif len(row) == 5:
                            # Event meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1],
                                'event_onset': float(row[2]),
                                'event_offset': float(row[3]),
                                'event_label': row[4]
                            })
            else:
                data = []
                for item in self.meta:
                    if 'event_label' in item:
                        data.append({'file': self.relative_to_absolute_path(item['file']),
                                     'scene_label': item['scene_label'],
                                     'event_onset': item['event_onset'],
                                     'event_offset': item['event_offset'],
                                     'event_label': item['event_label'],
                                     })
                    else:
                        data.append({'file': self.relative_to_absolute_path(item['file']),
                                     'scene_label': item['scene_label']
                                     })
                self.evaluation_data_train[0] = data
        return self.evaluation_data_train[fold]
class DevelopmentSet(Dataset):
    def __init__(self, data_path='data'):
        Dataset.__init__(self, data_path=data_path, name='development')
        self.evaluation_folds = 4
    def on_after_extract(self):
        if not os.path.isfile(self.meta_file):
            section_header('Generating meta file for dataset')
            meta_data = {}
            for fold in xrange(1, self.evaluation_folds):
                # Read train files in
                train_filename = os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt')
                f = open(train_filename, 'rt')
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row[0] not in meta_data:
                        meta_data[row[0]] = row[1]
                f.close()
                # Read evaluation files in
                eval_filename = os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_evaluate.txt')
                f = open(eval_filename, 'rt')
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row[0] not in meta_data:
                        meta_data[row[0]] = row[1]
                f.close()

            f = open(self.meta_file, 'wt')
            try:
                writer = csv.writer(f, delimiter='\t')
                for file in meta_data:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)
                    label = meta_data[file]
                    writer.writerow((os.path.join(relative_path, raw_filename), label))
            finally:
                f.close()
            foot()

class EvaluationSet(Dataset):
    def __init__(self, data_path='data'):
        Dataset.__init__(self, data_path=data_path, name='evaluation')
        self.evaluation_folds = 1
    def on_after_extract(self):
        eval_filename = os.path.join(self.evaluation_setup_path, 'evaluate.txt')
        if not os.path.isfile(self.meta_file) and os.path.isfile(eval_filename):
            section_header('Generating meta file for dataset')
            meta_data = {}
            f = open(eval_filename, 'rt')
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row[0] not in meta_data:
                    meta_data[row[0]] = row[1]
            f.close()
            f = open(self.meta_file, 'wt')
            try:
                writer = csv.writer(f, delimiter='\t')
                for file in meta_data:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)
                    label = meta_data[file]
                    writer.writerow((os.path.join(relative_path, raw_filename), label))
            finally:
                f.close()
            foot()
    def test(self, fold=0):
        if fold not in self.evaluation_data_test:
            self.evaluation_data_test[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.evaluation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
            else:
                data = []
                files = []
                for item in self.audio_files:
                    if self.relative_to_absolute_path(item) not in files:
                        data.append({'file': self.relative_to_absolute_path(item)})
                        files.append(self.relative_to_absolute_path(item))

                self.evaluation_data_test[fold] = data

        return self.evaluation_data_test[fold]
