########################################################
#####################データセットを用意#####################
#########################################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import socket
import locale
import zipfile
import tarfile
from sklearn.cross_validation import StratifiedShuffleSplit, KFold


from src.files import *


class Dataset(object):
    """Dataset base class.

    The specific dataset classes are inherited from this class, and only needed methods are reimplemented.

    """

    def __init__(self, data_path='data', name='dataset'):
        self.name = name
        self.local_path = os.path.join(data_path, self.name)
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        self.evaluation_setup_folder = 'evaluation_setup'
        self.evaluation_setup_path = os.path.join(self.local_path, self.evaluation_setup_folder)
        self.meta_filename = 'meta.txt'
        self.meta_file = os.path.join(self.local_path, self.meta_filename)
        self.error_meta_filename = 'error.txt'
        self.error_meta_file = os.path.join(self.local_path, self.error_meta_filename)
        self.filelisthash_filename = 'filelist.python.hash'
        self.evaluation_folds = 1
        self.package_list = []
        self.files = None
        self.meta_data = None
        self.error_meta_data = None
        self.evaluation_data_train = {}
        self.evaluation_data_test = {}
        self.audio_extensions = {'wav', 'flac'}
        self.authors = ''
        self.name_remote = ''
        self.url = ''
        self.audio_source = ''
        self.audio_type = ''
        self.recording_device_model = ''
        self.microphone_model = ''

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


# =====================================================
# DCASE 2016
# =====================================================
class TUTAcousticScenes_2016_DevelopmentSet(Dataset):
    def __init__(self, data_path='data'):
        Dataset.__init__(self, data_path=data_path, name='TUT-acoustic-scenes-2016-development')

        self.authors = 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen'
        self.name_remote = 'TUT Acoustic Scenes 2016, development dataset'
        self.url = 'https://zenodo.org/record/45739'
        self.audio_source = 'Field recording'
        self.audio_type = 'Natural'
        self.recording_device_model = 'Roland Edirol R-09'
        self.microphone_model = 'Soundman OKM II Klassik/studio A3 electret microphone'

        self.evaluation_folds = 4

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.error.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.error.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.4.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.4.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.5.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.5.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.6.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.6.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.7.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.7.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.8.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-development.audio.8.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    def on_after_extract(self):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

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

class TUTAcousticScenes_2016_EvaluationSet(Dataset):
    """TUT Acoustic scenes 2016 evaluation dataset

    This dataset is used in DCASE2016 - Task 1, Acoustic scene classification

    """

    def __init__(self, data_path='data'):
        Dataset.__init__(self, data_path=data_path, name='TUT-acoustic-scenes-2016-evaluation')

        self.authors = 'Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen'
        self.name_remote = 'TUT Acoustic Scenes 2016, evaluation dataset'
        self.url = 'http://www.cs.tut.fi/sgn/arg/dcase2016/download/'
        self.audio_source = 'Field recording'
        self.audio_type = 'Natural'
        self.recording_device_model = 'Roland Edirol R-09'
        self.microphone_model = 'Soundman OKM II Klassik/studio A3 electret microphone'

        self.evaluation_folds = 1

        self.package_list = [
            {
                'remote_package': None,
                'local_package': None,
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.doc.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.doc.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.1.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.1.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.2.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.2.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.3.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.audio.3.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            },
            {
                'remote_package': 'https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip',
                'local_package': os.path.join(self.local_path, 'TUT-acoustic-scenes-2016-evaluation.meta.zip'),
                'local_audio_path': os.path.join(self.local_path, 'audio'),
            }
        ]

    def on_after_extract(self):
        """After dataset packages are downloaded and extracted, meta-files are checked.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

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

    def train(self, fold=0):
        raise IOError('Train setup not available.')

    def test(self, fold=0):
        """List of testing items.

        Parameters
        ----------
        fold : int > 0 [scalar]
            Fold id, if zero all meta data is returned.
            (Default value=0)

        Returns
        -------
        list : list of dicts
            List containing all meta data assigned to testing set for given fold.

        """

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
