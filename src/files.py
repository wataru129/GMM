#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wave
import numpy
import csv
#import cPickle as pickle
import _pickle as pickle
import librosa
import yaml
import soundfile

def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        # Load audio
        audio_data, sample_rate = soundfile.read(filename)
        audio_data = audio_data.T

        if mono:
            # Down-mix audio
            audio_data = numpy.mean(audio_data, axis=0)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate

    elif file_extension == '.flac':
        audio_data, sample_rate = librosa.load(filename, sr=fs, mono=mono)

        return audio_data, sample_rate

    return None, None


def load_event_list(file):
    data = []
    with open(file, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) == 2:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1])
                    }
                )
            elif len(row) == 3:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1]),
                        'event_label': row[2]
                    }
                )
            elif len(row) == 4:
                data.append(
                    {
                        'file': row[0],
                        'event_onset': float(row[1]),
                        'event_offset': float(row[2]),
                        'event_label': row[3]
                    }
                )
            elif len(row) == 5:
                data.append(
                    {
                        'file': row[0],
                        'scene_label': row[1],
                        'event_onset': float(row[2]),
                        'event_offset': float(row[3]),
                        'event_label': row[4]
                    }
                )
    return data


def save_data(filename, data):
    pickle.dump(data, open(filename, 'wb'))#, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    return pickle.load(open(filename, "rb"))


def save_parameters(filename, parameters):
    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(parameters, default_flow_style=False))


def load_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return yaml.load(f)
    else:
        raise IOError("Parameter file not found [%s]" % filename)


def save_text(filename, text):
    with open(filename, "w") as text_file:
        text_file.write(text)


def load_text(filename):
    with open(filename, 'r') as f:
        return f.readlines()
