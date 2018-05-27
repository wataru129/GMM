#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import wave
import numpy
import csv
import _pickle as pickle
import librosa
import yaml
import soundfile

def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    # 音データを読み込む
    audio_data, sample_rate = soundfile.read(filename)
    audio_data = audio_data.T
    if mono:
        # Down-mix audio
        audio_data = numpy.mean(audio_data, axis=0)
    # リサンプリングを行う
    if fs != sample_rate:
        audio_data = librosa.core.resample(audio_data, sample_rate, fs)
        sample_rate = fs
    return audio_data, sample_rate
#pickleモジュールを使ってデータを保存
def save_data(filename, data):
    pickle.dump(data, open(filename, 'wb'))
#pickleモジュールの読み込み
def load_data(filename):
    return pickle.load(open(filename, "rb"))
#現在のパラメータを保存
def save_parameters(filename, parameters):
    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(parameters, default_flow_style=False))
#現在のパラメータを保存
def load_parameters(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)
def save_text(filename, text):
    with open(filename, "w") as text_file:
        text_file.write(text)
def load_text(filename):
    with open(filename, 'r') as f:
        return f.readlines()
