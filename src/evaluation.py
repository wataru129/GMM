#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy
import math
from sklearn import metrics

class Metrics():
    def __init__(self, class_list):
        self.accuracies_per_class = None
        self.correct_per_class = None
        self.Nsys = None
        self.Nref = None
        self.class_list = class_list
        self.eps = numpy.spacing(1)
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        return self.results()
    def accuracies(self, y_true, y_pred, labels):
        confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels).astype(float)
        return (numpy.diag(confusion_matrix), numpy.divide(numpy.diag(confusion_matrix), numpy.sum(confusion_matrix, 1)+self.eps))
    def evaluate(self, annotated_ground_truth, system_output):
        correct_per_class, accuracies_per_class = self.accuracies(y_pred=system_output, y_true=annotated_ground_truth, labels=self.class_list)
        if self.accuracies_per_class is None:
            self.accuracies_per_class = accuracies_per_class
        else:
            self.accuracies_per_class = numpy.vstack((self.accuracies_per_class, accuracies_per_class))

        if self.correct_per_class is None:
            self.correct_per_class = correct_per_class
        else:
            self.correct_per_class = numpy.vstack((self.correct_per_class, correct_per_class))
        Nref = numpy.zeros(len(self.class_list))
        Nsys = numpy.zeros(len(self.class_list))
        for class_id, class_label in enumerate(self.class_list):
            for item in system_output:
                if item == class_label:
                    Nsys[class_id] += 1

            for item in annotated_ground_truth:
                if item == class_label:
                    Nref[class_id] += 1
        if self.Nref is None:
            self.Nref = Nref
        else:
            self.Nref = numpy.vstack((self.Nref, Nref))

        if self.Nsys is None:
            self.Nsys = Nsys
        else:
            self.Nsys = numpy.vstack((self.Nsys, Nsys))
    def results(self):

        results = {
            'class_wise_data': {},
            'class_wise_accuracy': {},
            'overall_accuracy': float(numpy.mean(self.accuracies_per_class)),
            'class_wise_correct_count': self.correct_per_class.tolist(),
        }
        if len(self.Nsys.shape) == 2:
            results['Nsys'] = int(sum(sum(self.Nsys)))
            results['Nref'] = int(sum(sum(self.Nref)))
        else:
            results['Nsys'] = int(sum(self.Nsys))
            results['Nref'] = int(sum(self.Nref))

        for class_id, class_label in enumerate(self.class_list):
            if len(self.accuracies_per_class.shape) == 2:
                results['class_wise_accuracy'][class_label] = numpy.mean(self.accuracies_per_class[:, class_id])
                results['class_wise_data'][class_label] = {
                   'Nsys': int(sum(self.Nsys[:, class_id])),
                    'Nref': int(sum(self.Nref[:, class_id])),
                }
            else:
                results['class_wise_accuracy'][class_label] = numpy.mean(self.accuracies_per_class[class_id])
                results['class_wise_data'][class_label] = {
                   'Nsys': int(self.Nsys[class_id]),
                    'Nref': int(self.Nref[class_id]),
                }
        return results
