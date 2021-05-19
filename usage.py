#!/usr/bin/python3

from Text_Classifier import TextClassifier

tc = TextClassifier()
tc.loadDataset('./datasets/sample.csv')
tc.train()
