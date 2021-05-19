#!/usr/bin/python3

from Text_Classifier import TextClassifier

tc = TextClassifier()
tc.loadDataset('./train_short.json')
tc.train()
sentence = "good"

prediction = tc.predict(sentence)
print(prediction)
