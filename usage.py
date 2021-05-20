#!/usr/bin/python3

from Text_Classifier import TextClassifier

tc = TextClassifier()
tc.loadDatasetCsv('./datasets/Gunjan933-github/train_short.csv')
tc.train()
# for category in tc.summaryByClass:
#     for word in tc.summaryByClass[category]:
#         print('\t', word, tc.summaryByClass[category][word])
#     print()
# for category in tc.summaryByClass:
#     print(category)
#     print(len(tc.summaryByClass[category]))

tests = ["feeling fine",
         "I hate you",
         "I love india!!!"]

# for sentence in tests:
#     prediction = tc.predict(sentence)
#     print(prediction)
#     print()

tc.Test()
