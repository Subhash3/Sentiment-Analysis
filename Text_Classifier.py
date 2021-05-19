import pandas as pd
import re
import json


class TextClassifier:
    """
        Class to classify text using Naive Bayes algorithm

        Attributes
        ----------
        dataset: pandas.DataFrame

    """

    def __init__(self):
        self.dataset = pd.DataFrame()
        self.summaryByClass = dict()
        self.noOfSamples = 0
        self.noOfFeatures = 0

    def removeSpecialChars(self, sentence: str):
        """
            Removes special characters from a given string

            Attributes
            ----------
            sentence: str

            Returns
            -------
            str
                A new string without special characters.
        """

        return re.sub('[^a-zA-Z0-9 \n]', '', sentence)

    def tokenize(self, sentence: str):
        """
            Tokenizes the given sentence.

            Attributes
            ----------
            sentence: str

            Returns
            -------
            list
                A list of words(tokens).
        """
        return [token.strip() for token in sentence.lower().split()]

    def preProcess(self, data):
        """
            Applies several pre-processing steps such as tokenization, stemming ...etc to the data.

            Attributes
            ----------
            data: pd.DataFrame
                A pandas dataframe with sentence and category.

            Returns
            -------
            data: pd.DataFrame
                Preprocessed data
        """

        processed = list()
        for sample in data:
            sentence = sample["sentence"]
            sentence = self.removeSpecialChars(sentence)
            tokens = self.tokenize(sentence)
            processed.append({
                "tokens": tokens,
                "category": sample["category"]
            })

        return processed

    def loadDataset(self, jsonFile):
        """
            Loads the data from a json file into a pandas DataFrame

            Parameters
            ----------
            jsonFile: str
                Csv file containing the dataset

            Returns
            -------
            None
                if the dataset is loaded successfully.
            Error: Exception
                if error occurs.
        """

        try:
            data = json.load(open(jsonFile))
            self.dataset = self.preProcess(data)
        except Exception as e:
            raise Exception

    def _describeByClass(self, dataset):
        """
            Separates data by classname and computes the mean and std of each feature in each class.

            Parameters
            ----------
            dataset: pd.DataFrame
                Dataframe of features and class values

            Returns
            -------
            summary: Dict[str: List[mean, std]]
                Map from class to a list of mean and std values of each feature.
        """
        categories = set(dataset["class"])
        features = dataset.columns[:-1]

        summary = dict()

        for category in categories:
            samples = dataset[dataset["class"] == category]
            # print(samples.describe())

            summaryOfThisCategory = list()
            for f in features:
                mean = samples[f].describe().get("mean")
                std = samples[f].describe().get("std")

                summaryOfThisCategory.append({
                    "mean": mean,
                    "std": std
                })
            summary[category] = summaryOfThisCategory
        return summary

    def train(self):
        """
            Computes the mean and std of each feature in each class and stores the results in self.summaryByClass
        """

        self.summaryByClass = self._describeByClass(self.dataset)

        for category in self.summaryByClass:
            print(category)
            for i in range(self.noOfFeatures):
                print(f"\tfeature-{i+1}: {self.summaryByClass[category][i]}")
