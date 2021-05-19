import pandas as pd
import re


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

    def loadDataset(self, csvFile):
        """
            Loads the data from a csv file into a pandas DataFrame

            Parameters
            ----------
            csvFile: str
                Csv file containing the dataset

            Returns
            -------
            None
                if the dataset is loaded successfully.
            Error: Exception
                if error occurs.
        """

        try:
            data = pd.DataFrame(pd.read_csv(csvFile))
            self.dataset = data
            self.noOfSamples = data.shape[0]
            self.noOfFeatures = data.shape[1] - 1
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
        self.summaryByClass = self._describeByClass(self.dataset)

        for category in self.summaryByClass:
            print(category)
            for i in range(self.noOfFeatures):
                print(f"\tfeature-{i+1}: {self.summaryByClass[category][i]}")
