import pandas as pd
import re
import json
from helpers import customArgmax, shuffleArray, splitArr
from Pre_Processor import PreProcess


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
        self.DEFAULT_PROBABILITY = 1
        self.preProcessor = PreProcess()

    def loadDatasetJson(self, jsonFile):
        """
            Loads the data from a json file into a pandas DataFrame

            Parameters
            ----------
            jsonFile: str
                Json file containing the dataset, an array of data-samples with the following structure:
                {
                    "sentence": "some sentence",
                    "category": "positive/negative"
                }

            Returns
            -------
            None
                if the dataset is loaded successfully.
            Error: Exception
                if error occurs.
        """

        try:
            data = json.load(open(jsonFile))
            data = self.preProcessor.preProcess(data)

            train, test = splitArr(data, 4/5)
            self.testing = test
            self.training = train

            data = shuffleArray(train)
            self.dataset = pd.DataFrame(data)
            self.noOfSamples = self.dataset.shape[0]
        except Exception as e:
            raise e

    def loadDatasetCsv(self, csvFile):
        """
            Loads the data from a json file into a pandas DataFrame

            Parameters
            ----------
            jsonFile: str
                Json file containing the dataset, an array of data-samples with the following structure:
                {
                    "sentence": "some sentence",
                    "category": "positive/negative"
                }

            Returns
            -------
            None
                if the dataset is loaded successfully.
            Error: Exception
                if error occurs.
        """

        try:
            data = pd.read_csv(csvFile)
            del data["ItemID"]
            data.columns = ["category", "sentence"]
            # print(data)
            processedData = self.preProcessor.preProcess(data)
            self.dataset = processedData

            # data = shuffleArray(data)
            # self.dataset = pd.DataFrame(data)
            print(processedData)
            self.noOfSamples = self.dataset.shape[0]
        except Exception as e:
            raise e

    def _describeByClass(self, dataset: pd.DataFrame):
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
        categories = set(dataset["category"])

        summary = dict()

        for category in categories:
            samples = dataset[dataset["category"] == category]
            # print(samples)
            samplesCount = samples.shape[0]

            tokenProbabilities = dict()
            for tokenList in samples["tokens"]:
                # print(tokenList)
                for token in tokenList:
                    if token not in tokenProbabilities:
                        tokenProbabilities[token] = 0
                    else:
                        tokenProbabilities[token] += 1

            for token in tokenProbabilities:
                tokenProbabilities[token] /= samplesCount

            summary[category] = tokenProbabilities

        for category in categories:
            s = summary[category]
            for token in s:
                s[token] += self.DEFAULT_PROBABILITY

            # print()
        # print(summary)
        return summary

    def train(self):
        """
            Computes the mean and std of each feature in each class and stores the results in self.summaryByClass
        """

        self.summaryByClass = self._describeByClass(self.dataset)

    def computeProbabilities(self, tokens: list):
        """
            Computes the probability that the given tokens belong to each class.

            Attributes
            ----------
            tokens: list
                List of processed tokens

            Returns
            -------
            probabilities: Dict[str, float]
                probability of each class/category.
        """

        if len(tokens) == 0:
            return None

        categories = set(self.dataset["category"])

        probabilities = dict()
        for category in categories:
            samples = self.dataset[self.dataset["category"] == category]
            samplesCount = samples.shape[0]
            priorProbability = samplesCount/self.noOfSamples

            likelihood = 1
            for token in tokens:
                if token in self.summaryByClass[category]:
                    p = self.summaryByClass[category][token]
                else:
                    p = 1
                # print(category, token, p, priorProbability)
                likelihood *= p
            probabilities[category] = p * priorProbability
            # print(probabilities[category])
        return probabilities

    def predict(self, sentence: str):
        """
            Predicts the category of the given sentence.

            Attributes
            ----------
            sentence: str

            Returns
            -------
            Tuple[str, Dict[str, float]]
                Tuple containing the predicted category and probabilities of all categories.
        """

        tokens = self.preProcessor.processString(sentence)
        # print(self.summaryByClass)
        probabilities = self.computeProbabilities(tokens)
        # print(probabilities)
        if probabilities == None:
            return None
        return customArgmax(probabilities), probabilities

    def predictByTokens(self, tokens):
        probabilities = self.computeProbabilities(tokens)
        # print(probabilities)
        return customArgmax(probabilities), probabilities

    def Test(self):
        correct = 0
        total = 0
        total = len(self.testing)
        if total <= 0:
            return
        for i in range(total):
            sample = self.testing[i]
            try:
                prediction = self.predictByTokens(sample["tokens"])
                if prediction == None:
                    continue
                if prediction[0] == sample["category"]:
                    correct += 1
            except Exception as e:
                print(e)
                print(sample)
                # quit()
        print(correct*100/total)
        print(correct, total)
