import pandas as pd
import json
from helpers import customArgmax, shuffleArray, splitDataframe, splitArr
from Pre_Processor import PreProcess
import sys
from custom_exceptions import RequiredFieldsNotFoundError


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
            data = pd.read_csv(csvFile, encoding="ISO-8859-1")

            if ("sentence" not in data.columns) or ("category" not in data.columns):
                raise RequiredFieldsNotFoundError

            processedData = self.preProcessor.preProcess(data)
            self.dataset = processedData

            self.training, self.testing = splitDataframe(self.dataset, 3/4)

            self.noOfSamples = self.training.shape[0]
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

        self.summaryByClass = self._describeByClass(self.training)
        print(f"Trained {self.training.shape[0]} samples")

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

        categories = set(self.training["category"])

        probabilities = dict()
        for category in categories:
            samples = self.training[self.training["category"] == category]
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
        """
            Predicts the category of the given tokens of a sentence.

            Attributes
            ----------
            tokens: List[str]

            Returns
            -------
            Tuple[str, Dict[str, float]]
                Tuple containing the predicted category and probabilities of all categories.
        """
        probabilities = self.computeProbabilities(tokens)
        # print(probabilities)
        return customArgmax(probabilities), probabilities

    def Test(self):
        """
            Tests the model agaist the part of a dataset and compute the accuracy.

            Attributes
            ----------

            Returns
            -------
            accuracy: float
        """
        correct = 0
        total = 0
        total = self.testing.shape[0]

        if total <= 0:
            return

        testingProgress = 0

        testedSamples = 0
        nonEmptyTokens = 0
        tokensColumn = self.testing["tokens"]
        # print(self.testing.head(10))
        # print(self.testing.index.values)
        for i in self.testing.index.values:
            testingProgress = (testedSamples*100)/total
            print(
                f"Testing {round(testingProgress, 3)}% done {'.-'*(int(testingProgress/5)+1)}", end='\r')
            sys.stdout.flush()
            tokens = tokensColumn[i]
            # print(tokens)
            if len(tokens) == 0:
                continue

            prediction = self.predictByTokens(tokens)
            if prediction[0] == self.testing["category"][i]:
                correct += 1
            nonEmptyTokens += 1
            testedSamples += 1
        print()
        print(f"Tested: {total} samples.")

        # print(correct, total, nonEmptyTokens)
        accuracy = max(correct*100/total, correct*100/nonEmptyTokens)
        print(f"Accuracy: {accuracy}")
        return accuracy
