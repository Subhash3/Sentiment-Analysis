import re
from helpers import stopWords


class PreProcess:
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

        return re.sub('[^a-zA-Z0-9 \n\'\"]', '', sentence)

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

    def removeStopWords(self, words: list):
        """
            Removes the stop words from the given list of words

            Attributes
            ----------
            words:list
                List of tokens

            Returns
            -------
            list
                List of given words but with no stop words.
        """
        return [word for word in words if word not in stopWords]

    def processString(self, sentence: str):
        """
            Applies the pre-processing steps to a given strinf

            Attributes
            ----------
            sentence: str

            Returns
            -------
            tokens: list
                A list of processed tokens
        """

        sentence = self.removeSpecialChars(sentence)
        tokens = self.tokenize(sentence)
        tokens = self.removeStopWords(tokens)

        return tokens

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
            tokens = self.processString(sentence)
            if len(tokens) == []:
                continue
            processed.append({
                "tokens": tokens,
                "category": sample["category"]
            })

        return processed
