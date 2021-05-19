from typing import Dict
import random
from math import floor

stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
             "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def customArgmax(data: Dict[str, float]):
    maxKey = None
    maxValue = None

    for key in data:
        if maxKey == None:
            maxKey = key
        if maxValue == None or maxValue < data[key]:
            maxValue = data[key]
            maxKey = key

    return maxKey

def shuffleArray(array: list):
    arrayCopy = array.copy()
    random.shuffle(arrayCopy)

    return arrayCopy


def splitArr(array: list, ratio: float):
    n = len(array)

    m = floor(n * ratio)

    firstPart: list = array[0: m]
    secondPart: list = array[m: n]

    return [firstPart, secondPart]
