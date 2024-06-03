# processquery.py

from autocorrect import Speller
import pandas as pd
import os
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from process import handle
import nltk
from nltk.corpus import wordnet
import datefinder
import re
import re
from typing import Any
from datetime import datetime
from dateutil.parser import parse

# Ensure you have downloaded the required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def stem_words(txt):
    return [stemmer.stem(word) for word in txt]


def process(query):
    text = handle(query)
    query = text
    # Initialize Speller
    spell = Speller(lang='en')

    # Correct spelling
    corrected_sentence = " ".join([spell(word) for word in query.split()])

    # Tokenize the sentence
    tokens = word_tokenize(corrected_sentence)
    tokens = [w.lower() for w in tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in tokens if w not in stop_words]

    # Stem words
    stemmed_words = stem_words(words)

    # Lemmatize words
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in stemmed_words]

    # Join lemmatized words into a sentence
    lemmatized_sentence = " ".join(lemmas)

    return lemmatized_sentence
