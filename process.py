# -*- coding: utf-8 -*-


import os
import re
import csv
import shutil
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from dateutil.parser import parse
import glob

abbreviations = {
    'Dr.': 'Doctor',
    'Mr.': 'Mister',
    'Mrs.': 'Misess',
    'Ms.': 'Misess',
    'Jr.': 'Junior',
    'Sr.': 'Senior',
    'U.S': 'UNITED STATES',
    'U-S': 'UNITED STATES',
    'U_K': 'UNITED KINGDOM',
    'U_S': 'UNITED STATES',
    'U.K': 'UNITED KINGDOM',
    'U.S': 'UNITED STATES',
    'VIETNAM': 'VIET NAM',
    'VIET NAM': 'VIET NAM',
    'U-N': 'NITED NATIONS',
    'U_N': 'NITED NATIONS',
    'U.N': 'NITED NATIONS',
    'UK': 'UNITED KINGDOM',
    'US': 'UNITED STATES',
    'U-K': 'UNITED KINGDOM',
    'mar': 'March',
    'jan': 'January',
    'feb': 'February',
    'apr': 'April',
    'jun': 'June',
    'jul': 'July',
    'dec': 'December',
    'nov': 'November',
    'oct': 'October',
    'sep': 'September',
    'aug': 'August',
}
contractions_dict = {
    "n't": " not",
    "'s": " is",
    "'m": " am",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would"
}

# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
#
# def stem_words(txt):
#     stems = [stemmer.stem(word) for word in txt]
#     return stems
#
# def lemmatize_words(txt):
#     lemmas = [lemmatizer.lemmatize(word, pos='v') for word in txt]
#     return lemmas

def handle(text):
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                        REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    for key, value in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text, flags=re.IGNORECASE)
        all_dates = re.findall(COMBINATION_REGEX, text)

    for s in all_dates:
        new_date = parse(s[0]).strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)

    # Remove punctuation marks except for hyphen "-"
    text = re.sub(r'[^-\w\s]', '', text)

    return text

# def preprocess_text(text):
#     # Tokenize sentences
#     sentences = sent_tokenize(text)
#     # Tokenize words
#     tokens = word_tokenize(text)
#
#     # Expand contractions
#     expanded_tokens = []
#     for token in tokens:
#         if token in contractions_dict:
#             expanded_token = contractions_dict[token]
#             expanded_tokens.append(expanded_token)
#         else:
#             expanded_tokens.append(token)
#
#     expanded_text = " ".join(expanded_tokens)
#     tokens = word_tokenize(expanded_text)
#
#     # Convert to lowercase
#     tokens = [w.lower() for w in tokens]
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     # filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
#     filtered_tokens = [w for w in tokens if not w in stop_words]
#
#     # Lemmatize tokens
#     lemmatized_tokens = lemmatize_words(filtered_tokens)
#
#     # Stem tokens
#     stemmed_tokens = stem_words(lemmatized_tokens)
#
#     # stem_word = stem_words(filtered_tokens)
#     # lemmas = lemmatize_words(stem_word)
#
#     # Join tokens back into a string
#     preprocessed_text = " ".join(stemmed_tokens)
#
#     # return lemmas
#     return preprocessed_text
#
# # sample_text = "Dr. John visited the U.S on 12/25/2021. He will return on February 5, 2022. Happy New Year's!"
#
# # processed_text = handle(sample_text)
# # print(processed_text)
#
# # sample_text = "It's a lovely day. He'll go to the park."
#
# # processed_text = handle(sample_text)
# # print(processed_text)
#
# def process_documents(docs_folder):
#     file_paths = glob.glob(docs_folder + "/*.txt")
#
#     # Create the new directory if it doesn't exist
#     new_dir = r"C:\Users\user\Documents\IRSystem1\new_docs"
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)
#
#     for file_path in file_paths:
#         with open(file_path, "r", encoding='ISO-8859-1') as file:
#             text = file.read()
#             print("-------------------")
#             print("Original text:", text)
#
#             print("-------------------")
#             unified_text = handle(text)
#             print("File content after unification:")
#             print(unified_text)
#
#             preprocessed_text = preprocess_text(unified_text)
#             print("-------------------")
#             print("File content after pre-processing:")
#             print(preprocessed_text)
#
#             new_file_path = os.path.join(new_dir, os.path.basename(file_path))
#
#             # Convert the list of lemmas back to a string
#             lemmas_str = ' '.join(preprocessed_text)
#
#             # Save the preprocessed text to the new file
#             with open(new_file_path, "w", encoding='ISO-8859-1') as new_file:
#                 new_file.write(lemmas_str)
#
#             print("Preprocessed text saved to:", new_file_path)
#
# # Example usage
# docs_folder = r"C:\Users\user\Documents\IRSystem1\documents"
# process_documents(docs_folder)

