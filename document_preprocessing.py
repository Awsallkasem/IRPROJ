# -*- coding: utf-8 -*-

import csv
import os
import re
import glob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from dateutil.parser import parse
from datetime import datetime

# Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# folder_name = "documents"
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)

# with open(r'antique-collection.tsv', "r") as f:
#     reader = csv.reader(f, delimiter="\t")
#     for row in reader:
#         file_id = row[0]
#         content = row[1]


#         file_path = os.path.join(folder_name, f"{file_id}.txt")
#         with open(file_path, "w") as outfile:
#             outfile.write(content)

# Define abbreviations and contractions
abbreviations = {
    'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Misess', 'Ms.': 'Misess',
    'Jr.': 'Junior', 'Sr.': 'Senior', 'U.S': 'UNITED STATES', 'U-S': 'UNITED STATES',
    'U_K': 'UNITED KINGDOM', 'U_S': 'UNITED STATES', 'U.K': 'UNITED KINGDOM',
    'U.S': 'UNITED STATES', 'VIETNAM': 'VIET NAM', 'VIET NAM': 'VIET NAM',
    'U-N': 'NITED NATIONS', 'U_N': 'NITED NATIONS', 'U.N': 'NITED NATIONS',
    'UK': 'UNITED KINGDOM', 'US': 'UNITED STATES', 'U-K': 'UNITED KINGDOM',
    'mar': 'March', 'jan': 'January', 'feb': 'February', 'apr': 'April',
    'jun': 'June', 'jul': 'July', 'dec': 'December', 'nov': 'November',
    'oct': 'October', 'sep': 'September', 'aug': 'August'
}

contractions_dict = {
    "n't": " not", "'s": " is", "'m": " am", "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
}

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_words(txt):
    return [stemmer.stem(word) for word in txt]

def lemmatize_words(txt):
    return [lemmatizer.lemmatize(word, pos='v') for word in txt]

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

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
        try:
            date = datetime.strptime(s[0], "%d %B %Y")
        except ValueError:
            continue

        new_date = date.strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)

    text = re.sub(r'[^-\w\s]', '', text)

    return text

def preprocess_text(text):
    # Tokenize words
    tokens = word_tokenize(text)

    # Expand contractions
    expanded_tokens = []
    for token in tokens:
        if token in contractions_dict:
            expanded_token = contractions_dict[token]
            expanded_tokens.append(expanded_token)
        else:
            expanded_tokens.append(token)

    expanded_text = " ".join(expanded_tokens)
    tokens = word_tokenize(expanded_text)

    # Convert to lowercase
    tokens = [w.lower() for w in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]

    # Stemming and POS tagging for better lemmatization
    stemmed_tokens = stem_words(filtered_tokens)
    pos_tagged = nltk.pos_tag(stemmed_tokens)
    wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    return lemmatized_sentence

def process_documents(docs_folder):
    file_paths = glob.glob(docs_folder + "/*.txt")
    new_dir = r"C:\Users\user\Documents\IRSystem1\new_docs"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    processed_count = 0
    error_count = 0

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                text = file.read()
                print(f"Processing file: {file_path}")
                print("-------------------")
                print("Original text:", text)
                print("-------------------")

                unified_text = handle(text)
                print("File content after unification:")
                print(unified_text)

                preprocessed_text = preprocess_text(unified_text)
                print("-------------------")
                print("File content after pre-processing:")
                print(preprocessed_text)

                new_file_path = os.path.join(new_dir, os.path.basename(file_path))
                lemmas_str = ' '.join(preprocessed_text)
                with open(new_file_path, "w", encoding='utf-8') as new_file:
                    new_file.write(lemmas_str)
                print("Preprocessed text saved to:", new_file_path)

                processed_count += 1
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")
            error_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1

    print(f"Total processed files: {processed_count}")
    print(f"Total errors: {error_count}")

# Example usage
# docs_folder = r"C:\Users\user\Documents\IRSystem1\documents"
# process_documents(docs_folder)