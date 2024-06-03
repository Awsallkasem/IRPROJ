# -*- coding: utf-8 -*-

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

docs = {}

def read_docs_from_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                doc_text = file.read()
            doc_id = os.path.splitext(file_name)[0]  # Use the file name without extension as the document ID
            docs[doc_id] = doc_text


def calculate_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    return tfidf_matrix, vectorizer

def save_tfidf_matrix(tfidf_matrix, vectorizer, file_path):
    with open(file_path, 'wb') as tfidf_file:
        pickle.dump((tfidf_matrix, vectorizer), tfidf_file)

def build_inverted_index(tfidf_matrix, vectorizer, docs):
    inverted_index = {}
    feature_names = vectorizer.get_feature_names_out()

    for doc_id, doc_text in docs.items():
        doc_idx = list(docs.keys()).index(doc_id)
        feature_values = tfidf_matrix[doc_idx].toarray().flatten()
        for feature_idx, tfidf_value in enumerate(feature_values):
            if tfidf_value > 0:
                feature_name = feature_names[feature_idx]
                if feature_name in inverted_index:
                    inverted_index[feature_name][doc_id] = tfidf_value
                else:
                    inverted_index[feature_name] = {doc_id: tfidf_value}
    return inverted_index

# Specify the folder path where preprocessed documents are stored
folder_path = r"C:\Users\user\Documents\IRSystem1\new_docs"

# Read documents from the folder
read_docs_from_folder(folder_path)

# Print number of documents loaded
print(f"Loaded {len(docs)} documents.")

# Calculate TF-IDF matrix
tfidf_matrix, vectorizer = calculate_tfidf_matrix(docs)

# Build inverted index
inverted_index = build_inverted_index(tfidf_matrix, vectorizer, docs)

# Print a sample of the inverted index
print("Sample of inverted index:")
for term, doc_dict in list(inverted_index.items())[:5]:
    print(f"{term}: {doc_dict}")

# Save the TF-IDF matrix and vectorizer to a file
tfidf_file_path = r"C:\Users\user\Documents\IRSystem1\tfidf_matrix_new.pkl"
save_tfidf_matrix(tfidf_matrix, vectorizer, tfidf_file_path)

# Save the inverted index to a file
inverted_index_file_path = r"C:\Users\user\Documents\IRSystem1\inverted_index_new.pkl"
with open(inverted_index_file_path, 'wb') as file:
    pickle.dump(inverted_index, file)

# Save the docs variable as a pickle file
docs_file_path = r"C:\Users\user\Documents\IRSystem1\docs12.pkl"
with open(docs_file_path, 'wb') as file:
    pickle.dump(docs, file)

print("TF-IDF matrix, inverted index, and documents have been saved successfully.")

