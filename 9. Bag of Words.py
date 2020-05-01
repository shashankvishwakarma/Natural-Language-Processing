'''
Bag-of-Words -
We need a way to represent text data for machine learning algorithm and
the bag-of-words model helps us to achieve that task.
The bag-of-words model is simple to understand and implement.
It is a way of extracting features from the text for use in machine learning algorithms.
'''

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

documents_txt = ['Hello, how are you!',
                 'Win money, win from home.',
                 'Call me now.',
                 'Hello, Call hello you tomorrow?']

# cleaning data
def cleaning(raw_review):
    remove_tags = BeautifulSoup(raw_review, "html.parser").get_text()
    letters = re.sub("[^a-zA-Z]", " ", remove_tags)
    lower_case = letters.lower()
    words = lower_case.split()
    stopword = stopwords.words("english")
    meaningful_words = [w for w in words if not w in stopword]
    return (" ".join(meaningful_words))

# create a variable for the for loop results
documents = []

# Manual cleaning for data
for i in documents_txt:
    documents.append(cleaning(i))

# comment below line to get data after applying cleaning method
documents = documents_txt

classifiers = [
    CountVectorizer(),
    TfidfVectorizer()
]

for clf in classifiers:
    clf.fit(documents)
    print(clf.get_feature_names())

    doc_array = clf.transform(documents).toarray()
    # print(doc_array)

    frequency_matrix = pd.DataFrame(doc_array, index=documents, columns=clf.get_feature_names())
    print(frequency_matrix)
