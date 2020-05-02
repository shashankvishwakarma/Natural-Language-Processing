import nltk
import pandas as pd
import sklearn
from bs4 import BeautifulSoup

from nltk import ToktokTokenizer
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

imdb_data_set = pd.read_csv('dataset/IMDB Dataset.csv')
# Summary of the dataset
print(imdb_data_set.describe())

# selecting only first 1000 row so that can compare and verify the results
imdb_data = imdb_data_set.iloc[:]
print(imdb_data.keys())

X = imdb_data['review']
Y = imdb_data.drop('review', axis=1)


# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


# Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


# Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


# Tokenization of text
tokenizer = ToktokTokenizer()

# set stopwords to english
stop_words = set(stopwords.words('english'))


# removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop_words]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Apply function on review column
X = X.apply(denoise_text)

# Apply function on review column
X = X.apply(remove_special_characters)

# Apply function on review column
X = X.apply(simple_stemmer)

# Apply function on review column
X = X.apply(remove_stopwords)

# labeling the sentient data
label_binarizer = LabelBinarizer()

# transformed sentiment data
Y = label_binarizer.fit_transform(Y)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.20, random_state=5)

# Count Vectorizer for bag of words
count_vectorizer = CountVectorizer()

# transformed train reviews
count_vectorizer_train_reviews = count_vectorizer.fit_transform(X_train)

# transformed test reviews
count_vectorizer_test_reviews = count_vectorizer.transform(X_test)

# Tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer()

# transformed train reviews
tfidf_vectorizer_train_reviews = tfidf_vectorizer.fit_transform(X_train)

# transformed test reviews
tfidf_vectorizer_test_reviews = tfidf_vectorizer.transform(X_test)

classifiers_list = [
    LogisticRegression(solver='lbfgs'),
    SGDClassifier(),
    MultinomialNB()
]

for classifier in classifiers_list:
    print('============================================================')
    # Fitting the model for Bag of words
    classifier.fit(count_vectorizer_train_reviews, Y_train)

    # Fitting the model for tfidf features
    classifier.fit(tfidf_vectorizer_train_reviews, Y_train)

    # Predicting the model for bag of words
    bow_prediction = classifier.predict(count_vectorizer_test_reviews)
    print(bow_prediction)

    # Predicting the model for tfidf features
    tfidf_prediction = classifier.predict(tfidf_vectorizer_test_reviews)
    print(tfidf_prediction)

    # Accuracy score for bag of words
    accuracy_score_bow = accuracy_score(Y_test, bow_prediction)
    print("Accuracy score for bag of words :", accuracy_score_bow)

    # Accuracy score for tfidf features
    accuracy_score_tfidf = accuracy_score(Y_test, tfidf_prediction)
    print("Accuracy score for tfidf features :", accuracy_score_tfidf)

    # Classification report for bag of words
    # lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
    # print(lr_bow_report)

    # Classification report for tfidf features
    # lr_tfidf_report = classification_report(test_sentiments, lr_tfidf_predict, target_names=['Positive', 'Negative'])
    # print(lr_tfidf_report)

    # confusion matrix for bag of words
    confusion_matrix_bow = confusion_matrix(Y_test, bow_prediction, labels=[1, 0])
    print('confusion matrix for bag of words \n', confusion_matrix_bow)

    # confusion matrix for tfidf features
    confusion_matrix_tfidf = confusion_matrix(Y_test, tfidf_prediction, labels=[1, 0])
    print('confusion matrix for bag of words \n', confusion_matrix_tfidf)
