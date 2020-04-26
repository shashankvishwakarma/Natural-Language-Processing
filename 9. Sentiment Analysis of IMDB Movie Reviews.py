import nltk
import pandas as pd
from bs4 import BeautifulSoup

from nltk import ToktokTokenizer
import re, string, unicodedata

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

imdb_data = pd.read_csv('dataset/IMDB Dataset.csv')
imdb_data.shape
imdb_data.head(10)

# Summary of the dataset
imdb_data.describe()

# sentiment count
imdb_data['sentiment'].value_counts()

# split the dataset
# train dataset
train_reviews = imdb_data.review[:40000]
train_sentiments = imdb_data.sentiment[:40000]

# test dataset
test_reviews = imdb_data.review[40000:]
test_sentiments = imdb_data.sentiment[40000:]

# print(train_reviews.shape, train_sentiments.shape)
# print(test_reviews.shape, test_sentiments.shape)

# Tokenization of text
tokenizer = ToktokTokenizer()

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


# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(denoise_text)


# Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)


# Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Apply function on review column
imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
imdb_data.head(10)

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
imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

# normalized train reviews
norm_train_reviews = imdb_data.review[:40000]
norm_train_reviews[0]
# convert dataframe to string
# norm_train_string=norm_train_reviews.to_string()
# Spelling correction using Textblob
# norm_train_spelling=TextBlob(norm_train_string)
# norm_train_spelling.correct()
# Tokenization using Textblob
# norm_train_words=norm_train_spelling.words
# norm_train_words

# Normalized test reviews
norm_test_reviews = imdb_data.review[40000:]
norm_test_reviews[45005]
##convert dataframe to string
# norm_test_string=norm_test_reviews.to_string()
# spelling correction using Textblob
# norm_test_spelling=TextBlob(norm_test_string)
# print(norm_test_spelling.correct())
# Tokenization using Textblob
# norm_test_words=norm_test_spelling.words
# norm_test_words

'''
Bags of words model
It is used to convert text documents to numerical vectors or bag of words.
'''

# Count vectorizer for bag of words
cv = CountVectorizer()
# transformed train reviews
cv_train_reviews = cv.fit_transform(norm_train_reviews)
# transformed test reviews
cv_test_reviews = cv.transform(norm_test_reviews)

print('BOW_cv_train:', cv_train_reviews.shape)
print('BOW_cv_test:', cv_test_reviews.shape)

'''
Term Frequency-Inverse Document Frequency model (TFIDF)
It is used to convert text documents to matrix of tfidf features.
'''
# Tfidf vectorizer
tv = TfidfVectorizer()
# transformed train reviews
tv_train_reviews = tv.fit_transform(norm_train_reviews)
# transformed test reviews
tv_test_reviews = tv.transform(norm_test_reviews)
print('Tfidf_train:', tv_train_reviews.shape)
print('Tfidf_test:', tv_test_reviews.shape)

# labeling the sentient data
lb = LabelBinarizer()
# transformed sentiment data
sentiment_data = lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)

# Spliting the sentiment data
train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)


'''
Modelling the dataset
Let us build logistic regression model for both bag of words and tfidf features.
'''
# training the model
lr = LogisticRegression()
# Fitting the model for Bag of words
lr_bow = lr.fit(cv_train_reviews, train_sentiments)
print(lr_bow)
# Fitting the model for tfidf features
lr_tfidf = lr.fit(tv_train_reviews, train_sentiments)
print(lr_tfidf)

'''
Logistic regression model performane on test dataset.
'''

# Predicting the model for bag of words
lr_bow_predict = lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict = lr.predict(tv_test_reviews)
print(lr_tfidf_predict)

'''
Accuracy of the model
'''
# Accuracy score for bag of words
lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
print("lr_bow_score :", lr_bow_score)
# Accuracy score for tfidf features
lr_tfidf_score = accuracy_score(test_sentiments, lr_tfidf_predict)
print("lr_tfidf_score :", lr_tfidf_score)


#Classification report for bag of words
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)

#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)

# confusion matrix for bag of words
cm_bow = confusion_matrix(test_sentiments, lr_bow_predict, labels=[1, 0])
print(cm_bow)
# confusion matrix for tfidf features
cm_tfidf = confusion_matrix(test_sentiments, lr_tfidf_predict, labels=[1, 0])
print(cm_tfidf)
