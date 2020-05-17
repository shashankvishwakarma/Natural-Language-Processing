# https://github.com/KunalArora/SpamDetection-NLP-UCIData/blob/master/SpamDetection-NLP.ipynb
# import modules
import pandas as pd
import numpy as np
import string

# Set ipython's max row display
from nltk.corpus import stopwords

pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# Making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

# load data and view data
sms_dataset = pd.read_csv('./dataset/spam.csv', sep=',', encoding='Latin-1')
print(sms_dataset.shape)
#print(sms_dataset.head())

# Remove empty columns and give existing columns meaningful names.¶
# drop meaningless columns
sms_dataset = sms_dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Give meaningful names for exist columns
sms_dataset.columns = ['label', 'texts']
print(sms_dataset.describe())
print(sms_dataset.shape)
#print(sms_dataset.head(5))

# group by to use describe by label, this way we can begin to think about the features that separate ham and spam!
# print(sms_dataset.groupby('label').describe())


# Text Pre-processing
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    no_punctuation = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    no_punctuation = ''.join(no_punctuation)

    # Now just remove any stopwords
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]

# creating character count feature as "char_count"
sms_dataset['char_count'] = sms_dataset.texts.apply(len)
#sms_dataset['texts'] = sms_dataset['texts'].apply(text_process)

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(sms_dataset['texts'])
# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:
text_4 = sms_dataset['texts'][3]
print('Text at index 4 {}'.format(text_4))

# Now let's see its vector representation:
bow_4 = bow_transformer.transform([text_4])
print(bow_4)
print(bow_4.shape)

'''
output of above shows, that there are seven unique words in message number 4 (after removing common stop words). 
Two of them appear twice, the rest only once. 
'''

# Let's go ahead and check and confirm which ones appear twice:
print(bow_transformer.get_feature_names()[3996])
print(bow_transformer.get_feature_names()[9445])

'''
Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. 
Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:
'''
texts_bow = bow_transformer.transform(sms_dataset['texts'])
print('Shape of Sparse Matrix: ', texts_bow.shape)
print('Amount of Non-Zero occurrences: ', texts_bow.nnz)

sparsity = (100.0 * texts_bow.nnz / (texts_bow.shape[0] * texts_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(texts_bow)
tfidf4 = tfidf_transformer.transform(bow_4)
print('tfidf4 is',tfidf4)

# We'll check what is the IDF (inverse document frequency) of the word "u" and of word "university"?
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

# To transform the entire bag-of-words corpus into TF-IDF corpus at once:
texts_tfidf = tfidf_transformer.transform(texts_bow)
print(texts_tfidf.shape)

# Training a model
# With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms.

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(texts_tfidf, sms_dataset['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', sms_dataset.label[3])

# Model Evaluation
#Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:
all_predictions = spam_detect_model.predict(texts_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report
print (classification_report(sms_dataset['label'], all_predictions))


# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(sms_dataset['texts'], sms_dataset['label'], test_size=0.2)
print(len(X_train), len(X_test), len(Y_train) + len(Y_test))

# Creating a Data Pipeline
'''Let's run our model again and then predict off the test set. 
We will use SciKit Learn's pipeline capabilities to store a pipeline of workflow. 
This will allow us to set up all the transformations that we will do to the data for future use. 
'''

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

'''
Now we can directly pass message text data and the pipeline will do our pre-processing for us! 
'''
pipeline.fit(X_train,Y_train)
predictions = pipeline.predict(X_test)
print(classification_report(predictions,Y_test))
