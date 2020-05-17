#import modules
import pandas as pd
import numpy as np

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)

# Making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)

# load data and view data
sms_dataset = pd.read_csv('./dataset/spam.csv', sep=',', encoding='Latin-1')
print(sms_dataset.shape)
print(sms_dataset.head())

# Remove empty columns and give existing columns meaningful names.¶
# drop meaningless columns
sms_dataset = sms_dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Give meaningful names for exist columns
sms_dataset.columns = ['label', 'texts']

# lowering case characters and stripping punctuation of texts
from string import punctuation
def preprocess1 (str):
    """ str --> str
    lower case and strip punctuations of a string
    """
    for p in list(punctuation):
        str = str.lower().replace(p, '')
    return str
sms_dataset['texts'] = sms_dataset.texts.apply(preprocess1)

# creating word count feature as "word_count"
def word_count (str):
    """str --> int
    return the number of words appeared in a string
    """
    return len(str.split())
sms_dataset['word_count'] = sms_dataset.texts.apply(word_count)

# creating character count feature as "char_count"
def char_count (str):
    """ str --> int
    return the number of character count of a string
    """
    return len(list(str))
sms_dataset['char_count'] = sms_dataset.texts.apply(char_count)

# view new data set
print(sms_dataset.shape)
print(sms_dataset.head())

# Statistical Inferance for Word Count.
'''
If we can confirm the ham sample and spam sample come from populations that are normally distributed, 
we can use t-test to test if spam messages has significantly greater word count than ham messages.
There's a Shapiro-Wilk test to test for normality. 
If p-value is less than 0.05, then there's a low chance that the distribution is normal.
'''
ham_word_count = sms_dataset.loc[sms_dataset.label=='ham']['word_count']
spam_word_count = sms_dataset.loc[sms_dataset.label=='spam']['word_count']

from scipy import stats
print ("Shapiro-Wilk test results for ham sample:", stats.shapiro(ham_word_count))
print ("Shapiro-Wilk test results for spam sample", stats.shapiro(spam_word_count))
print ('mean-ham:', ham_word_count.mean())
print ('mean-spam:', spam_word_count.mean())
print('stand-deviation-ham:', ham_word_count.std())
print('stand-deviation-spam:', spam_word_count.std())
print('effect size - mean difference:', ham_word_count.mean() - spam_word_count.mean())

# Cohan's D
SS_ham = sum(np.square(i-ham_word_count.mean()) for i in ham_word_count)
SS_spam = sum(np.square(i-ham_word_count.mean()) for i in spam_word_count)
pooled_variance = (SS_ham + SS_spam) / (len(ham_word_count) + len(spam_word_count) - 2)
cohans_D = (ham_word_count.mean() - spam_word_count.mean()) / np.sqrt(pooled_variance)
print("effect size - Cohan's D:", cohans_D)

# Logistic Regression
# dummy code variable
sms_dataset = pd.get_dummies(sms_dataset, columns=['label'])

# Identify explaining variables and target variable
y = sms_dataset[['label_spam']]
X = sms_dataset[['word_count']]

# split data into training and test data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# apply LogisticRegression classifier
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
print (lg.coef_)
print('training set score obtained Logistic Regression: {:.2f}'.format(lg.score(X_train, y_train)))
print('test set score obtained Logistic Regression: {:.2f}'.format(lg.score(X_test, y_test)))

prediction = lg.predict(X_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,prediction)
print('Accuracy {}'.format(accuracy))

confusion_matrix = metrics.confusion_matrix(y_test, prediction)
print('confusion_matrix {}'.format(confusion_matrix))

classification_report = metrics.classification_report(y_test, prediction)
print('classification_report {}'.format(classification_report))



