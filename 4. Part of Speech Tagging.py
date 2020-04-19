'''
One of the more powerful aspects of the NLTK module is the Part of Speech tagging that it can do for you.
This means labeling words in a sentence as nouns, adjectives, verbs...etc.
Even more impressive, it also labels by tense, and more.
'''

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#from nltk.tokenize import word_tokenize

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
#print(tokenized)
#tokenized = word_tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()



