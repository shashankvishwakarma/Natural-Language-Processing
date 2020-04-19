#import nltk
#nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? " \
               "The weather is great, and Python is awesome. " \
               "The sky is pinkish-blue. " \
               "You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))
'''
There are a few things to note here. 
First, notice that punctuation is treated as a separate token. 
Also, notice the separation of the word "shouldn't" into "should" and "n't." 
Finally, notice that "pinkish-blue" is indeed treated like the "one word" it was meant to be turned into. Pretty cool!
'''
print(word_tokenize(EXAMPLE_TEXT))

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])