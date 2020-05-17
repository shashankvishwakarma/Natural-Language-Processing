from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
print(word_tokens)

#Example 1;
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(filtered_sentence)

filtered_sentence = [];

# Example 2;
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

# Example 3
import string
message = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
no_punctuation = [char for char in message if char not in string.punctuation]

# Join the characters again to form the string.
no_punctuation = ''.join(no_punctuation)

# Now let's see how to remove stopwords.
# We can impot a list of english stopwords from NLTK (check the documentation for more languages and info).

from nltk.corpus import stopwords
print(stopwords.words('english')[0:10]) # Show some stop words

# Now just remove any stopwords
clean_message = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
print('clean message {}'.format(clean_message))