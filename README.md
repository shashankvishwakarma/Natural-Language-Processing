# Natural Language Processing - NLP

## How to install NLTK
    pip install nltk

### How to Download Packages for NLTK 
	import nltk
	nltk.download()

## How to Install spaCy
    pip install spacy

### How to Download Models and Data for spaCy
	python -m spacy download en

### Basics of Natural Language Processing(NLP) using the Natural Language Toolkit (NLTK) module with Python.

#### What are Corpus, Tokens, and Engrams?
	A Corpus is defined as a collection of text documents for example a data set containing news is a corpus or the tweets containing Twitter data is a corpus. So corpus consists of documents, documents comprise paragraphs, paragraphs comprise sentences and sentences comprise further smaller units which are called Tokens.
	
	Tokens can be words, phrases, or Engrams, and Engrams are defined as the group of n words together.
    For example, consider this given sentence-
    “I love my phone.”
    In this sentence, the uni-grams(n=1) are: I, love, my, phone
    Di-grams(n=2) are: I love, love my, my phone
    And tri-grams(n=3) are: I love my, love my phone
    So, uni-grams are representing one word, di-grams are representing two words together and tri-grams are representing three words together.

#### What is Tokenization?
	Tokenization is a process of splitting a text object into smaller units which are also called tokens. Examples of tokens can be words, numbers, engrams, or even symbols. The most commonly used tokenization process is White-space Tokenization.

#### What is White-space Tokenization?
    Also known as unigram tokenization. In this process, the entire text is split into words by splitting them from white spaces. For example, in a sentence- “I went to New-York to play football.”
    This will be splitted into following tokens: “I”, “went”, “to”, “New-York”, “to”, “play”, “football.” Notice that “New-York” is not split further because the tokenization process was based on whitespaces only.

#### What is Regular Expression Tokenization?
    The other type of tokenization process is Regular Expression Tokenization, in which a regular expression pattern is used to get the tokens. For example, consider the following string containing multiple delimiters such as comma, semi-colon, and white space.
    
    Sentence= “Football, Cricket; Golf Tennis"
    re.split(r’[;,\s]’, Sentence
    Tokens= “Football”, ”Cricket”, “Golf”, “Tennis”

    Using Regular expression, we can split the text by passing a splitting pattern.

#### What is Normalization?
	Before further processing, text needs to be normalized. Normalization generally refers to a series of related tasks meant to put all text on a level playing field: converting all text to the same case (upper or lower), removing punctuation, expanding contractions, converting numbers to their word equivalents, and so on. Normalization puts all words on equal footing, and allows processing to proceed uniformly.

#### What is Stemming?
	Refers to the process of slicing the end or the beginning of words with the intention of removing prefixes and suffixes from root of the word. The problem is that this process can create or expand new forms of the same word or even create new words themselves.
    Stemmers use algorithmics approaches, the result of the stemming process may not be an actual word or even change the word (and sentence) meaning.
    
    Playing --> play (correct)
    news    --> new  (wrong)
    
    So if stemming has serious limitations, why do we use it? First of all, it can be used to correct spelling errors from the tokens. Stemmers are simple to use and run very fast (they perform simple operations on a string), and if speed and performance are important in the NLP model, then stemming is certainly the way to go.

#### What is Lemmatization?
    Has the objective of reducing a word to its base form and grouping together different forms of the same word. 
    For example, verbs in past tense are changed into present (e.g. “went” is changed to “go”) and synonyms are unified (e.g. “best” is changed to “good”), hence standardizing words with similar meaning to their root. 
    
    Although it seems closely related to the stemming process, lemmatization uses a different approach to reach the root forms of words.
    
    Lemmatization resolves words to their dictionary form (known as lemma) for which it requires detailed dictionaries in which the algorithm can look into and link words to their corresponding lemmas.

    For example, the words “running”, “runs” and “ran” are all forms of the word “run”, so “run” is the lemma of all the previous words.
    
    Lemmatization : caring --> care (correct)
    Stemming      : caring --> car  (wrong)
    
    Lemmatization also takes into consideration the context of the word in order to solve other problems like disambiguation, which means it can discriminate between identical words that have different meanings depending on the specific context. 
    Think about words like “bat” (which can correspond to the animal or to the metal/wooden club used in baseball) or “bank” (corresponding to the financial institution or to the land alongside a body of water). 
    By providing a part-of-speech parameter to a word ( whether it is a noun, a verb, and so on) it’s possible to define a role for that word in the sentence and remove disambiguation.
    
    At the same time, since it requires more knowledge about the language structure than a stemming approach, it demands more computational power than setting up or adapting a stemming algorithm.   

#### What is the Difference Amid Stemming and Lemmatization?
| Stemming  | Lemmatization  |
| ------------- | ------------- |
| Stemming is faster because it chops words without knowing the context of the word in given sentences.  | Lemmatization is slower as compared to stemming but it knows the context of the word before proceeding.  |
| It is a rule-based approach.  | It is a dictionary-based approach.  |
| Accuracy is less.  | Accuracy is more as compared to Stemming.  |
| When we convert any word into root-form then stemming may create the non-existence meaning of a word.  | Lemmatization always gives the dictionary meaning word while converting into root-form.  |
| Stemming is preferred when the meaning of the word is not important for analysis. Example: Spam Detection  | Lemmatization would be recommended when the meaning of the word is important for analysis. Example: Question Answer  |
| For Example: “Studies” => “Studi”  | For Example: “Studies” => “Study”  |

#### What is Stop Words?
	Includes getting rid of common language articles, pronouns and prepositions such as “and”, “the” or “to” in English. In this process some very common words that appear to provide little or no value to the NLP objective are filtered and excluded from the text to be processed, hence removing widespread and frequent terms that are not informative about the corresponding text.
    
    Stop words can be safely ignored by carrying out a lookup in a pre-defined list of keywords, freeing up database space and improving processing time.
    
    Stop words removal can wipe out relevant information and modify the context in a given sentence. 
    For example, if we are performing a sentiment analysis we might throw our algorithm off track if we remove a stop word like “not”. Under these conditions, you might select a minimal stop word list and add additional terms depending on your specific objective.

#### What is Part of Speech(PoS) Tags in Natural Language Processing?
	Part of speech tags or PoS tags is the properties of words that define their main context, their function, and the usage in a sentence. Some of the commonly used parts of speech tags are- Nouns, which define any object or entity; Verbs, which define some action; and Adjectives or Adverbs, which act as the modifiers, quantifiers, or intensifiers in any sentence. In a sentence, every word will be associated with a proper part of the speech tag, for example, 
	“David has purchased a new laptop from the Apple store.”

	In the below sentence, every word is associated with a part of the speech tag which defines their functions.

	POS Tag -
	In this case “David’ has NNP tag which means it is a proper noun, “has” and “purchased” belongs to verb indicating that they are the actions and “laptop” and “Apple store” are the nouns, “new” is the adjective whose role is to modify the context of laptop.
	Part of speech tags is defined by the relations of words with the other words in the sentence. Machine learning models or rule-based models are applied to obtain the part of speech tags of a word. The most commonly used part of speech tagging notations is provided by the Penn Part of Speech Tagging.
	Part of speech tags have a large number of applications and they are used in a variety of tasks such as text cleaning, feature engineering tasks, and word sense disambiguation. For example, consider these two sentences-

	Sentence 1:  “Please book my flight for NewYork”
	Sentence 2: “I like to read a book on NewYork”
		In both sentences, the keyword “book” is used but in sentence one, it is used as a verb while in sentence two it is used as a noun.


#### Grammar in NLP and its types
	Grammar refers to the rules for forming well-structured sentences. The first type of Grammar is Constituency grammar.

#### What is Constituency Grammar?
    Any word, group of words, or phrases can be termed as Constituents and the goal of constituency grammar is to organize any sentence into its constituents using their properties. These properties are generally driven by their part of speech tags, noun or verb phrase identification.
    For example, constituency grammar can define that any sentence can be organized into three constituents- a subject, a context, and an object.

#### What is Dependency Grammar?
    A different type of grammar is Dependency Grammar which states that words of a sentence are dependent upon other words of the sentence. For example, in the previous sentence “barking dog” was mentioned and the dog was modified by barking as the dependency adjective modifier exists between the two.
    Dependency grammar organizes the words of a sentence according to their dependencies. One of the words in a sentence acts as a root and all the other words are directly or indirectly linked to the root using their dependencies. These dependencies represent relationships among the words in a sentence and dependency grammars are used to infer the structure and semantics dependencies between the words.

    Dependency grammars can be used in different use case-
        1.  Named Entity Recognition– they are used to solve named entity recognition problems.
        2.  Question Answering System– they can be used to understand relational and structural aspects of question-answering systems.
        3.  Coreference Resolution– they are also used in coreference resolutions in which the task is to map the pronouns to the respective noun phrases.
        4.  Text summarization and Text classification– they can also be used for text summarization problems and they are also used as features for text classification problems.

#### What is Bag of Words? 
	Is a commonly used model that allows you to count all words in a piece of text. Basically it creates an occurrence matrix for the sentence or document, disregarding grammar and word order. 
	These word frequencies or occurrences are then used as features for training a classifier.

    Actual storage mechanisms for the bag of words representation can vary, but the following is a simple example using a dictionary for intuitiveness. 
    Sample text:
    "Well, well, well," said John.
    "There, there," said James. "There, there."
    
    The resulting bag of words representation as a dictionary:
       {
          'well': 3,
          'said': 2,
          'john': 1,
          'there': 4,
          'james': 1
       }

    This approach may reflect several downsides like the absence of semantic meaning and context, and the facts that stop words (like “the” or “a”) add noise to the analysis and some words are not weighted accordingly (“universe” weights less than the word “they”).
    
    To solve this problem, one approach is to rescale the frequency of words by how often they appear in all texts (not just the one we are analyzing) so that the scores for frequent words like “the”, that are also frequent across other texts, get penalized. 
    This approach to scoring is called “Term Frequency — Inverse Document Frequency” (TFIDF), and improves the bag of words by weights. Through TFIDF frequent terms in the text are “rewarded” (like the word “they” in our example), but they also get “punished” if those terms are frequent in other texts we include in the algorithm too. On the contrary, this method highlights and “rewards” unique or rare terms considering all texts. Nevertheless, this approach still has no context nor semantics.

#### What is TF-IDF?
    TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
    One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
    Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
    
    TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
    TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
    
    See below for a simple example.
    Consider a document containing 100 words wherein the word cat appears 3 times.
    The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.