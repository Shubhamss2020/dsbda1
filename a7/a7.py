import nltk
import re
nltk.download('omw-1.4')
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# # Step 2: Initialize the text
text= "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."

# # Step 3: Perform Tokenization
# #Sentence Tokenization
from nltk.tokenize import sent_tokenize
tokenized_text= sent_tokenize(text)
print(tokenized_text)

# #Word Tokenization
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# # Step 4: Removing Punctuations and Stop Word
# # Print stop words of English
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)
text= "How to remove stop words with NLTK library in Python?"
text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
	if w not in stop_words:
		filtered_text.append(w)
print ("Tokenized Sentence:",tokens)
print ("Filterd Sentence:",filtered_text)

# # Step 5: Perform Stemming

from nltk.stem import PorterStemmer
e_words= ["wait", "waiting", "waited", "waits"]
ps =PorterStemmer()
for w in e_words:
	rootWord=ps.stem(w)
print(rootWord)

# # perform lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
	print("Lemma for {} is {}".format(w,wordnet_lemmatizer.lemmatize(w)))

# # apply pos tagging to text
import nltk
from nltk.tokenize import word_tokenize
data="The pink sweater fit her perfectly"
words=word_tokenize(data)
for word in words:
	print(nltk.pos_tag([word]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Initialize the Documents.
documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'

# Step 3: Create BagofWords (BoW) for Document A and B.
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

# Step 4: Create Collection of Unique words from Document A and B.
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

# Step 5: Create a dictionary of words and their occurrence for each document in the corpus
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
	numOfWordsA[word] += 1
	numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
	numOfWordsB[word] += 1

# Step 6: Compute the term frequency for each of our documents.
def computeTF(wordDict, bagOfWords):
	tfDict = {}
	bagOfWordsCount = len(bagOfWords)
	for word, count in wordDict.items():
		tfDict[word] = count / float(bagOfWordsCount)
	return tfDict
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)

# Step 7: Compute the term Inverse Document Frequency.
import math
def computeIDF(documents):
	N = len(documents)
	idfDict = dict.fromkeys(documents[0].keys(), 0)
	for document in documents:
		for word, val in document.items():
			if val > 0:
				idfDict[word] += 1
	for word, val in idfDict.items():
		idfDict[word] = math.log(N / float(val))
	return idfDict
idfs = computeIDF([numOfWordsA, numOfWordsB])
print(idfs)

# Step 8: Compute the term TF/IDF for all words.
def computeTFIDF(tfBagOfWords, idfs):
	tfidf = {}
	for word, val in tfBagOfWords.items():
		tfidf[word] = val * idfs[word]
	return tfidf
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
print(df)