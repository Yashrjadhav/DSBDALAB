import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Sample document
document = "This is a sample document for preprocessing. We will apply various techniques such as tokenization, POS tagging, stop words removal, stemming, and lemmatization."

# Tokenization
tokens = word_tokenize(document)

# POS tagging
pos_tags = pos_tag(tokens)

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Calculate Term Frequency (TF) and Inverse Document Frequency (IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([document])

# Print the results
print("Original Document:")
print(document)

print("\nTokenization:")
print(tokens)

print("\nPOS Tagging:")
print(pos_tags)

print("\nStop Words Removal:")
print(filtered_tokens)

print("\nStemming:")
print(stemmed_tokens)

print("\nLemmatization:")
print(lemmatized_tokens)

print("\nTF-IDF Representation:")
print(tfidf_matrix.toarray())
