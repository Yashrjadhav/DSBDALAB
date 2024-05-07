import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag

# Sample document
document = "This is a sample document for preprocessing. We will apply various techniques such as tokenization, POS tagging, stop words removal, stemming, and lemmatization."

# Preprocessing
tokens = word_tokenize(document)
pos_tags = pos_tag(tokens)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Print the results
print("Original Document:")
print(document)

print("\nTokenization:")
print(tokens)

print("\npos tagging:")
print(pos_tags)

print("\nStop Words Removal:")
print(filtered_tokens)

print("\nStemming:")
print(stemmed_tokens)

print("\nLemmatization:")
print(lemmatized_tokens)

# Calculate TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([document])

print("\nTF-IDF Representation:")
print(tfidf_matrix.toarray())

