import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def stem_text(text):
  return " ".join([stemmer.stem(word) for word in text.split()])