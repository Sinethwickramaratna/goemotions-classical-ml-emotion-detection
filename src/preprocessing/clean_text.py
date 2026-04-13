import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    #text = contractions.fix(text)
    return text


aux_verbs = set(['be', 'am', 'is', 'are', 'was', 'were', 'being', 'been',
                 'have', 'has', 'had', 'having',
                 'do', 'does', 'did', 'doing',
                 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could'])
def remove_aux_verbs(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in aux_verbs]
    return ' '.join(filtered_tokens)


def stem_text(text):
  return " ".join([stemmer.stem(word) for word in text.split()])

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def stem_text(text):
  return " ".join([stemmer.stem(word) for word in text.split()])