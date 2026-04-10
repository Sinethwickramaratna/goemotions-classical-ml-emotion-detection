from sklearn.feature_extraction.text import CountVectorizer

def get_bow(max_features=5000, ngram_range=(1, 1)):
    return CountVectorizer(max_features=max_features, ngram_range=ngram_range)

