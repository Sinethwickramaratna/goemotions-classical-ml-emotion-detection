from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(max_features=5000):
    return TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.8, ngram_range=(1, 3))