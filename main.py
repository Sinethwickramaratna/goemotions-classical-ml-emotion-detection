import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.multioutput import ClassifierChain

from src.preprocessing.clean_text import basic_clean, remove_stopwords, lemmatize_text
from src.vectorization.bow import get_bow
from src.vectorization.tfidf import get_tfidf
from src.models.knn import get_knn_model
from src.models.logistic_regression import get_logistic_regression_model
from src.models.naive_bayes import get_multinomial_nb_model
from src.models.svm import get_svm_model
from src.models.xgboost import get_xgboost_model
from src.models.sgd import get_sgd_model
from src.models.passive_agressive import get_passive_aggressive_model
from src.evaluation.compare import compare_models

# First lets load the data
df_1 = pd.read_csv("data/raw/goemotions_1.csv")
df_2 = pd.read_csv("data/raw/goemotions_2.csv")
df_3 = pd.read_csv("data/raw/goemotions_3.csv")
df = pd.concat([df_1, df_2, df_3], ignore_index=True)

# Next, let's first identify the text and label columns
text_col = 'text'
target_cols = ['admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']

# Now we can preprocess the text data by applying the cleaning, stopword removal, and lemmatization functions
df[text_col] = df[text_col].apply(basic_clean)
df[text_col] = df[text_col].apply(remove_stopwords)
df[text_col] = df[text_col].apply(lemmatize_text)

# Next, let's add the vectorizations I am interested in comparing to the dataframe.
vectorization_methods = {
  'tfidf': get_tfidf(),
  'bow': get_bow()
}  

# Now let's compare the models using the compare_models function
models = {
  'Logistic Regression': ClassifierChain(get_logistic_regression_model(), order='random', random_state=42),
  'Multinomial Naive Bayes': ClassifierChain(get_multinomial_nb_model(), order='random', random_state=42),
  'XGBoost': ClassifierChain(get_xgboost_model(), order='random', random_state=42),
  'SGD': ClassifierChain(get_sgd_model(), order='random', random_state=42)  
}

grid_params = {
  'Logistic Regression': {
    'estimator__C': [0.01, 0.1, 1, 10],
    'estimator__penalty': ['l2'],
    'estimator__solver': ['lbfgs', 'newton-cg']
  },
  'Multinomial Naive Bayes': {
    'estimator__alpha': [0.01, 0.1, 1, 10]
  },
  'XGBoost': {
    'estimator__n_estimators': [100, 200],
    'estimator__learning_rate': [0.01, 0.1],
    'estimator__max_depth': [3, 5]
  },
  'SGD': {
    'estimator__alpha': [0.0001, 0.001, 0.01],
    'estimator__penalty': ['l2', 'elasticnet'],
    'estimator__loss': ['log_loss'],
    'estimator__learning_rate': ['optimal', 'invscaling']
  }
}


# We will use the iterative_train_test_split function from skmultilearn to split the data into training and testing sets
X = df[text_col].values 
y = df[target_cols].values

results_df = compare_models(vectorization_methods, models, X, y)
print(results_df)
results_df.to_csv('./comaprsion results.csv')
