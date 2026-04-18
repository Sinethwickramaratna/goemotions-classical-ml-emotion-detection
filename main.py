import pandas as pd

from src.preprocessing.clean_text import basic_clean, remove_aux_verbs, stem_text
from src.preprocessing.target_preprocessing import target_processing
from src.vectorization.bow import get_bow
from src.vectorization.tfidf import get_tfidf

from src.models.knn import get_knn_model
from src.models.logistic_regression import get_logistic_regression_model
from src.models.random_forest import get_random_forest
from src.models.svm import get_svm_model
from src.models.xgboost import get_xgboost_model
from src.models.sgd import get_sgd_model
from src.evaluation.compare import compare_models

# First lets load the data
df_1 = pd.read_csv("data/raw/goemotions_1.csv")
df_2 = pd.read_csv("data/raw/goemotions_2.csv")
df_3 = pd.read_csv("data/raw/goemotions_3.csv")
df = pd.concat([df_1, df_2, df_3], ignore_index=True)

# Next, let's first identify the text and label columns
text_col = 'text'
target_col='sentiment_label'

# Now we can preprocess the text data by applying the cleaning, stopword removal, and lemmatization functions
df[text_col] = df[text_col].apply(basic_clean)
df[text_col] = df[text_col].apply(remove_aux_verbs)
df[text_col] = df[text_col].apply(stem_text)

# Next, let's preprocess the target labels by applying the target_processing function to the dataframe
df = target_processing(df)

# Next, let's add the vectorizations I am interested in comparing to the dataframe.
vectorization_methods = {
  'tfidf': get_tfidf(),
  'bow': get_bow()
}  

# Now let's compare the models using the compare_models function
models = {
  'Logistic Regression': get_logistic_regression_model(),
  'Random Forest': get_random_forest(),
  'XGBoost': get_xgboost_model(),
  'SGD': get_sgd_model(),
  'KNN': get_knn_model(),
  'SVM': get_svm_model()
}

# We will use the iterative_train_test_split function from skmultilearn to split the data into training and testing sets
X = df[text_col].values 
y = df[target_col].values

results_df = compare_models(vectorization_methods, models, X, y)
print(results_df)
results_df.to_csv('./comparison_results.csv', index=False)
