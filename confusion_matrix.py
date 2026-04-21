import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.preprocessing.clean_text import basic_clean, remove_aux_verbs, stem_text
from src.preprocessing.target_preprocessing import target_processing
from src.vectorization.bow import get_bow
from src.vectorization.tfidf import get_tfidf
from src.evaluation.metrics import evaluate_model
from src.preprocessing.feature_selection import feature_selection

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

# Next, let's add the vectorization
vect = get_tfidf()


model = get_random_forest()

X = df[text_col].values 
y = df[target_col].values

X_vec = vect.fit_transform(X)

X_vec = feature_selection(X_vec)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.1, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)


print(evaluate_model(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['ambiguous', 'positive', 'negative'])
disp.plot()
plt.show()

