import pandas as pd
from .metrics import evaluate_model
from sklearn.model_selection import train_test_split
from src.preprocessing.feature_selection import feature_selection
from imblearn.under_sampling import RandomUnderSampler
def compare_models(vectorizers, models, X, y, test_size=0.1):
  results = []
  for vec_name, vec in vectorizers.items():
    print(f"Evaluating models with {vec_name} vectorization...")

    X_vec = vec.fit_transform(X)
    
    X_vec = feature_selection(X_vec)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42, stratify=y)

    undersampler = RandomUnderSampler(random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

    for model_name, model in models.items():
      print(f"Training {model_name} with {vec_name} vectorizer...")

      model.fit(X_train, y_train)
      
      y_pred = model.predict(X_test)

      metrics_dict = evaluate_model(y_test, y_pred)
      metrics_dict.update({"vectorizer": vec_name, "model": model_name})
      results.append(metrics_dict)

  return pd.DataFrame(results).sort_values(by='accuracy', ascending=False)