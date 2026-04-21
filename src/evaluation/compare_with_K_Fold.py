import pandas as pd
from .metrics import evaluate_model
from sklearn.model_selection import train_test_split
from src.preprocessing.feature_selection import feature_selection
from sklearn.model_selection import StratifiedKFold

def compare_models(vectorizers, models, X, y, test_size=0.1):
  results = []
  for vec_name, vec in vectorizers.items():
    print(f"Evaluating models with {vec_name} vectorization...")
    X_vec = vec.fit_transform(X)
    
    X_vec = feature_selection(X_vec)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42, stratify=y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
      print(f"Training {model_name} with {vec_name} vectorizer...")

      predictions = []
      for train_idx, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_index]

        model.fit(X_train_fold, y_train_fold)

        y_pred_val = model.predict(X_val_fold)
        y_pred_test = model.predict(X_test)
        predictions.append(y_pred_test)
      
      # Mode of predictions across folds
      y_pred = pd.DataFrame(predictions).mode().iloc[0].values

      metrics_dict = evaluate_model(y_test, y_pred)
      metrics_dict.update({"vectorizer": vec_name, "model": model_name})
      results.append(metrics_dict)

  return pd.DataFrame(results).sort_values(by='accuracy', ascending=False)