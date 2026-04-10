import pandas as pd
from .metrics import evaluate_model
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import GridSearchCV

threshold = 0.15
def compare_models(vectorizers, models, X, y, test_size=0.2):
  results = []
  for vec_name, vec in vectorizers.items():
      print(f"Evaluating models with {vec_name} vectorization...")
      X_vec = vec.fit_transform(X)
      X_train, y_train, X_test, y_test = iterative_train_test_split(X_vec, y, test_size=test_size)

      for model_name, model in models.items():
        print(f"Training {model_name} with {vec_name} vectorizer...")

        #grid_search = GridSearchCV(model, grid_params[model_name], cv=3, n_jobs=-1, verbose=1)
        #grid_search.fit(X_train, y_train)
        #print(f'Best parameters for {model_name} with {vec_name}: {grid_search.best_params_}')

        #model = grid_search.best_estimator_

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba >= 0.1).astype(int)
        metrics_dict = evaluate_model(y_test, y_pred)
        metrics_dict.update({"vectorizer": vec_name, "model": model_name})
        results.append(metrics_dict)
  return pd.DataFrame(results)