from sklearn.linear_model import SGDClassifier

def get_sgd_model():
  return SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)