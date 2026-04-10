from sklearn.linear_model import PassiveAggressiveClassifier

def get_passive_aggressive_model():
  return PassiveAggressiveClassifier(max_iter=1000, random_state=42)