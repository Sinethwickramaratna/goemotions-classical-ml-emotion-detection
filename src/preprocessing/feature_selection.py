from sklearn.feature_selection import VarianceThreshold

vathr = VarianceThreshold(threshold=0.0001)
def feature_selection(X):
  return vathr.fit_transform(X)