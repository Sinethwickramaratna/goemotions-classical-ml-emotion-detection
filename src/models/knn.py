from sklearn.neighbors import KNeighborsClassifier

def get_knn_model(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)