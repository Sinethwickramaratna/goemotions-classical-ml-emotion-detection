from sklearn.svm import SVC

def get_svm_model():
    return SVC(kernel='linear', random_state=42)