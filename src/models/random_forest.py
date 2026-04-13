from sklearn.ensemble import RandomForestClassifier

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)