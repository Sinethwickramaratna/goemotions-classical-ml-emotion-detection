from sklearn.model_selection import cross_val_score, StratifiedKFold

def cross_validate_model(model, X, y, cv=5, scoring=['accuracy', 'f1_macro', 'f1_sample', 'hamming_loss', 'jaccard_macro', 'jaccard_samples']):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return scores