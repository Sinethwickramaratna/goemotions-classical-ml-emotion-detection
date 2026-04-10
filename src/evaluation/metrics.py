from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

def evaluate_model(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_sample': f1_score(y_true, y_pred, average='samples'),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_macro': jaccard_score(y_true, y_pred, average='macro'),
        'jaccard_samples': jaccard_score(y_true, y_pred, average='samples')
    }
    return metrics