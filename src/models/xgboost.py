from xgboost import XGBClassifier

def get_xgboost_model():
    return XGBClassifier(eval_metric='logloss', random_state=42)