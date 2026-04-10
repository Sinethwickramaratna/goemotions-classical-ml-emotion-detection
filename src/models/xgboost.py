from xgboost import XGBClassifier

def get_xgboost_model():
    return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)