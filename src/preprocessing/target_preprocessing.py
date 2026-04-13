import pandas as pd
from pathlib import Path

def target_processing(df):
    emotions = ['admiration',
        'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise']
    # Let's turn the these labels into a 3 labels dataset: positive, negative, neutral
    project_root = Path(__file__).resolve().parents[2]
    sentiment_dict_path = project_root / 'data' / 'sentiment_dict.json'
    sentiment_dict = pd.read_json(sentiment_dict_path, typ='series').to_dict()
    # In this sentiment dictionary, we have the following mapping:
    #{
    #"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    #"negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    #"ambiguous": ["realization", "surprise", "curiosity", "confusion"]
    #}
    def label_emotion(row):
        active_emotions = [emotion for emotion in emotions if row.get(emotion, 0) == 1]

        if not active_emotions:
            return 'neutral'

        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'ambiguous': 0
        }

        for emotion in active_emotions:
            for sentiment, emotion_list in sentiment_dict.items():
                if emotion in emotion_list:
                    sentiment_counts[sentiment] += 1

        # Decide based on majority
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        # Optional: handle ties
        # 
        if list(sentiment_counts.values()).count(sentiment_counts[max_sentiment]) > 1:
            return 'mixed'

        return max_sentiment

    df['sentiment'] = df.apply(label_emotion, axis=1)
    # let's remove records with 'neutral' sentiment to focus on clear positive and negative examples
    df = df[df['sentiment'] != 'neutral']
    df = df[df['sentiment'] != 'mixed']

    sentiment_mapping = {
            'ambiguous':0,
            'positive':1,
            'negative':2
    }
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)

    return df
