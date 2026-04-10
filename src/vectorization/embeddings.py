import numpy as np

def get_text_embeddings(text, model):
  vectors = [model[word] for word in text.split() if word in model]
  return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
