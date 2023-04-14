import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartModel
import config

FILE_PATH = "../data/extended_dataset.csv"
DEVICE = config.DEVICE
BATCH_SIZE = 16

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large').to(DEVICE)

def extract_bart_embeddings(df, column_name):
  sentences = df[column_name].tolist()
  input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(DEVICE)

  num_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
  batches = [input_ids[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(num_batches)]

  embeddings = []
  for batch in batches:
    with torch.no_grad():
        outputs = model(input_ids=batch)
        last_hidden_states = outputs.last_hidden_state
        mean_last_4_layers = torch.mean(last_hidden_states[:, -4:, :], dim=1)
        embeddings.append(mean_last_4_layers)
  return torch.cat(embeddings, dim=0)


df = pd.read_csv(FILE_PATH)
utterances = df[df['Sarcasm'].notna()]
context = df[df['Sarcasm'].isna()]

context_embeddings = extract_bart_embeddings(context, 'SENTENCE')
utterance_embeddings = extract_bart_embeddings(utterances, 'SENTENCE')

torch.save({
    'context': context_embeddings,
    'utterance': utterance_embeddings
})
