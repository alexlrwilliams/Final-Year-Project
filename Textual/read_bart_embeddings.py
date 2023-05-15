import pandas as pd
import torch
from config import CONFIG

def get_context_bart_encoding(df):
  length = [df[df['SCENE']==idx]['SCENE'].count()-1 for idx in df['SCENE'].unique()]
  utterances = torch.load('./data/features/text/bart-context-embeddings.pt')

  cumulative_length = [length[0]]
  cumulative_value = length[0]
  for val in length[1:]:
      cumulative_value += val
      cumulative_length.append(cumulative_value)

  end_index = cumulative_length
  start_index = [0] + cumulative_length[:-1]

  test = [[utterances[idx] for idx in range(start, end)]
                for start, end in zip(start_index, end_index)]

  result = []
  for context in test:
      context_tensor = torch.stack(context)
      mean_tensor = context_tensor.mean(dim=0)
      result.append(mean_tensor)

  result_tensor = torch.stack(result)
  return result_tensor

def get_target_bart_embedding():
    return torch.load('./data/features/text/bart-embeddings.pt')


if __name__ == '__main__':
    get_context_bart_encoding(pd.read_csv("../"+CONFIG.DATA_PATH))