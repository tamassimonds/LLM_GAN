
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


class StoryDataset(TorchDataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['title', 'genre', 'human_story'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'title': row['title'],
            'genre': row['genre'], 
            'human_story': row['human_story']
        }