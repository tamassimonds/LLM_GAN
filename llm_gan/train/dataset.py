
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


class StoryDataset(TorchDataset):
    def __init__(self, csv_path: str, min_story_length: int = 100):
        self.data = pd.read_csv(csv_path)
        
        # Only check for non-null title and human_story (genre can be null)
        self.data = self.data.dropna(subset=['title', 'human_story'])
        
        # Fill null genres with a default value
        self.data['genre'] = self.data['genre'].fillna('General Fiction')
        
        # Filter out human stories that are too short
        self.data = self.data[self.data['human_story'].str.len() >= min_story_length]
        
        print(f"Loaded {len(self.data)} stories (filtered to have human stories >= {min_story_length} characters)")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'title': row['title'],
            'genre': row['genre'], 
            'human_story': row['human_story']
        }