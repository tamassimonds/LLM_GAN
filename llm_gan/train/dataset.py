
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


class StoryDataset(TorchDataset):
    def __init__(self, csv_path: str, min_story_length: int = 100):
        # Try to read CSV with robust error handling
        try:
            self.data = pd.read_csv(csv_path)
        except pd.errors.ParserError as e:
            print(f"CSV parsing error: {e}")
            print("Trying with alternative parsing options...")
            try:
                # Try with different options for malformed CSV
                self.data = pd.read_csv(csv_path, quoting=1, escapechar='\\', on_bad_lines='skip')
                print("Successfully loaded with alternative parsing")
            except Exception as e2:
                print(f"Alternative parsing also failed: {e2}")
                raise e2
        
        self.data = self.data.dropna(subset=['title', 'genre', 'human_story'])
        
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