"""Parse training logs to extract generated stories and metadata."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GeneratedStory:
    """Container for a generated story with metadata."""
    story: str
    title: str
    genre: str
    epoch: int
    batch_idx: int
    step: int
    story_idx: int  # Index within the batch
    generator_reward: float
    judge_accuracy: float
    story_length: int
    
    def __post_init__(self):
        self.story_length = len(self.story)


class LogParser:
    """Parse training log directories to extract generated stories."""
    
    def __init__(self, log_dir: str):
        """Initialize with path to training log directory."""
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    def parse_all_stories(self, min_story_length: int = 50, 
                         filter_placeholders: bool = True) -> List[GeneratedStory]:
        """
        Parse all batch log files and extract generated stories.
        
        Args:
            min_story_length: Minimum story length to include
            filter_placeholders: Whether to filter out placeholder stories
            
        Returns:
            List of GeneratedStory objects
        """
        stories = []
        
        # Find all batch log files
        batch_files = list(self.log_dir.glob("batch_*.json"))
        print(f"Found {len(batch_files)} batch log files")
        
        for batch_file in sorted(batch_files):
            try:
                batch_stories = self._parse_batch_file(batch_file, min_story_length, filter_placeholders)
                stories.extend(batch_stories)
            except Exception as e:
                print(f"Error parsing {batch_file}: {e}")
                continue
        
        print(f"Extracted {len(stories)} stories from {self.log_dir}")
        return stories
    
    def _parse_batch_file(self, batch_file: Path, min_story_length: int, 
                         filter_placeholders: bool) -> List[GeneratedStory]:
        """Parse a single batch log file."""
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        stories = []
        epoch = batch_data.get('epoch', 0)
        batch_idx = batch_data.get('batch_idx', 0)
        step = batch_data.get('step', 0)
        judge_accuracy = batch_data.get('judge_accuracy', 0.0)
        
        # Extract generated stories
        generated_stories = batch_data.get('generated_stories', [])
        titles = batch_data.get('titles', [])
        genres = batch_data.get('genres', [])
        generator_rewards = batch_data.get('generator_rewards', [])
        
        # Ensure all lists have the same length
        min_len = min(len(generated_stories), len(titles), len(genres))
        if len(generator_rewards) < min_len:
            generator_rewards.extend([0.0] * (min_len - len(generator_rewards)))
        
        for i in range(min_len):
            story = generated_stories[i]
            title = titles[i] if i < len(titles) else "Unknown"
            genre = genres[i] if i < len(genres) else "Unknown"
            reward = generator_rewards[i] if i < len(generator_rewards) else 0.0
            
            # Apply filters
            if len(story) < min_story_length:
                continue
                
            if filter_placeholders and self._is_placeholder_story(story):
                continue
            
            story_obj = GeneratedStory(
                story=story,
                title=title,
                genre=genre,
                epoch=epoch,
                batch_idx=batch_idx,
                step=step,
                story_idx=i,
                generator_reward=reward,
                judge_accuracy=judge_accuracy,
                story_length=len(story)
            )
            stories.append(story_obj)
        
        return stories
    
    def _is_placeholder_story(self, story: str) -> bool:
        """Check if story appears to be a placeholder."""
        story_lower = story.lower().strip()
        
        # Check for common placeholder patterns
        placeholder_patterns = [
            "your story here",
            "**your story here**",
            "*your story here*",
            "write your story",
            "story here",
            "[story]",
            "insert story here",
            "add your story",
            "put your story here",
            "a general fiction story",
            "a story titled"
        ]
        
        # Check if story IS a placeholder (not just contains)
        for pattern in placeholder_patterns:
            if story_lower == pattern or story_lower.startswith(pattern.strip('*')):
                return True
        
        # Check for very short generic stories
        if len(story.strip()) < 50 and any(word in story_lower for word in ["story", "title", "genre"]):
            return True
            
        return False
    
    def get_story_stats(self, stories: List[GeneratedStory]) -> Dict:
        """Get statistics about the parsed stories."""
        if not stories:
            return {}
        
        # Group by genre
        genres = {}
        epochs = set()
        story_lengths = []
        rewards = []
        
        for story in stories:
            genres[story.genre] = genres.get(story.genre, 0) + 1
            epochs.add(story.epoch)
            story_lengths.append(story.story_length)
            rewards.append(story.generator_reward)
        
        stats = {
            'total_stories': len(stories),
            'unique_epochs': len(epochs),
            'epoch_range': (min(epochs), max(epochs)) if epochs else (0, 0),
            'genre_distribution': genres,
            'avg_story_length': sum(story_lengths) / len(story_lengths),
            'min_story_length': min(story_lengths),
            'max_story_length': max(story_lengths),
            'avg_generator_reward': sum(rewards) / len(rewards),
            'story_length_distribution': {
                'under_100': sum(1 for l in story_lengths if l < 100),
                '100_300': sum(1 for l in story_lengths if 100 <= l < 300),
                '300_500': sum(1 for l in story_lengths if 300 <= l < 500),
                'over_500': sum(1 for l in story_lengths if l >= 500)
            }
        }
        
        return stats


def parse_log_directory(log_dir: str, min_story_length: int = 50, 
                       filter_placeholders: bool = True) -> Tuple[List[GeneratedStory], Dict]:
    """
    Convenience function to parse a log directory and return stories with stats.
    
    Args:
        log_dir: Path to training log directory
        min_story_length: Minimum story length to include
        filter_placeholders: Whether to filter out placeholder stories
        
    Returns:
        Tuple of (stories, stats)
    """
    parser = LogParser(log_dir)
    stories = parser.parse_all_stories(min_story_length, filter_placeholders)
    stats = parser.get_story_stats(stories)
    return stories, stats