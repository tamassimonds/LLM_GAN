"""Smart sampling of stories for comparison evaluation."""

import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from .log_parser import GeneratedStory


@dataclass 
class StoryPair:
    """A pair of stories for comparison."""
    story_a: GeneratedStory
    story_b: GeneratedStory
    comparison_id: str
    
    def __post_init__(self):
        if not self.comparison_id:
            self.comparison_id = f"{self.story_a.step}_{self.story_a.story_idx}_vs_{self.story_b.step}_{self.story_b.story_idx}"


class StorySampler:
    """Smart sampling of stories for evaluation comparisons."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize sampler with optional random seed for reproducibility."""
        if random_seed is not None:
            random.seed(random_seed)
    
    def sample_balanced_pairs(self, stories_a: List[GeneratedStory], 
                             stories_b: List[GeneratedStory],
                             num_pairs: int,
                             strategy: str = "balanced") -> List[StoryPair]:
        """
        Sample story pairs for comparison using different strategies.
        
        Args:
            stories_a: Stories from first training run
            stories_b: Stories from second training run  
            num_pairs: Number of pairs to sample
            strategy: Sampling strategy ("balanced", "random", "genre_matched", "epoch_matched")
            
        Returns:
            List of StoryPair objects
        """
        if strategy == "balanced":
            return self._sample_balanced(stories_a, stories_b, num_pairs)
        elif strategy == "random":
            return self._sample_random(stories_a, stories_b, num_pairs)
        elif strategy == "genre_matched":
            return self._sample_genre_matched(stories_a, stories_b, num_pairs)
        elif strategy == "epoch_matched":
            return self._sample_epoch_matched(stories_a, stories_b, num_pairs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_balanced(self, stories_a: List[GeneratedStory], 
                        stories_b: List[GeneratedStory], 
                        num_pairs: int) -> List[StoryPair]:
        """Sample pairs with balanced representation across genres and epochs."""
        
        # Group stories by genre for balanced sampling
        genres_a = self._group_by_genre(stories_a)
        genres_b = self._group_by_genre(stories_b)
        
        # Find common genres
        common_genres = set(genres_a.keys()) & set(genres_b.keys())
        if not common_genres:
            print("Warning: No common genres found, falling back to random sampling")
            return self._sample_random(stories_a, stories_b, num_pairs)
        
        pairs = []
        pairs_per_genre = max(1, num_pairs // len(common_genres))
        
        for genre in common_genres:
            genre_stories_a = genres_a[genre]
            genre_stories_b = genres_b[genre]
            
            # Sample pairs for this genre
            for _ in range(min(pairs_per_genre, len(genre_stories_a), len(genre_stories_b))):
                if len(pairs) >= num_pairs:
                    break
                    
                story_a = random.choice(genre_stories_a)
                story_b = random.choice(genre_stories_b)
                
                pair = StoryPair(
                    story_a=story_a,
                    story_b=story_b,
                    comparison_id=f"balanced_{len(pairs)}"
                )
                pairs.append(pair)
                
                # Remove selected stories to avoid duplicates
                genre_stories_a.remove(story_a)
                genre_stories_b.remove(story_b)
        
        # Fill remaining pairs randomly if needed
        while len(pairs) < num_pairs and stories_a and stories_b:
            remaining_a = [s for s in stories_a if not any(p.story_a == s for p in pairs)]
            remaining_b = [s for s in stories_b if not any(p.story_b == s for p in pairs)]
            
            if not remaining_a or not remaining_b:
                break
                
            story_a = random.choice(remaining_a)
            story_b = random.choice(remaining_b)
            
            pair = StoryPair(
                story_a=story_a,
                story_b=story_b,
                comparison_id=f"balanced_extra_{len(pairs)}"
            )
            pairs.append(pair)
        
        return pairs
    
    def _sample_random(self, stories_a: List[GeneratedStory], 
                      stories_b: List[GeneratedStory], 
                      num_pairs: int) -> List[StoryPair]:
        """Sample pairs completely randomly."""
        pairs = []
        
        for i in range(min(num_pairs, len(stories_a), len(stories_b))):
            story_a = random.choice(stories_a)
            story_b = random.choice(stories_b)
            
            pair = StoryPair(
                story_a=story_a,
                story_b=story_b,
                comparison_id=f"random_{i}"
            )
            pairs.append(pair)
        
        return pairs
    
    def _sample_genre_matched(self, stories_a: List[GeneratedStory], 
                             stories_b: List[GeneratedStory], 
                             num_pairs: int) -> List[StoryPair]:
        """Sample pairs ensuring both stories are from the same genre."""
        genres_a = self._group_by_genre(stories_a)
        genres_b = self._group_by_genre(stories_b)
        
        pairs = []
        common_genres = set(genres_a.keys()) & set(genres_b.keys())
        
        for genre in common_genres:
            genre_stories_a = genres_a[genre]
            genre_stories_b = genres_b[genre]
            
            # Sample all possible pairs for this genre, up to our limit
            max_pairs_for_genre = min(
                len(genre_stories_a), 
                len(genre_stories_b),
                num_pairs - len(pairs)
            )
            
            for _ in range(max_pairs_for_genre):
                if len(pairs) >= num_pairs:
                    break
                    
                story_a = random.choice(genre_stories_a)
                story_b = random.choice(genre_stories_b)
                
                pair = StoryPair(
                    story_a=story_a,
                    story_b=story_b,
                    comparison_id=f"genre_{genre}_{len(pairs)}"
                )
                pairs.append(pair)
                
                genre_stories_a.remove(story_a)
                genre_stories_b.remove(story_b)
        
        return pairs
    
    def _sample_epoch_matched(self, stories_a: List[GeneratedStory], 
                             stories_b: List[GeneratedStory], 
                             num_pairs: int) -> List[StoryPair]:
        """Sample pairs from similar training epochs."""
        epochs_a = self._group_by_epoch(stories_a)
        epochs_b = self._group_by_epoch(stories_b)
        
        pairs = []
        common_epochs = set(epochs_a.keys()) & set(epochs_b.keys())
        
        for epoch in sorted(common_epochs):
            epoch_stories_a = epochs_a[epoch]
            epoch_stories_b = epochs_b[epoch]
            
            max_pairs_for_epoch = min(
                len(epoch_stories_a), 
                len(epoch_stories_b),
                num_pairs - len(pairs)
            )
            
            for _ in range(max_pairs_for_epoch):
                if len(pairs) >= num_pairs:
                    break
                    
                story_a = random.choice(epoch_stories_a)
                story_b = random.choice(epoch_stories_b)
                
                pair = StoryPair(
                    story_a=story_a,
                    story_b=story_b,
                    comparison_id=f"epoch_{epoch}_{len(pairs)}"
                )
                pairs.append(pair)
                
                epoch_stories_a.remove(story_a)
                epoch_stories_b.remove(story_b)
        
        return pairs
    
    def _group_by_genre(self, stories: List[GeneratedStory]) -> Dict[str, List[GeneratedStory]]:
        """Group stories by genre."""
        grouped = defaultdict(list)
        for story in stories:
            grouped[story.genre].append(story)
        return dict(grouped)
    
    def _group_by_epoch(self, stories: List[GeneratedStory]) -> Dict[int, List[GeneratedStory]]:
        """Group stories by epoch."""
        grouped = defaultdict(list)
        for story in stories:
            grouped[story.epoch].append(story)
        return dict(grouped)
    
    def get_sampling_stats(self, pairs: List[StoryPair]) -> Dict[str, Any]:
        """Get statistics about the sampled pairs."""
        if not pairs:
            return {}
        
        genres_a = [pair.story_a.genre for pair in pairs]
        genres_b = [pair.story_b.genre for pair in pairs]
        epochs_a = [pair.story_a.epoch for pair in pairs]
        epochs_b = [pair.story_b.epoch for pair in pairs]
        
        # Count genre matches
        genre_matches = sum(1 for a, b in zip(genres_a, genres_b) if a == b)
        
        # Count epoch matches  
        epoch_matches = sum(1 for a, b in zip(epochs_a, epochs_b) if a == b)
        
        stats = {
            'total_pairs': len(pairs),
            'genre_matches': genre_matches,
            'genre_match_rate': genre_matches / len(pairs),
            'epoch_matches': epoch_matches,
            'epoch_match_rate': epoch_matches / len(pairs),
            'genres_a_distribution': {genre: genres_a.count(genre) for genre in set(genres_a)},
            'genres_b_distribution': {genre: genres_b.count(genre) for genre in set(genres_b)},
            'epoch_range_a': (min(epochs_a), max(epochs_a)),
            'epoch_range_b': (min(epochs_b), max(epochs_b)),
        }
        
        return stats


def sample_story_pairs(stories_a: List[GeneratedStory], 
                      stories_b: List[GeneratedStory],
                      num_pairs: int,
                      strategy: str = "balanced",
                      random_seed: Optional[int] = None) -> Tuple[List[StoryPair], Dict[str, Any]]:
    """
    Convenience function to sample story pairs and return with stats.
    
    Args:
        stories_a: Stories from first training run
        stories_b: Stories from second training run
        num_pairs: Number of pairs to sample
        strategy: Sampling strategy
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (pairs, sampling_stats)
    """
    sampler = StorySampler(random_seed=random_seed)
    pairs = sampler.sample_balanced_pairs(stories_a, stories_b, num_pairs, strategy)
    stats = sampler.get_sampling_stats(pairs)
    return pairs, stats