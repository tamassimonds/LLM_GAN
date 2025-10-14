#!/usr/bin/env python3
"""Test parsing functions to debug story and judge extraction."""

import sys
sys.path.append('/Users/tobysimonds/Documents/Code/Ai/LLM_GAN')

from llm_gan.utils.parse import parse_tags

def test_story_parsing():
    """Test story parsing with various formats."""
    
    # Test case 1: Simple story tags
    story1 = "<story>This is a simple story</story>"
    result1 = parse_tags(story1, "story")
    print(f"Test 1 - Simple tags: {result1}")
    assert result1 == "This is a simple story"
    
    # Test case 2: Story with extra content
    story2 = """user\n\nWrite a story...\n<story>This is the actual story content</story>assistant\n\nExtra content"""
    result2 = parse_tags(story2, "story")
    print(f"Test 2 - With extra content: {result2}")
    assert result2 == "This is the actual story content"
    
    # Test case 3: Your actual failing case
    story3 = """user

Write a creative story in the Science Fiction genre with the title "The Chronicles of the Cosmic Rift". 

Make the story feel human-written and engaging. Write approximately 200-300 words.

Put your story inside <story> tags like this:
<story>Your story here</story>assistant

The Chronicles of the Cosmic Rift

In the year 2256, humanity had long abandoned the ravaged planet of Kepler-62f..."""
    
    result3 = parse_tags(story3, "story")
    print(f"Test 3 - Actual case: {result3}")
    
    # Test case 4: No story tags
    story4 = "Just some plain text without story tags"
    result4 = parse_tags(story4, "story")
    print(f"Test 4 - No tags: {result4}")
    assert result4 is None
    
    # Test case 5: Multiple story tags (should return list)
    story5 = "<story>First story</story> and <story>Second story</story>"
    result5 = parse_tags(story5, "story")
    print(f"Test 5 - Multiple tags: {result5}")
    
    print("✅ Story parsing tests completed")

def test_judge_parsing():
    """Test judge answer parsing."""
    
    # Test case 1: Simple answer
    judge1 = "I think story 1 is better. <answer>1</answer>"
    result1 = parse_tags(judge1, "answer")
    print(f"Judge Test 1 - Simple: {result1}")
    assert result1 == "1"
    
    # Test case 2: Multiple answers (should return list)
    judge2 = "First I think <answer>1</answer> but actually <answer>2</answer>"
    result2 = parse_tags(judge2, "answer")
    print(f"Judge Test 2 - Multiple: {result2}")
    
    # Test case 3: No answer tags
    judge3 = "Some reasoning but no answer tags"
    result3 = parse_tags(judge3, "answer")
    print(f"Judge Test 3 - No tags: {result3}")
    assert result3 is None
    
    # Test case 4: Your repetitive output case
    judge4 = """<OUTPUT>
    <OUTPUT>1</OUTPUT>
    </OUTPUT>
    </ASSISTANT>
    <OUTPUT>
    <OUTPUT>2</OUTPUT>
    </OUTPUT>"""
    result4 = parse_tags(judge4, "answer")
    print(f"Judge Test 4 - Complex case: {result4}")
    
    # Test case 5: Test the "take last answer" logic
    def test_last_answer_logic(judge_output):
        all_answers = parse_tags(judge_output, "answer")
        
        # If multiple answers, take the last one
        if isinstance(all_answers, list) and len(all_answers) > 0:
            answer = all_answers[-1]  # Take last answer
        else:
            answer = all_answers  # Single answer or None
        
        if answer == "1":
            return 0
        elif answer == "2":
            return 1
        else:
            return -1
    
    judge5 = "I think <answer>1</answer> wait actually <answer>2</answer>"
    result5 = test_last_answer_logic(judge5)
    print(f"Judge Test 5 - Last answer logic: {result5} (should be 1 for answer=2)")
    
    print("✅ Judge parsing tests completed")

def test_improved_story_extraction():
    """Test improved story extraction logic."""
    
    story_text = """user

Write a creative story in the Science Fiction genre with the title "The Chronicles of the Cosmic Rift". 

Make the story feel human-written and engaging. Write approximately 200-300 words.

Put your story inside <story> tags like this:
<story>Your story here</story>assistant

The Chronicles of the Cosmic Rift

In the year 2256, humanity had long abandoned the ravaged planet of Kepler-62f, a world ravaged by war and environmental disaster. The once-thriving colony had been reduced to a mere shadow of its former self, a testament to the devastating consequences of humanity's hubris. Yet, in the depths of the planet's vast underground tunnels, a small group of survivors clung to the hope of a better tomorrow."""
    
    # Use the new improved logic
    all_stories = parse_tags(story_text, "story")
    story = None
    
    if isinstance(all_stories, list):
        # Multiple story tags found - filter out the example
        for s in all_stories:
            if s and "Your story here" not in s and len(s) > 50:
                story = s
                break
    elif all_stories and "Your story here" not in all_stories:
        story = all_stories
    
    print(f"Story from filtered tags: {story}")
    
    if story is None:
        # Look for content after assistant header - this is where the actual story is
        if "assistant" in story_text:
            # Split on assistant and get the last part
            parts = story_text.split("assistant")
            if len(parts) > 1:
                content = parts[-1].strip()
                # Take the first substantial paragraph
                lines = content.split('\n')
                story_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and len(line) > 10:
                        story_lines.append(line)
                if story_lines:
                    story = ' '.join(story_lines)
                    print(f"Story from assistant content: {story[:100]}...")
    
    if not story:
        story = "Fallback story"
    
    print(f"Final extracted story: {story[:200]}...")
    return story

if __name__ == "__main__":
    print("🧪 Testing parsing functions...\n")
    
    try:
        test_story_parsing()
        print()
        test_judge_parsing()
        print()
        print("🔧 Testing improved extraction:")
        test_improved_story_extraction()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()