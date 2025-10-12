import unittest

from llm_gan.utils.parse import parse_tags


class ParseTagsTests(unittest.TestCase):
    def test_parse_all_tags_returns_dict(self):
        text = "<story> Once upon a time </story>\n<answer>1</answer>"
        result = parse_tags(text)
        self.assertEqual(result, {"story": "Once upon a time", "answer": "1"})

    def test_request_single_tag_returns_string(self):
        text = "<story> content </story> <answer>2</answer>"
        result = parse_tags(text, "answer")
        self.assertEqual(result, "2")

    def test_request_multiple_tags_returns_tuple(self):
        text = "<story>Foo</story><reason>Because</reason><answer>1</answer>"
        story, answer = parse_tags(text, ["story", "answer"])
        self.assertEqual(story, "Foo")
        self.assertEqual(answer, "1")

    def test_missing_tag_returns_none(self):
        text = "<story>Foo</story>"
        story, answer = parse_tags(text, ["story", "answer"])
        self.assertEqual(story, "Foo")
        self.assertIsNone(answer)

    def test_strict_mode_raises_for_missing_tag(self):
        text = "<story>Foo</story>"
        with self.assertRaises(ValueError):
            parse_tags(text, ["story", "answer"], strict=True)

    def test_multiple_tag_occurrences_return_list(self):
        text = "<tag>a</tag><tag>b</tag><tag>c</tag>"
        result = parse_tags(text, "tag")
        self.assertEqual(result, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
