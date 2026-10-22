import unittest
from backend.prompt_enhancer import keep_mentions, missing_mentions


class MentionTests(unittest.TestCase):
    def test_lost_references_come_back_with_their_label(self):
        prompt = 'the woman from <image1> on a bench with the dog from <image2>, full body'
        self.assertEqual(missing_mentions(prompt, 'A woman and a dog.'), ['<image1>', '<image2>'])
        self.assertEqual(keep_mentions(prompt, 'A woman and a dog.'),
                         'A woman and a dog. Use the woman from <image1>. Use the dog from <image2>.')

    def test_kept_references_change_nothing(self):
        answer = 'The woman from <image1> sits; the dog from <image2> waits.'
        self.assertEqual(keep_mentions('the woman from <image1> and the dog from <image2>', answer), answer)
        self.assertEqual(keep_mentions('a cat', 'A cat.'), 'A cat.')


if __name__ == '__main__':
    unittest.main()
