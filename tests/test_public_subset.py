import unittest
from pathlib import Path

from scripts.validate_release import validate


class PublicSubsetTest(unittest.TestCase):
    def test_public_subset_alignment(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(validate(repo_root), [])


if __name__ == "__main__":
    unittest.main()
