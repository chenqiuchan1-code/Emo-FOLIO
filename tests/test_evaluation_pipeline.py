import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvaluationPipelineTest(unittest.TestCase):
    def test_postprocess_and_evaluate_smoke(self):
        book = json.loads((REPO_ROOT / "data/books/book_1.json").read_text(encoding="utf-8"))
        pages = []
        for row in book["pages"]:
            page = int(row["page"])
            pages.append({
                "page": page,
                "emotion_intensity": {
                    "快乐": page % 6,
                    "悲伤": 0,
                    "恐惧": 0,
                    "愤怒": 0,
                    "惊讶": 0,
                    "平静": 2,
                    "好奇": 3,
                },
                "intent_tag": "信息传递",
            })

        raw = {
            "book_id": "book_1",
            "intent_segments": [{
                "start_page": 1,
                "end_page": len(pages),
                "intent": "信息传递",
            }],
            "pages": pages,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            post_dir = root / "post"
            eval_dir = root / "eval"
            raw_dir.mkdir()
            (raw_dir / "book_1_MOSAIC_merged.json").write_text(
                json.dumps(raw, ensure_ascii=False), encoding="utf-8"
            )

            subprocess.run([
                sys.executable, "-m", "scripts.evaluation.postprocess",
                "--in_dir", str(raw_dir),
                "--out_dir", str(post_dir),
                "--secondary_threshold", "1",
            ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            normalized = json.loads(
                (post_dir / "book_1_MOSAIC_merged_normalized.json").read_text(encoding="utf-8")
            )
            # Trends are derived from intensity differences for all emotions;
            # the evaluator subsequently selects the gold-anchored categories.
            self.assertEqual(len(normalized["pages"][1]["prev_trend_marks"]), 7)

            subprocess.run([
                sys.executable, "-m", "scripts.evaluation.evaluate",
                "--gold_dir", str(REPO_ROOT / "data/annotations"),
                "--pred_dir", str(post_dir),
                "--out_dir", str(eval_dir),
            ], cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            self.assertTrue((eval_dir / "overall_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
