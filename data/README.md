# Public data subset

This directory contains books 1-34 from the public Emo-FOLIO subset: 34 books
and 603 aligned image-text pages.

## `books/`

Each `book_<id>.json` stores book metadata, a relative `picture_dir`, and an ordered `pages` array. Each page record contains:

- `page`: one-based page index;
- `text`: manually corrected page text; and
- `image`: file name of the corresponding text-free page image.

## `annotations/`

Each `book_<id>_annotated.json` retains the source page record and adds:

- `primary_emotion`;
- `secondary_emotions`;
- `prev_trend_marks`; and
- `intent_tag`.

Book-level `intent_segments` store contiguous narrative stages through `start_page`,
`end_page`, and `intent`.

## `images/`

Each `book_<id>/` directory contains the text-free page images referenced by the two JSON files. Images preserve their aspect ratio and have a maximum side length of 512 pixels.

Run `python scripts/validate_release.py` from the repository root to verify alignment. See the repository-level `DATA_NOTICE.md` for copyright and reuse information.
