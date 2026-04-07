import re
from typing import List, Union  # , Dict, Iterable, Tuple, Union

import requests_cache
from requests import get


class GutenbergTextProvider:
    def __init__(self) -> None:
        self.text = ""

    def get_text(self, to_char: Union[int, None] = None) -> str:
        # TODO be a generator over SpaCy "sentencizer"
        if to_char is not None:
            return self.text[:to_char]

        return self.text

    def fetch(self, book_id: int) -> None:
        requests_cache.install_cache(f"tm_{book_id}", backend="filesystem")
        res = get(f"https://gutenberg.org/ebooks/{book_id}.txt.utf-8", timeout=30)
        self.text = self.preprocess(res.text)

    def preprocess(self, text: str) -> str:
        lines = text.splitlines()

        # Chop off header and footer
        endpoints = {"start": 0, "end": len(lines)}
        # Tracks which end rule matched so slicing can be marker-aware.
        end_marker = None
        chapter_one_re = re.compile(
            r"^\s*(?:CHAPTER I(?:[.:])?|I\.)\s*$", re.IGNORECASE
        )
        start_markers = {"chapter": None, "gutenberg": None, "produced_by": None}

        for k, line in enumerate(lines):
            if start_markers["chapter"] is None and chapter_one_re.match(line):
                start_markers["chapter"] = k

            if start_markers["gutenberg"] is None and line.startswith(
                "*** START OF THE PROJECT"
            ):
                start_markers["gutenberg"] = k

            if start_markers["produced_by"] is None and line.startswith("Produced by"):
                start_markers["produced_by"] = k

            if line.startswith("THE END"):
                endpoints["end"] = k
                end_marker = "the_end"

                # halt here since there could be much advertising text after this,
                # if THE END exists in the text; usually does, but not guaranteed.
                break

            if line.startswith("*** END OF THE PROJECT GUTENBERG"):
                endpoints["end"] = k
                end_marker = "gutenberg_end"

        start_marker = "default"
        if start_markers["chapter"] is not None:
            endpoints["start"] = start_markers["chapter"]
            start_marker = "chapter"
        elif start_markers["gutenberg"] is not None:
            endpoints["start"] = start_markers["gutenberg"]
            start_marker = "gutenberg"
        elif start_markers["produced_by"] is not None:
            endpoints["start"] = start_markers["produced_by"]
            start_marker = "produced_by"

        # Skip metadata-like start lines, but keep chapter headers in the output.
        start_idx = (
            endpoints["start"] + 1
            if start_marker in {"gutenberg", "produced_by"}
            else endpoints["start"]
        )
        # If no end marker matched, keep the full tail (don't drop the last line).
        end_idx = (
            endpoints["end"]
            if end_marker in {"the_end", "gutenberg_end"}
            else len(lines)
        )

        # TODO not so great because we're copying the whole book text in memory...
        cleaned_text: List[str] = lines[start_idx:end_idx]

        # Reconstruct the text without line breaks
        return " ".join(cleaned_text)
