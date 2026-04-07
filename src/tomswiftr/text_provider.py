import io
import re
from typing import Union

import requests_cache
from requests import get


class GutenbergTextProvider:
    _cache_installed = False
    _cache_name = "tm_gutenberg"

    def __init__(self) -> None:
        if not self.__class__._cache_installed:
            requests_cache.install_cache(
                self.__class__._cache_name, backend="filesystem"
            )
            self.__class__._cache_installed = True
        self.text = ""

    def get_text(self, to_char: Union[int, None] = None) -> str:
        # TODO be a generator over SpaCy "sentencizer"
        if to_char is not None:
            return self.text[:to_char]

        return self.text

    def fetch(self, book_id: int) -> None:
        res = get(f"https://gutenberg.org/ebooks/{book_id}.txt.utf-8", timeout=30)
        self.text = self.preprocess(res.text)

    def preprocess(self, text: str) -> str:
        # First pass: find start/end boundaries without materializing all lines.
        endpoints = {"start": 0, "end": None}
        n_lines = 0

        # Tracks which end rule matched so slicing can be marker-aware.
        end_marker = None
        chapter_one_re = re.compile(
            r"^\s*(?:CHAPTER I(?:[.:])?|I\.)\s*$", re.IGNORECASE
        )
        start_markers = {"chapter": None, "gutenberg": None, "produced_by": None}

        for k, raw_line in enumerate(io.StringIO(text)):
            n_lines = k + 1
            line = raw_line.rstrip("\r\n")

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
            endpoints["end"] if end_marker in {"the_end", "gutenberg_end"} else n_lines
        )

        # Second pass: stream cleaned lines, avoiding large intermediate copies.
        return " ".join(
            raw_line.rstrip("\r\n")
            for i, raw_line in enumerate(io.StringIO(text))
            if start_idx <= i < end_idx
        )
