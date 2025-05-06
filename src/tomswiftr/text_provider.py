import re
from typing import List  # , Dict, Iterable, Tuple, Union

import requests_cache
from requests import get


class GutenbergTextProvider:
    def __init__(self) -> None:
        self.text = ""

    def get_text(self, to_char=None):
        # TODO be a generator over SpaCy "sentencizer"
        if to_char is not None:
            return self.text[:to_char]
        else:
            return self.text

    def fetch(self, id: int) -> None:
        requests_cache.install_cache(f"tm_{id}", backend="filesystem")
        res = get(f"https://gutenberg.org/ebooks/{id}.txt.utf-8", timeout=30)

        self.text = self.preprocess(res.text)

    def preprocess(self, text: str) -> str:
        lines = text.splitlines()

        # Chop off header and footer
        endpoints = {"start": 0, "end": len(lines)}

        for k, line in enumerate(lines):
            if line.startswith("*** START OF THE PROJECT"):
                endpoints["start"] = k

            if line.startswith("Produced by") and k > endpoints["start"]:
                # Update it since sometimes the (gutenberg text) produced by
                # line comes after the actual, official start!
                endpoints["start"] = k

            # This is a better feature...
            # if line.startswith("CHAPTER I") and k > endpoints["start"]:
            if re.match(r"^CHAPTER I\b", line) and k > endpoints["start"]:
                endpoints["start"] = k

            if line.startswith("THE END"):
                endpoints["end"] = k

                # halt here since there could be much advertising text after this,
                # if THE END exists in the text; usually does, but not guaranteed.
                break

            if line.startswith("*** END OF THE PROJECT GUTENBERG"):
                endpoints["end"] = k

        cleaned_text: List[str] = lines[endpoints["start"] + 1 : endpoints["end"] - 1]
        return " ".join(cleaned_text)
