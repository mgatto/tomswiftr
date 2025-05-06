"""Tom Swifter ngram model and supporting classes

Raises:
    TypeError: _description_

Returns:
    _type_: _description_
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import requests_cache
from numpy.typing import NDArray
from requests import get

# ex. [("I", "am"), ("she", "is")]
NgramType = Tuple[str, ...]

# TODO use Apache Arrow/Parquet
# TODO save the probabilities a scipy sparse matrix -> npz (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html)
# TODO save the vocabulary as a numpy array -> compressed npz (https://numpy.org/doc/2.0/reference/generated/numpy.savez_compressed.html)
"""
For most cases: JSON is the safe and versatile choice, especially if you need to share data with other systems or want human-readable output.
For performance-critical applications: MessagePack is a good option if you don't need human-readable output.
For large datasets: Parquet is ideal for efficient storage and querying.
For Python-specific objects: Pickle is suitable, but be cautious about security risks when loading untrusted data.
"""


class Vocabulary:
    """
    Stateful vocabulary.
    Provides a mapping from term to ID and a reverse mapping of ID to term.
    """

    # symbol for unknown terms
    UNKNOWN = "<UNK>"

    # TODO decide where to save the indexes t2i and i2t
    # Pylance complains: default arg [] is dangerous, but only if the fun modifies it.
    def __init__(self, terms: Iterable[str] = [], min_count: int = 1):
        self.t2i: Dict[str, int] = Vocabulary.create_t2i(terms, min_count=min_count)
        self.i2t: Dict[int, str] = Vocabulary.create_i2t(self.t2i)

    # see https://www.python.org/dev/peps/pep-0484/#forward-references
    @staticmethod
    def from_sentences(
        sentences: Iterable[Iterable[str]], min_count: int = 1
    ) -> "Vocabulary":
        """
        Convenience method
        for converting a sequence of tokenized sentences to a Vocabulary instance
        """
        return Vocabulary(
            terms=[term for sentence in sentences for term in sentence],
            min_count=min_count,
        )

    def id_for(self, term: str) -> int:
        """
        Looks up ID for term using self.t2i.
        If the feature is unknown, returns ID of Vocabulary.UNKNOWN.
        """
        try:
            return self.t2i[term]
        except KeyError:
            return 0  # = canonical id of Vocabulary.UNKNOWN

    def term_for(self, term_id: int) -> Union[str, None]:
        """
        Looks up term corresponding to term_id.
        If term_id is unknown, returns None.
        """
        try:
            return self.i2t[term_id]
        except KeyError:
            return None

    @property
    def terms(self) -> List[str]:
        """_summary_

        Returns:
            List[str]: _description_
        """
        return [self.i2t[i] for i in range(len(self.i2t))]

    @staticmethod
    def create_t2i(terms: Iterable[str], min_count: int = 1) -> Dict[str, int]:
        """
        Takes a flat iterable of terms (i.e., unigrams) and returns
        a dictionary of term -> int. Assumes terms have already
        been normalized.

        If the frequency of a term is less than min_count,
        do not include the term in the vocabulary

        Requirements:
        - First term in vocabulary (ID 0) is reserved for Vocabulary.UNKNOWN.
        - Sort the features alphabetically
        - Only include terms occurring >= min_count
        """
        # terms must be strings
        if not all(isinstance(term, str) for term in terms):
            raise TypeError("terms must be strings")

        counted_terms = Counter(terms)
        filtered_terms = Counter(
            {term: n for term, n in counted_terms.items() if n >= min_count}
        )

        # Counter does keep its keys in order of insertion :-)
        return {Vocabulary.UNKNOWN: 0} | {
            t: i for i, t in enumerate(sorted(list(filtered_terms.keys())), 1)
        }

    @staticmethod
    def create_i2t(t2i: Dict[str, int]) -> Dict[int, str]:
        """
        Takes a dict of str -> int and returns a reverse mapping of int -> str.
        """
        return {i: t for (t, i) in t2i.items()}

    def empty_vector(self) -> NDArray[np.float64]:
        """
        Creates an empty numpy array based on the vocabulary of terms
        """
        return np.zeros(len(self.t2i), dtype="float")

    def __len__(self):
        """
        Defines what should happen when `len` is called on an instance of
        this class.
        """
        return len(self.t2i)

    def __contains__(self, other):
        """
        Example:

        v = Vocabulary(["I", "am"])
        assert "am" in v
        """
        return True if other in self.t2i else False


class LanguageModel:
    """
    An _n_-gram language model using an _n_ - 1 order Markov assumption.
    """

    def __init__(
        self,
        corpus: Iterable[Iterable[str]],
        n=2,
        min_count=1,
        use_start_end: bool = False,
    ):
        assert n >= 2
        self.n = n
        self.use_start_end = use_start_end
        # though not stored as an instance attribute, we need this temporarily to calculate other attributes
        ngrams: Iterable[Tuple[str, ...]] = [
            gram
            for sentence in corpus
            for gram in self.ngrams_for(
                n=self.n, tokens=sentence, use_start_end=self.use_start_end
            )
        ]
        self.vocab: Vocabulary = Vocabulary.from_sentences(corpus, min_count)
        self.pdist: Dict[Tuple[str, ...], NDArray[np.float64]] = (
            self.make_conditional_probs(ngrams, self.vocab)
        )

    def make_conditional_probs(
        self, ngrams: Iterable[NgramType], vocab: Vocabulary
    ) -> Dict[Tuple[str, ...], NDArray[np.float64]]:
        """
        Takes a sequence of n-grams and a vocabulary
        Returns a dictionary of conditional probability distributions for each n-gram's history
        """
        # if ngram length is 2, then tuple len must become n-1 ...per Jurafsky. TODO but why?
        new_tuples = [(t[:-1]) for t in ngrams]
        cond_probs = {k: np.zeros(len(vocab.terms), dtype=float) for k in new_tuples}

        for (
            tup,
            probs,
        ) in (
            cond_probs.items()
        ):  # TODO later: may want to loop over the ngrams instead?
            # ALWAYS select the next-to-last term in the tuple to search with!
            term = tup[-1]

            # find the tuples in which tup[-1] occurs
            occurrences = Counter()
            tuples_with_term = [t for t in ngrams if term in t]

            # Does term precede any vocab.terms?
            for twt in tuples_with_term:
                # means our term's index in the tuple is ONE less than the index of the vocab
                # all others are zero, so only update the [id_for] in probs with the count
                try:
                    t_idx = twt.index(term)

                    # only consider when our term is the next to last item in the ngram; TODO why?
                    if (
                        t_idx == len(twt) - 1 - 1
                    ):  # doubling -1 since index() is zero-based while len() counts from 1
                        # then count the next term
                        occurrences.update(
                            [twt[t_idx + 1]]
                        )  # making it an iterable to satisfy update() is dumb
                except IndexError:
                    continue

            total = occurrences.total()
            for term, count in occurrences.items():
                if vocab.id_for(term) == 0:
                    # and divide by total vocab.terms to normalize it
                    probs[0] = probs[0] + (count / total)
                else:
                    # critical to keep the order
                    probs[vocab.id_for(term)] = count / total

            # Now just shunt probs back into the dict!
            cond_probs[tup] = probs

        return cond_probs

    def ngrams_for(
        # the size of the n-gram
        self,
        n: int,
        # a list of tokens
        tokens: List[str],
        # whether or not to use the start and end symbols
        use_start_end: bool = True,
        # the symbol to use for the start of the sequence (assuming user_start_end is true)
        start_symbol: str = "<S>",
        # the symbol to use for the end of the sequence (assuming user_start_end is true)
        end_symbol: str = "</S>",
    ) -> List[Tuple[str]]:
        """
        Generates a list of n-gram tuples for the provided sequence of tokens.
        """
        tokens_size = len(tokens)
        possible_ngrams: int = tokens_size + 2 if use_start_end else tokens_size

        if n == 0 or n > possible_ngrams:
            return []

        if use_start_end:
            # Calculate final tokens list, accounting for optional start/end symbols
            starts = [start_symbol] * (n - 1)
            ends = [end_symbol] * (n - 1)

            # Copy tokens to avoid shifting elements by n, n times = wasteful
            compiled_tokens = starts + tokens + ends
        else:
            compiled_tokens = tokens

        # Now loop to shift the window of size n to generate n-grams
        return [
            tuple(window)
            for window in [
                compiled_tokens[i : i + n]
                for i in range(
                    0, len(compiled_tokens) - (n - 1)
                )  # parentheses are key for (n - 1)
            ]
        ]

    def cond_prob(self, term: str, given: Tuple[str, ...]) -> float:
        """
        Calculates the conditional probability for the provided term and the term's context.

        P(am|I) = cond_prob(term = "am", given = ("I",))
        """
        if given in self.pdist:
            return self.pdist[given][self.vocab.id_for(term)]
        else:
            return 0

    def prob_of(self, tokens: Iterable[str]) -> float:
        """
        Calculates the probability of a token sequence using an _n_ - 1 order Markov assumption.
        """
        # the starting probability of the sequence
        p = 1
        for gram in self.ngrams_for(
            n=self.n, tokens=tokens, use_start_end=self.use_start_end
        ):
            next_tok = gram[-1]
            history = gram[:-1]

            gram_prob = self.cond_prob(next_tok, history)
            p *= gram_prob

        return p


# TODO should create make_model() and use __main__ for various querying tasks


# # tokenize content, clean it up, and use it to train a bigram language model

if __name__ == "__main__":
    # text is from circa 1910 from the Tom Swift series of books.
    requests_cache.install_cache("tm_949", backend="filesystem")
    res = get("https://gutenberg.org/ebooks/949.txt.utf-8", timeout=30)

    content = res.text
    lines = content.splitlines()
    # TODO is text merely split at a linebreak still a "sentence" per se??

    # = Hello! https://www.reddit.com/r/LanguageTechnology/comments/bnb0p1/how_to_clean_the_gutenbergs_dataset/

    # Chop off header and footer
    endpoints = {"start": 0, "end": len(lines)}

    for k, line in enumerate(lines):
        if line.startswith("*** START OF THE PROJECT"):
            endpoints["start"] = k

        if line.startswith("*** END OF THE PROJECT GUTENBERG"):
            endpoints["end"] = k

    better_text = lines[endpoints["start"] + 1 : endpoints["end"] - 1]
    split_regex = re.compile(
        r"""
                        \s # first split on whitespace, then discard it.
                        | # or
                        ('\w+) # then split on apostrophes, using a capturing group to retain it.
                        | # or
                        ((?<![A-Z][a-z])[,\?\.!](?!\w)) # separate out but keep punctuation, but only if it's not followed by a word char. AND don't split if pattern = "Xx.".
    """,
        flags=re.VERBOSE,
    )

    filtered_corpus = [
        list(filter(lambda x: x is not None and len(x) > 0, re.split(split_regex, l)))
        for l in better_text
        if len(l.strip()) > 0
    ]

    # print(corpus[40:60], len(corpus))

    lm = LanguageModel(corpus=filtered_corpus, n=2)
    # print(lm)
    print(lm.prob_of(["from", "Shopton"]))
    print(lm.prob_of(["Mr.", "Sharp"]))
