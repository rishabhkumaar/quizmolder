"""
utils/quizgen.py

Features:
- QuizGenerator class producing MCQ / True-False / Short-answer questions
- Pluggable distractor strategies:
    * RandomDistractor (fast, default)
    * WordNetDistractor (plausible lexical distractors)
    * EmbeddingDistractor (semantic distractors using sentence-transformers) - optional
- Lightweight caching, logging, and clean tokenization
- Returns list of QuizQuestion dicts: {"question","answer","options","meta"}
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
import random
import re
import logging
import functools

# Ensure resources
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)  # multilingual WordNet mappings (safe)

# Optional: semantic distractors using sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

# Configure logger
logger = logging.getLogger("quizgen")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --------- Data model ----------
@dataclass
class QuizQuestion:
    question: str
    answer: str
    options: List[str]
    meta: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


# --------- Utility functions ----------
WORD_RE = re.compile(r"[A-Za-z]+")


def clean_word(w: str) -> str:
    return w.strip()


def is_content_word(w: str, min_len: int = 4) -> bool:
    return bool(WORD_RE.fullmatch(w)) and len(w) >= min_len


# --------- Distractor Strategy Interfaces ----------
class BaseDistractor:
    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary

    def get(self, answer: str, n: int) -> List[str]:
        raise NotImplementedError()


class RandomDistractor(BaseDistractor):
    def get(self, answer: str, n: int) -> List[str]:
        candidates = [w for w in self.vocab if w.lower() != answer.lower()]
        if not candidates:
            return []
        k = min(n, len(candidates))
        return random.sample(candidates, k)


class WordNetDistractor(BaseDistractor):
    def get_synonym_lemmas(self, term: str) -> List[str]:
        syns = wordnet.synsets(term)
        lemmas = set()
        for s in syns:
            for l in s.lemmas():
                lemmas.add(l.name().replace("_", " "))
        return list(lemmas)

    def get(self, answer: str, n: int) -> List[str]:
        # Try synonyms for the same lemma
        syns = self.get_synonym_lemmas(answer)
        syns = [s for s in syns if s.lower() != answer.lower()]
        # If enough synonyms, pick from them
        chosen = []
        if syns:
            chosen = random.sample(syns, min(n, len(syns)))
        # Fill remaining from vocab (similar length)
        if len(chosen) < n:
            remaining = [w for w in self.vocab if w.lower() != answer.lower() and w not in chosen]
            # prefer similar-length words
            remaining.sort(key=lambda x: abs(len(x) - len(answer)))
            needed = n - len(chosen)
            chosen += remaining[:needed]
        return chosen


class EmbeddingDistractor(BaseDistractor):
    """
    Requires sentence-transformers. Precomputes embeddings of vocabulary and returns nearest neighbors.
    Falls back gracefully if package isn't installed.
    """

    def __init__(self, vocabulary: List[str], model_name: str = "all-MiniLM-L6-v2"):
        if not EMBEDDING_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")
        super().__init__(vocabulary)
        self.model = SentenceTransformer(model_name)
        self.vocab = list(set(vocabulary))
        self.embeddings = None
        self._build_embeddings()

    def _build_embeddings(self):
        if not self.vocab:
            self.embeddings = None
            return
        self.embeddings = self.model.encode(self.vocab, convert_to_numpy=True, show_progress_bar=False)

    def get(self, answer: str, n: int) -> List[str]:
        if not self.embeddings.any():
            return []
        ans_emb = self.model.encode([answer], convert_to_numpy=True)[0]
        # cosine similarity
        sims = np.dot(self.embeddings, ans_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(ans_emb) + 1e-9)
        # sort desc
        idx_sorted = sims.argsort()[::-1]
        candidates = []
        for idx in idx_sorted:
            candidate = self.vocab[int(idx)]
            if candidate.lower() == answer.lower():
                continue
            candidates.append(candidate)
            if len(candidates) >= n:
                break
        return candidates[:n]


# --------- Main generator ----------
class QuizGenerator:
    """
    Usage:
      qg = QuizGenerator(text)
      qg.generate_mcq(5, strategy="wordnet")
      qg.generate_true_false(5)
    """

    def __init__(self, text: str, min_word_len: int = 4):
        self.raw = (text or "").strip()
        self.min_word_len = min_word_len
        self.sentences = sent_tokenize(self.raw) if self.raw else []
        self.vocab = self._build_vocab()
        self._embedding_strategy_cache = {}

    def _build_vocab(self) -> List[str]:
        words = [clean_word(w) for w in word_tokenize(self.raw) if is_content_word(w, self.min_word_len)]
        # de-duplicate while preserving some randomness
        return list(dict.fromkeys([w for w in words]))

    def _choose_distractor_strategy(self, name: str):
        name = (name or "random").lower()
        if name == "random":
            return RandomDistractor(self.vocab)
        if name == "wordnet":
            return WordNetDistractor(self.vocab)
        if name == "embedding":
            # caching multiple instances to avoid re-init cost
            if "embedding" not in self._embedding_strategy_cache:
                try:
                    self._embedding_strategy_cache["embedding"] = EmbeddingDistractor(self.vocab)
                except Exception as e:
                    logger.warning("Embedding strategy unavailable: %s", e)
                    return RandomDistractor(self.vocab)
            return self._embedding_strategy_cache["embedding"]
        # default
        return RandomDistractor(self.vocab)

    @functools.lru_cache(maxsize=128)
    def _pick_sentences_for_mcq(self, num_questions: int) -> List[str]:
        if not self.sentences:
            return []
        # prefer longer sentences and variety
        sorted_sents = sorted(self.sentences, key=lambda s: max(len(s), 0), reverse=True)
        k = min(num_questions, len(sorted_sents))
        # sample from top half to keep quality but add randomness
        top_half = sorted_sents[: max(1, len(sorted_sents)//2)]
        chosen = random.sample(top_half, k) if len(top_half) >= k else random.sample(sorted_sents, k)
        return chosen

    def _select_answer_from_sentence(self, sent: str) -> Optional[str]:
        words = [w for w in word_tokenize(sent) if is_content_word(w, self.min_word_len)]
        if not words:
            return None
        return random.choice(words)

    def generate_mcq(
        self,
        num_questions: int = 5,
        num_options: int = 4,
        distractor_strategy: str = "random"
    ) -> List[dict]:
        """
        Returns list of dicts: {question, answer, options, meta}
        """
        questions: List[QuizQuestion] = []
        if not self.sentences:
            return []

        strategy = self._choose_distractor_strategy(distractor_strategy)

        candidate_sentences = self._pick_sentences_for_mcq(num_questions)
        for sent in candidate_sentences:
            answer = self._select_answer_from_sentence(sent)
            if not answer:
                continue
            # blank only the chosen token (first occurrence, exact match ignoring case where possible)
            pattern = re.compile(re.escape(answer), re.IGNORECASE)
            question_text = pattern.sub("_____", sent, count=1)

            # generate distractors
            distractors = strategy.get(answer, max(0, num_options - 1))
            # ensure uniqueness
            opts_set = []
            for d in distractors:
                if d.lower() != answer.lower() and d not in opts_set:
                    opts_set.append(d)
            # if not enough distractors, pad from vocab
            if len(opts_set) < (num_options - 1):
                extra = [w for w in self.vocab if w.lower() != answer.lower() and w not in opts_set]
                extra = extra[: max(0, (num_options - 1) - len(opts_set))]
                opts_set += extra

            options = opts_set + [answer]
            # final shuffle with stable reproducibility if needed
            random.shuffle(options)

            q = QuizQuestion(
                question=question_text,
                answer=answer,
                options=options,
                meta={"source_sentence": sent}
            )
            questions.append(q)

        return [q.to_dict() for q in questions]

    def generate_true_false(self, num_questions: int = 5, perturb_probability: float = 0.5) -> List[dict]:
        """
        Create True/False statements. We take sentences and perturb a content word
        to create plausible false statements.
        """
        questions = []
        if not self.sentences:
            return []

        selected = random.sample(self.sentences, min(num_questions, len(self.sentences)))
        for sent in selected:
            is_true = random.random() > perturb_probability
            statement = sent
            if not is_true:
                # attempt a small perturbation: replace a content word with another vocab word
                answer_word = self._select_answer_from_sentence(sent)
                if answer_word and self.vocab:
                    replacement = random.choice([w for w in self.vocab if w.lower() != answer_word.lower()])
                    statement = re.sub(re.escape(answer_word), replacement, sent, count=1, flags=re.IGNORECASE)
            q = QuizQuestion(
                question=f"True or False: {statement}",
                answer="True" if is_true else "False",
                options=["True", "False"],
                meta={"source_sentence": sent}
            )
            questions.append(q)
        return [q.to_dict() for q in questions]

    def generate_short_answer(self, num_questions: int = 5) -> List[dict]:
        """
        Extract factual short-answer prompts by blanking a key token and returning answer.
        (A lighter-weight cloze)
        """
        questions = []
        if not self.sentences:
            return []

        selected = self._pick_sentences_for_mcq(num_questions)
        for sent in selected:
            answer = self._select_answer_from_sentence(sent)
            if not answer:
                continue
            question_text = re.sub(re.escape(answer), "_____", sent, count=1, flags=re.IGNORECASE)
            q = QuizQuestion(
                question=question_text,
                answer=answer,
                options=[],
                meta={"type": "short_answer", "source_sentence": sent}
            )
            questions.append(q)
        return [q.to_dict() for q in questions]
