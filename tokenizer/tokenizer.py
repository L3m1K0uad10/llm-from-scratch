import json
from typing import Dict, List, Tuple


class Tokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer for LLM training and inference.
    """

    def __init__(self):
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        
        # maps (token_a, token_b) -> merge rank (0 = learned first / highest priority)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.vocab_size: int = 0

    def _rebuild_lookups(self) -> None:
        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    @staticmethod
    def split(corpus: List[str]) -> List[str]:
        """
        Extract all unique characters from the corpus,
        add the </w> end-of-word token, and return them sorted.
        """
        unique_characters = []
        for sentence in corpus:
            for character in sentence:
                if character not in unique_characters:
                    unique_characters.append(character)
        if "</w>" not in unique_characters:
            unique_characters.append("</w>")
        unique_characters.sort()
        return unique_characters

    @staticmethod
    def get_pair_frequency(corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Tokenize the corpus into words, add </w> to the end of every word,
        and count their frequencies across the corpus.
        """
        pair_frequency_dictionary: Dict[Tuple[str, ...], int] = {}
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                pair_tuple = tuple(list(word) + ["</w>"])
                pair_frequency_dictionary[pair_tuple] = (
                    pair_frequency_dictionary.get(pair_tuple, 0) + 1
                )
        return pair_frequency_dictionary

    @staticmethod
    def get_pair_stats(
        pair_frequency_dict: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, str], int]:
        """
        Get adjacent symbol pair counts weighted by pre-token frequency.
        """
        pair_stats: Dict[Tuple[str, str], int] = {}
        for pre_token, freq in pair_frequency_dict.items():
            n = len(pre_token)
            for i in range(1, n):
                pair = (pre_token[i - 1], pre_token[i])
                pair_stats[pair] = pair_stats.get(pair, 0) + freq
        return pair_stats

    @staticmethod
    def merge_pair(
        pair_frequency: Dict[Tuple[str, ...], int],
        frequent_pair: Tuple[str, str],
    ) -> Dict[Tuple[str, ...], int]:
        """
        Merge the specified frequent pair across all pre-tokens in pair_frequency.
        """
        new_pair_frequency: Dict[Tuple[str, ...], int] = {}
        for key_pair, freq in pair_frequency.items():
            new_key: List[str] = []
            i = 0
            n = len(key_pair)
            while i < n:
                if i < n - 1 and (key_pair[i], key_pair[i + 1]) == frequent_pair:
                    new_key.append(key_pair[i] + key_pair[i + 1])
                    i += 2
                else:
                    new_key.append(key_pair[i])
                    i += 1
            new_key_tuple = tuple(new_key)
            new_pair_frequency[new_key_tuple] = (
                new_pair_frequency.get(new_key_tuple, 0) + freq
            )
        return new_pair_frequency

    def train(self, corpus: List[str], target_vocab_size: int) -> None:
        """
        Trains the BPE tokenizer on a given text corpus up to target_vocab_size.
        """
        chars = self.split(corpus)
        vocab = {idx: char for idx, char in enumerate(chars)}
        merge_ranks: Dict[Tuple[str, str], int] = {}

        pair_freqs = self.get_pair_frequency(corpus)
        num_merges = max(0, target_vocab_size - len(vocab))

        for rank in range(num_merges):
            stats = self.get_pair_stats(pair_freqs)
            if not stats:
                break

            frequent_pair = max(stats, key=stats.get)
            pair_freqs = self.merge_pair(pair_freqs, frequent_pair)

            merged_token = frequent_pair[0] + frequent_pair[1]
            merge_ranks[frequent_pair] = rank
            vocab[len(vocab)] = merged_token

        self.vocab = vocab
        self.merge_ranks = merge_ranks
        self._rebuild_lookups()

    def _encode_word(self, word: str) -> List[str]:
        """
        Applies learned BPE merges to a single whitespace-delimited word.
        """
        tokens = list(word) + ["</w>"]
        if len(tokens) == 1:
            return tokens

        while len(tokens) > 1:
            # Find the adjacent pair in `tokens` with the lowest rank (highest priority)
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of `best_pair` in `tokens`
            merged_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encodes a text string into a list of vocabulary token IDs.
        """
        if not self.token_to_id:
            self._rebuild_lookups()

        token_ids: List[int] = []
        for word in text.split():
            for piece in self._encode_word(word):
                if piece in self.token_to_id:
                    token_ids.append(self.token_to_id[piece])
                else:
                    # Fallback to character-level pieces if a merged token is absent
                    for ch in piece:
                        if ch in self.token_to_id:
                            token_ids.append(self.token_to_id[ch])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of vocabulary token IDs back into a string.
        """
        pieces = [self.vocab[idx] for idx in token_ids if idx in self.vocab]
        joined = "".join(pieces)
        # Convert end-of-word markers back into spaces
        text = joined.replace("</w>", " ").strip()
        return text

    def save(self, filepath: str) -> None:
        """Saves tokenizer vocabulary and ranked merges to disk."""
        ordered_merges = [
            list(pair)
            for pair, _ in sorted(self.merge_ranks.items(), key=lambda item: item[1])
        ]
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": ordered_merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """Loads tokenizer vocabulary and ranked merges from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        raw_merges = data.get("merges", [])
        self.merge_ranks = {
            (pair[0], pair[1]): idx for idx, pair in enumerate(raw_merges)
        }
        self._rebuild_lookups()


if __name__ == "__main__":
    corpus = [
        "The cat sat on the mat.",
        "The dog sat on the rug.",
        "Cats and dogs like the mat!",
        "Does a cat like a rug?",
    ]

    tokenizer = Tokenizer()
    tokenizer.train(corpus, target_vocab_size=45)

    sample = "The cat sat on the rug."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Original:", sample)
    print("Encoded: ", encoded)
    print("Decoded: ", decoded)