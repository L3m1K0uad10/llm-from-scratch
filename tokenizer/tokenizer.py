import json
from typing import List, Dict, Tuple



class Tokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer for LLM training and inference.
    """
    def __init__(self):
        self.vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab_size: int = 0
        self.special_tokens: Dict[str, int] = {}

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
        unique_characters.append("</w>")
        unique_characters.sort()
        return unique_characters
    
    @staticmethod
    def get_pair_frequency(corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """  
        Tokenize the corpus into words, add </w> to the end of every word,
        and count their frequencies across the corpus.
        """
        pair_frequency_dictionary = {}
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                pair_tuple = tuple(list(word) + ["</w>"])
                if pair_tuple in pair_frequency_dictionary: 
                    pair_frequency_dictionary[pair_tuple] += 1
                else:
                    pair_frequency_dictionary[pair_tuple] = 1
        return pair_frequency_dictionary

    @staticmethod
    def get_pair_stats(pair_frequency_dict: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """  
        Get sub pre-token pair statistics (number of occurrences in the whole dictionary).
        """
        pair_stats = {}
        for pre_token, freq in pair_frequency_dict.items():
            n = len(pre_token)
            for i in range(1, n):
                pair = (pre_token[i - 1], pre_token[i])
                pair_stats[pair] = pair_stats.get(pair, 0) + freq
        return pair_stats       

    @staticmethod
    def merge_pair(pair_frequency: Dict[Tuple[str, ...], int], frequent_pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """  
        Merge the specified frequent pair across all pre-tokens in pair_frequency.
        """
        new_pair_frequency = {}
        for key_pair, freq in pair_frequency.items():
            new_key = []
            i = 0
            n = len(key_pair)
            while i < n:
                if i < n - 1 and (key_pair[i], key_pair[i+1]) == frequent_pair:
                    new_key.append(key_pair[i] + key_pair[i+1])
                    i += 2
                else:
                    new_key.append(key_pair[i])
                    i += 1
            new_key_tuple = tuple(new_key)
            new_pair_frequency[new_key_tuple] = new_pair_frequency.get(new_key_tuple, 0) + freq
        return new_pair_frequency

    def train(self, corpus: List[str], target_vocab_size: int):
        """
        Trains the BPE tokenizer on a given text corpus up to a target vocabulary size.
        """
        # initializing base vocabulary from unique characters
        chars = self.split(corpus)
        vocab = {idx: char for idx, char in enumerate(chars)}
        
        pair_freqs = self.get_pair_frequency(corpus)
        num_merges = target_vocab_size - len(vocab)
        
        print(f"Training BPE tokenizer. Base vocab size: {len(vocab)}. Performing {num_merges} merges...")

        for i in range(max(0, num_merges)):
            stats = self.get_pair_stats(pair_freqs)
            if not stats:
                break
            
            # finding most frequent pair
            frequent_pair = max(stats, key=stats.get)
            
            # performing merge
            pair_freqs = self.merge_pair(pair_freqs, frequent_pair)
            
            # recording merge rule and new token
            merged_token = frequent_pair[0] + frequent_pair[1]
            self.merges[frequent_pair] = merged_token
            
            new_id = len(vocab)
            vocab[new_id] = merged_token

        self.vocab = vocab
        self.vocab_size = len(vocab)
        print(f"Training complete! Final vocabulary size: {self.vocab_size}")

    def save(self, filepath: str):
        """Saves tokenizer vocabulary and merges to disk."""
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        with open(filepath, "w", encoding = "utf-8") as f:
            json.dump(data, f, ensure_ascii = False, indent = 2)

    def load(self, filepath: str):
        """Loads tokenizer vocabulary and merges from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.merges = {tuple(k.split(",")): v for k, v in data["merges"].items()}
        self.vocab_size = len(self.vocab)


if __name__ == "__main__":
    corpus = [ 
        "The cat sat on the mat.",
        "The dog sat on the rug.",
        "Cats and dogs like the mat!",
        "Does a cat like a rug?"
    ]

    tokenizer = Tokenizer()
    tokenizer.train(corpus, target_vocab_size=35)
    print("Sample Vocab entries:", list(tokenizer.vocab.items())[:])