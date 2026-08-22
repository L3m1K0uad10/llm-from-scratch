

# implementing Byte-Pair Encoding (BPE) tokenizer
class Tokenizer:
    def __init__(self):
        pass 
    
    @staticmethod
    def split(corpus:list) -> list:
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
    def get_pair_frequency(corpus:list) -> dict:
        """  
        Tokenize the corpus into words, add </w> to the end of every word,
        and count their frequencies across the corpus.
        """
        
        pair_frequency_dictionary = {}

        for sentence in corpus:
            # splitting the sentence into a list of words once.
            # Doing sentence.split() once is much faster than calling it repeatedly in a loop.
            words = sentence.split()
            
            # iterating through each word in the sentence exactly once.
            for index, word in enumerate(words):
                
                # appending the "</w>" token to the end of each character list.
                pair_tuple = tuple(list(word) + ["</w>"])
                
                # updating the dictionary.
                # Since we are visiting every word occurrence one by one,
                # we just add 1 to its total count in our dictionary.
                if pair_tuple in pair_frequency_dictionary: 
                    pair_frequency_dictionary[pair_tuple] += 1
                else:
                    pair_frequency_dictionary[pair_tuple] = 1

        return pair_frequency_dictionary

    @staticmethod
    def get_pair_stats(pair_frequency_dict:dict) -> dict:
        """  
        get sub pre-token pair statistics(no of occurence in the whole dictionary)
        """

        pair_stats = {}

        for pre_token in pair_frequency_dict:
            freq = pair_frequency_dict[pre_token] # retrieves the frequency of the pre-token
            n = len(pre_token)

            for i in range(1, n):
                pair = (pre_token[i - 1], pre_token[i])
                
                if pair in pair_stats: 
                    pair_stats[pair] += freq
                else:
                    pair_stats[pair] = freq

        return pair_stats       

    @staticmethod
    def merge_pair(pair_frequency:dict, pair_stats:dict) -> dict:
        """  
        merge most frequent pair from pair_stats to pair_frequency.
        """

        # finding the most frequent pair present in pair_stats
        higher_frequency_value = max(pair_stats.values())
        frequent_pair = None

        for key_pair in pair_stats:
            if pair_stats[key_pair] == higher_frequency_value:
                frequent_pair = key_pair 
                break 
        
        # merging the most frequent pair accordingly to pair_frequency
        new_pair_frequency = {}
        
        for key_pair, freq in pair_frequency.items():
            new_key = []
            i = 0
            n = len(key_pair)

            while i < n:
                # if we find a matching pair, merge them and jump forward by 2
                if i < n - 1 and (key_pair[i], key_pair[i+1]) == frequent_pair:
                    new_key.append(key_pair[i] + key_pair[i+1])
                    i += 2
                else:
                    new_key.append(key_pair[i])
                    i += 1
            
            new_key_tuple = tuple(new_key)
            # Accumulate frequencies in case different keys merge to the same target tuple
            new_pair_frequency[new_key_tuple] = new_pair_frequency.get(new_key_tuple, 0) + freq

        return new_pair_frequency

    @staticmethod
    def prettier(dictionary:dict) -> None:
        """  
        print dictionary in a row wise pretty format
        """

        for pair in dictionary.items():
            print(pair)



corpus = [ 
    "The cat sat on the mat.",
    "The dog sat on the rug.",
    "Cats and dogs like the mat!",
    "Does a cat like a rug?"
]

tokenizer = Tokenizer()
splitted_characters = tokenizer.split(corpus)
print(splitted_characters)

print("\n")

pair_frequency = tokenizer.get_pair_frequency(corpus)
print(pair_frequency)

print("\n")
tokenizer.prettier(pair_frequency)

print("\n")

pair_stats = tokenizer.get_pair_stats(pair_frequency)
print(pair_stats)

print("\n")
tokenizer.prettier(pair_stats)




print("\n")
merged_pair = tokenizer.merge_pair(pair_frequency, pair_stats)
print(merged_pair)

pair_stats = tokenizer.get_pair_stats(merged_pair)

print("\n")
merged_pair = tokenizer.merge_pair(merged_pair, pair_stats)
print(merged_pair)

pair_stats = tokenizer.get_pair_stats(merged_pair)

print("\n")
merged_pair = tokenizer.merge_pair(merged_pair, pair_stats)
print(merged_pair)

pair_stats = tokenizer.get_pair_stats(merged_pair)

print("\n")
merged_pair = tokenizer.merge_pair(merged_pair, pair_stats)
print(merged_pair)

pair_stats = tokenizer.get_pair_stats(merged_pair)

print("\n")
merged_pair = tokenizer.merge_pair(merged_pair, pair_stats)
print(merged_pair)