

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
        pair_frequency_copy = pair_frequency.copy()
        
        for key_pair in pair_frequency:
            n = len(key_pair)

            for i in range(1, n):
                pair = (key_pair[i - 1], key_pair[i])
                
                if pair == frequent_pair: 
                    temp = list(key_pair)
                    value_joined = temp.pop(i - 1) + temp.pop(i - 1) # should have been temp.pop(i - 1) and temp.pop(i) but after the first pop temp got 1 value shorter
                    temp.insert(i - 1, value_joined)

                    freq = pair_frequency_copy.pop(key_pair)
                    if i == len(key_pair) - 1:
                        pair_frequency_copy[tuple(temp + ["</w>"])] = freq
                    else:
                        pair_frequency_copy[tuple(temp)] = freq
        
        return pair_frequency_copy

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