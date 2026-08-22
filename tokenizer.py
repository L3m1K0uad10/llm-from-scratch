

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
            # Split the sentence into a list of words once.
            # Doing sentence.split() once is much faster than calling it repeatedly in a loop.
            words = sentence.split()
            
            # Iterate through each word in the sentence exactly once.
            for index, word in enumerate(words):
                
                # Append the "</w>" token to the end of each character list.
                pair_tuple = tuple(list(word) + ["</w>"])
                
                # Update the dictionary.
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
            freq = pair_frequency_dict[pre_token] # retrieve the frequency of the pre-token
            n = len(pre_token)

            for i in range(1, n):
                pair = (pre_token[i - 1], pre_token[i])
                
                if pair in pair_stats: 
                    pair_stats[pair] += freq
                else:
                    pair_stats[pair] = freq

        return pair_stats            

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