


class Tokenizer:
    def __init__(self):
        pass 

    def split(corpus:list) -> list:
        """ 
        split whole corpus into unique character.
        add </w> at the end of the unique characters list.
        sort the unique characters list
        """

        unique_characters = []

        for sentence in corpus:
            for character in list(sentence):
                if character not in unique_characters:
                    unique_characters.append(character) 
        
        unique_characters.append("</w>")

        unique_characters.sort()
    
        return unique_characters
    
    
    def get_pair_frequency(corpus:list) -> dict:
        """  
        split whole corpus into word composed character.
        get and set the frequency of each word composed character.
        """
        
        pair_frequency_dictionary = {}

        for sentence in corpus:
            
            # comparing a word to other word in the same sentence to look if they match or not
            for index, word in enumerate(sentence.split()):
                count = 0
                prev_word = word

                for word in sentence.split():
                    if prev_word == word: 
                        count += 1

                if index == len(sentence.split()) - 1:
                    pair_tuple = tuple(list(prev_word) + ["</w>"])
                else:
                    pair_tuple = tuple(list(prev_word))
                
                if pair_tuple in pair_frequency_dictionary.keys(): # checks if word frequency already recorded in pair_frequency_dictionary
                    count += pair_frequency_dictionary[pair_tuple]

                pair_frequency_dictionary[pair_tuple] = count

        return pair_frequency_dictionary
    

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
splitted_characters = Tokenizer.split(corpus)
print(splitted_characters)

print("\n")

pair_frequency = Tokenizer.get_pair_frequency(corpus)
print(pair_frequency)

print("\n")
Tokenizer.prettier(pair_frequency)