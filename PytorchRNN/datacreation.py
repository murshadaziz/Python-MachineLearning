import os
import torch
from string import ascii_letters
from sklearn.model_selection import train_test_split
import unidecode    

# Data directory
data_dir = "./data/names"
# Create a mapping from language to label
lang2label = {
    # Splits at '.' with [0] being the language name and [1] being the file extension
    # so uses name of the file as the key and the label as the value starting from 0
    file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
    for i, file_name in enumerate(os.listdir(data_dir))
}
# Total languages
num_langs = len(lang2label)
# Create a mapping from character to index uses the 52 ascii characters (uppercase and lowercase) plus space, period, comma, colon, semicolon, hyphen, and apostrophe
char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
num_letters = len(char2idx)

# Function to convert a name to a tensor
def name2tensor(name) -> torch.Tensor: 
    # tensor of shape (name_length, 1, num_letters) means an array for every letter in the name 
    # will have a one-hot encoded vector of length num_letters (1 row, num_letters (59) columns)
    tensor = torch.zeros(len(name), 1, num_letters)
    for i, char in enumerate(name):
        tensor[i][0][char2idx[char]] = 1
    return tensor

# Function to be used in main file to return the train and test datasets, number of languages, and number of letters
def load_data() -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], int, int]:
    # lists to hold the tensors of names and their corresponding target language's labels
    tensor_names = []
    target_langs = []

    # Loops over every file in the data/names directory
    for file in os.listdir(data_dir):
        # Joins directory and file name to create a full path to the file and opens it
        with open(os.path.join(data_dir, file)) as f:
            # Splits the file name at the '.' and takes the first part as the language name
            lang = file.split(".")[0]
            # Reads name in each line in the file, strips whitespace from the right side, and converts it to ASCII using unidecode
            # line.rstrip() removes white characters from the right side of the string
            names = [unidecode(line.rstrip()) for line in f]
            # Iterates over each name in the list of names
            for name in names:
                try: 
                    # adds tensor of name and label of lang to corresponding lists if the exist in the dicts
                    tensor_names.append(name2tensor(name))
                    target_langs.append(lang2label[lang])
                except KeyError:
                    pass

    # Splits the data into train and test set using scikit learn
    train_idx, test_idx = train_test_split(
        # total size 
        range(len(target_langs)), 
        # ratio of train to test set 9:1
        test_size=0.1, 
        shuffle=True, 
        stratify=target_langs
    )
    # Creates train set
    train_dataset = [
        (tensor_names[i], target_langs[i])
        for i in train_idx
    ]

    # Created test set
    test_dataset = [
        (tensor_names[i], target_langs[i])
        for i in test_idx
    ]

    return train_dataset, test_dataset, num_langs, num_letters