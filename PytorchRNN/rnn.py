import os
import random
from string import ascii_letters

import torch
from torch import nn
import torch.nn.functional as F
from unidecode import unidecode

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_dir = "./data/names"
    lang2label = {
        file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
        for i, file_name in enumerate(os.listdir(data_dir))
    }
    print(f"Language to label mapping: {lang2label}")

main()