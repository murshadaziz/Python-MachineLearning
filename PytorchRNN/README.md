# PytorchRNN

Classify the language of origin of a name from its characters, using three recurrent architectures in PyTorch:

| File | Model | Notes |
|---|---|---|
| `simplernn.py` | Hand-written RNN cell (`SimpleRNN`) | Loops over characters manually, sigmoid hidden update, gradient clipping (max norm 1), 1 epoch |
| `grurnn.py` | `nn.GRU` (2 layers, hidden size 256) | Built-in GRU, 2 epochs |
| `lstmrnn.py` | `nn.LSTM` (2 layers, hidden size 256) | Built-in LSTM with hidden and cell state, averaged loss logging, 2 epochs |

All three share the same data pipeline (`datacreation.py`), optimizer (Adam, lr `0.001`), and loss (`CrossEntropyLoss`), so their results are directly comparable.

## How it works

1. **Data loading** (`datacreation.py`): each file in `data/names/` is one language, and its filename (minus the extension) becomes the class label. Every line in a file is a name.
2. **Preprocessing**: names are converted to ASCII with `unidecode`, then one-hot encoded character by character into a tensor of shape `(name_length, 1, 59)`. The vocabulary is the 52 ASCII letters plus space and `. , : ; - '`. Names containing any other character are skipped.
3. **Split**: 90% train / 10% test, stratified by language.
4. **Training**: one name per step (batch size 1), shuffled every epoch. The loss is printed every 100 steps.
5. **Prediction**: the model reads the whole name, and the output at the **final character** is passed through a linear layer to produce a score for each language. The highest score is the predicted language.

## Setup

Requires Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

### Data

The `data/` folder is git-ignored, so you need to add it yourself. Expected layout:

```
PytorchRNN/
├── data/
│   └── names/
│       ├── Arabic.txt
│       ├── English.txt
│       ├── Russian.txt
│       └── ...        # one .txt file per language, one name per line
├── datacreation.py
├── simplernn.py
├── grurnn.py
└── lstmrnn.py
```

This is the format of the names dataset used in the PyTorch character-level RNN classification tutorial.

## Usage

Run from inside this folder, because the data path (`./data/names`) is relative:

```bash
cd PytorchRNN
python simplernn.py
python grurnn.py
python lstmrnn.py
```

Each script trains the model, prints test accuracy, and then prints predictions for a few example names. To try your own names, edit the example names at the bottom of `main()`, or import a script's predict function:

```python
from lstmrnn import lstm_predict
```

The predict functions are `simplernn_predict`, `grurnn_predict`, and `lstm_predict`. Each takes `(model, name)` and returns the language name.

## Notes and limitations

- **CPU only in practice.** The scripts define a `device`, but the dataset tensors from `datacreation.py` are created on the CPU. Running on a GPU would require moving the data (and, for `simplernn.py` and `grurnn.py`, the model) to the same device.
- **Batch size 1.** Simple and easy to follow, but slow. Batching would need padding or packed sequences.
- **Class imbalance.** Language files differ a lot in size, so accuracy can be dominated by the largest languages. Per-class metrics would give a fuller picture.
- **Importing runs the data load.** Each model script loads the dataset at import time, so importing one triggers a full data load.