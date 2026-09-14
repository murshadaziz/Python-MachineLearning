import random


import torch
from torch import nn
import torch.nn.functional as F
from unidecode import unidecode
import datacreation 

_ = torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, test_dataset, num_langs, num_letters = datacreation.load_data()

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)
    
    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

def train(model, train_dataset, criterion, optimizer, num_epochs, print_interval=100):
    for epoch in range(num_epochs):
        random.shuffle(train_dataset)

        for i, (name, label) in enumerate(train_dataset):
            hidden_state = model.init_hidden()

            for char in name:
                output, hidden_state = model(char, hidden_state)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_dataset)}], "
                    f"Loss: {loss.item():.4f}"
                )
def test(model, test_dataset):
    num_correct = 0
    num_samples = len(test_dataset)

    model.eval()

    with torch.no_grad():
        for name, label in test_dataset:
            hidden_state = model.init_hidden()
            for char in name:
                output, hidden_state = model(char, hidden_state)
            _, pred = torch.max(output, dim=1)
            num_correct += bool(pred == label)

    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")
def main():
    hidden_size = 256
    learning_rate = 0.001

    model = MyRNN(num_letters, hidden_size, num_langs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(
    model,
    train_dataset,
    criterion,
    optimizer,
    num_epochs=3
    )
    test(model, test_dataset)

if __name__ == "__main__":
    main()