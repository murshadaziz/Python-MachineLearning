import random

import torch
from torch import nn
import torch.nn.functional as F
import datacreation 

# Set random seed for reproducibility
_ = torch.manual_seed(42)
# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and testing datasets, number of languages, and number of letters from the datacreation module
train_dataset, test_dataset, num_langs, num_letters = datacreation.load_data()

class GRUModel(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=num_letters, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, num_langs)
    
    def forward(self, x):
        hidden_state = self.init_hidden()
        output, hidden_state = self.gru(x, hidden_state)
        output = self.fc(output[-1])
        return output
    
    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).to(device)

def train(model, train_dataset, criterion, optimizer, num_epochs, print_interval=100):
    for epoch in range(num_epochs):
        random.shuffle(train_dataset)
        for i, (name, label) in enumerate(train_dataset):
            output = model(name)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
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
            output = model(name)
            _, pred = torch.max(output, dim=1)
            num_correct += bool(pred == label)

    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")

def grurnn_predict(model, name):
    model.eval()
    tensor_name = datacreation.name2tensor(name)
    with torch.no_grad():
        output = model(tensor_name)
        _, pred = torch.max(output, dim=1)
    model.train()
    return datacreation.label2lang[pred.item()]

def main():
    hidden_size = 256
    learning_rate = 0.001

    model = GRUModel(num_layers=2, hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train(
    model,
    train_dataset,
    criterion,
    optimizer,
    num_epochs=2
    )
    test(model, test_dataset)

    # Example prediction
    name = "Ali"
    predicted_language = grurnn_predict(model, name)
    print(f"The predicted language for the name '{name}' is: {predicted_language}")
    name = "Ahmed"
    predicted_language = grurnn_predict(model, name)
    print(f"The predicted language for the name '{name}' is: {predicted_language}")

if __name__ == "__main__":
    main()