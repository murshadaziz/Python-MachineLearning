import random

import torch
from torch import nn
import datacreation

# Set random seed for reproducibility
_ = torch.manual_seed(42)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and testing datasets, number of languages, and number of letters
train_dataset, test_dataset, num_langs, num_letters = datacreation.load_data()


class LSTMModel(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(LSTMModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_letters,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(hidden_size, num_langs)

    def forward(self, x):
        hidden_state, cell_state = self.init_hidden()

        output, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))

        # Take the output from the final character (top layer)
        output = self.fc(output[-1])

        return output

    def init_hidden(self):
        hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size
        ).to(device)

        cell_state = torch.zeros(
            self.num_layers, 1, self.hidden_size
        ).to(device)

        return hidden_state, cell_state


def train(model, train_dataset, criterion, optimizer, num_epochs, print_interval=100):
    model.train()

    for epoch in range(num_epochs):
        random.shuffle(train_dataset)
        running_loss = 0.0

        for i, (name, label) in enumerate(train_dataset):
            output = model(name)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_dataset)}], "
                    f"Avg Loss: {running_loss / print_interval:.4f}"
                )
                running_loss = 0.0


def test(model, test_dataset):
    num_correct = 0
    num_samples = len(test_dataset)

    model.eval()

    with torch.no_grad():
        for name, label in test_dataset:
            output = model(name)

            _, pred = torch.max(output, dim=1)

            num_correct += bool(pred == label)

    model.train()

    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")


def lstm_predict(model, name):
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

    model = LSTMModel(num_layers=2, hidden_size=hidden_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate
    )

    train(
        model,
        train_dataset,
        criterion,
        optimizer,
        num_epochs=2
    )

    test(model, test_dataset)

    # Example predictions
    for name in ["Qin", "Slaveya"]:
        predicted_language = lstm_predict(model, name)
        print(
            f"The predicted language for the name "
            f"'{name}' is: {predicted_language}"
        )


if __name__ == "__main__":
    main()