import random


import torch
from torch import nn
import torch.nn.functional as F
from unidecode import unidecode
import datacreation 

# Set random seed for reproducibility
_ = torch.manual_seed(42)
# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and testing datasets, number of languages, and number of letters from the datacreation module
train_dataset, test_dataset, num_langs, num_letters = datacreation.load_data()

# Define the SimpleRNN class, which is a subclass of the base nn.Module class from PyTorch
class SimpleRNN(nn.Module):
    # Constructor method taking input size, hidden size (recurrent), and output size as parameters
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__() # Call the constructor of the base class of Pytorch
        self.hidden_size = hidden_size # Store the hidden size for later use
        # Layer1 that takes the concatenated input and hidden state and maps it to the hidden state
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size) 
        # Layer2 that takes the concatenated input and hidden state and maps it to the output
        self.in2output = nn.Linear(input_size + hidden_size, output_size)

    # Forward method that defines the forward pass of the RNN taking an input tensor and a hidden state tensor as parameters
    def forward(self, x, hidden_state):
        # Concatenate the input tensor and the hidden state tensor along dimension 1
        combined = torch.cat((x, hidden_state), 1)
        # Pass the combined tensor through the in2hidden layer and apply the sigmoid activation function to get the new hidden state
        hidden = torch.sigmoid(self.in2hidden(combined))
        # Pass the combined tensor through the in2output layer to get the output tensor
        output = self.in2output(combined)
        # return the output tensor and the new hidden state tensor
        return output, hidden

    # Method to initialize the hidden state tensor at start of each sequence
    def init_hidden(self):
        # Initialize the hidden state tensor with random values using Kaiming uniform initialization
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

# Function to train the model using the training dataset, loss criterion, optimizer, and number of epochs as parameters
def train(model, train_dataset, criterion, optimizer, num_epochs, print_interval=100):
    # Loops over the number of epochs specified
    for epoch in range(num_epochs):
        # Shuffle the training dataset at the start of each epoch to ensure that the model does not learn any order-based patterns
        random.shuffle(train_dataset)
        # iterate over the traning dataset
        # contains the index for printing purpose
        # name is the tensor of the name string and label is the tensor of the target language's label int
        for i, (name, label) in enumerate(train_dataset):
            # inits the hidden state tensor at the start of each sequence
            hidden_state = model.init_hidden()

            # passes each char and the hidden state through the model to get the output and new hidden state
            for char in name:
                output, hidden_state = model(char, hidden_state)

            # Measures the loss between the model's output and the target label using the specified loss function
            loss = criterion(output, label)
            # Clears the previous gradients
            optimizer.zero_grad()
            # Calculates gradients using backpropagation
            loss.backward()
            # Clips the gradients to prevent exploding gradients by limiting their norm to 1
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Updates the model's parameters using the specified optimizer
            optimizer.step()

            # Prints the loss every print_interval steps to monitor the training progress
            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_dataset)}], "
                    f"Loss: {loss.item():.4f}"
                )

# Function to test the model using the testing dataset
def test(model, test_dataset):
    # tracks the number of correct predictions made by the model
    num_correct = 0
    # total number of samples in the test dataset
    num_samples = len(test_dataset)

    # Set the model to evaluation mode, which disables dropout and batch normalization layers
    model.eval()

    # Tells model to not compute the gradients during forward pass to save memory and computation
    with torch.no_grad():
        # iterates over test dataser
        for name, label in test_dataset:
            # inits hidden state tensor
            hidden_state = model.init_hidden()
            # passes each char and the hidden state through the model to get the output and new hidden state
            for char in name:
                output, hidden_state = model(char, hidden_state)
            # Gets the index of the maximum value in the output tensor along dimension 1
            _, pred = torch.max(output, dim=1)
            # Compares the predicted label with the true label and increments the num_correct counter if they match
            num_correct += bool(pred == label)
    # Prints accuracy
    print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")

# Function to predict the language of a given name using the trained model
def myrnn_predict(model, name):
    model.eval()
    tensor_name = datacreation.name2tensor(name)
    with torch.no_grad():
        hidden_state = model.init_hidden()
        for char in tensor_name:
            output, hidden_state = model(char, hidden_state)
        _, pred = torch.max(output, dim=1)
    model.train()    
    return datacreation.label2lang[pred.item()]


# Main function that initializes the model, loss criterion, optimizer, and trains and tests the model
def main():
    hidden_size = 256
    learning_rate = 0.001

    model = SimpleRNN(num_letters, hidden_size, num_langs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(
    model,
    train_dataset,
    criterion,
    optimizer,
    num_epochs=1
    )
    test(model, test_dataset)

    # Example prediction
    name = "Ahmed"
    predicted_language = myrnn_predict(model, name)
    print(f"The predicted language for the name '{name}' is: {predicted_language}")
    name = "Ali"
    predicted_language = myrnn_predict(model, name)
    print(f"The predicted language for the name '{name}' is: {predicted_language}")

if __name__ == "__main__":
    main()