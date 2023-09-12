import sys
import torch
import numpy as np
from torch import nn


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, vocab, sequence_length=32):
        self.text = text
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.sequence_length = sequence_length
        self.char_to_int = {ch: i for i, ch in enumerate(self.vocab)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        input_sequence = self.text[idx : idx + self.sequence_length]
        target_sequence = self.text[idx + 1 : idx + self.sequence_length + 1]
        input_data = torch.tensor(
            [self.char_to_int[ch] for ch in input_sequence], dtype=torch.long
        )
        target_data = torch.tensor(
            [self.char_to_int[ch] for ch in target_sequence], dtype=torch.long
        )
        return input_data, target_data[-1]


def load_text():
    with open("shakespeare.txt") as f:
        return f.read()


text = load_text()
vocab = sorted(list(set(text)))
trainset = TextDataset(text, vocab)
testset = TextDataset(text, vocab)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
i, t = next(dataiter)


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 100)
        self.rnn = nn.RNN(100, 200, batch_first=True)
        self.fc = nn.Linear(200, len(vocab))

    def forward(self, X):
        X = self.embedding(X)
        X, _ = self.rnn(X)
        X = self.fc(X[:, -1, :])
        return X


model = LanguageModel()


def train_model():
    model.load_state_dict(torch.load("./language-model.pth"))
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 1

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            X_train, y_train = data
            # X_train = X_train.float()
            # y_train = y_train.float()
            y_pred = model.forward(X_train)
            # print(X_train)
            # print(y_train)
            # print(y_pred)
            # input()
            loss = loss_function(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
            if i % 10000 == 9999:
                torch.save(model.state_dict(), "./language-model.pth")

    torch.save(model.state_dict(), "./language-model.pth")


def test_model():
    model.load_state_dict(torch.load("./language-model.pth"))
    model.eval()

    for i_batch, t_batch in iter(testloader):
        i = i_batch[0]
        t = t_batch[0]
        ij = i.unsqueeze(0)
        prediction = model(ij).argmax()
        print(prediction)
        print(
            'input: "' + "".join([vocab[x] for x in i]) + '"',
            ' target: "' + vocab[t] + '"',
            ' pred: "' + vocab[prediction.item()] + '"',
        )
        input()


def sample(logits, temperature=1.0):
    # Apply temperature
    logits = logits / temperature
    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=0)
    # Sample from the distribution
    return torch.multinomial(probs, 1)


def generate_text():
    model.load_state_dict(torch.load("./language-model.pth"))
    model.eval()  # Sets the model to evaluation mode

    prompt = "Are thou?"
    length = 128
    char_to_int = {ch: i for i, ch in enumerate(vocab)}

    input_text = [char_to_int[ch] for ch in prompt]  # Convert prompt to tensor indices
    for _ in range(length):
        i = torch.tensor(input_text[-32:])  # Use the last predicted character as input
        i = i.unsqueeze(
            0
        )  # Adds an extra dimension at the beginning to simulate a batch size of 1
        output = model(i)  # Forward pass
        prediction = sample(
            output.squeeze(), 0.7
        )  # Squeezes the output and gets the index of the max value
        input_text.append(prediction.item())  # Append to input_text

    generated_text = "".join(
        [vocab[x] for x in input_text]
    )  # Convert tensor indices to text
    print(generated_text)


if __name__ == "__main__" and len(sys.argv) > 1:
    globals()[sys.argv[1]]()
