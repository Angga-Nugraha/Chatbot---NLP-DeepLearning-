import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_word

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# melakukan looping untuk setiap kalimat dalam pola
for intent in intents['intents']:
    # menambah ke dalam tags list
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize kata dalam setiap kalimat
        w = tokenize(pattern)
        # memasukan kedalam list kata
        all_words.extend(w)
        # menambah pasangan antara tag dan kata
        xy.append((w, tag))

# stemming dan merubah kata menjadi huruf kecil
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# menghapus duplicated dan terurut
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# membuat data training
x_train = []
y_train = []
for(pattern_sentence, tag) in xy:
    # X: sekantong kata untuk setiap pola kalimat
    bag =  bag_of_word(pattern_sentence, all_words)
    x_train.append(bag)
    # Y: PyTorch CrossEntropyLoss hanya membutuhkan label kelas, bukan one-hot
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
 
# Hyperparameter
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    # melakukan indexing pada data untuk mendapatkan sample ke-i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


correct = 0
total = 0
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for(words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            total += labels.shape[0]

            #forward
            outputs = model(words)
            loss = criterion(outputs, labels)
            correct += torch.sum(labels == outputs.argmax(dim=-1))
           
    
            #backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if(epoch +1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
    accuracy = correct/total

    print(f'final loss, loss = {loss.item():.4f}')
    print(f'Accuracy = {accuracy}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training data complete, file saves to {FILE}')