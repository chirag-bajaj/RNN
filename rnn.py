import io
import glob
import os
import unicodedata
from unicodedata import category
import unidecode
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
print(n_letters)

def unicodeToAscii(s):
    return unidecode.unidecode(s)

print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

import torch

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import torch.nn as nn
import torch.optim as opt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size*1, output_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x, h):
        out, hid = self.rnn(x, h)
        out = out.squeeze()
        out = self.fc(out)
        out = self.log_softmax(out)
        return out, hid
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)]).to(device)
    line_tensor = lineToTensor(line).to(device)
    return category, line, category_tensor, line_tensor

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories).to(device)

# Example
#input = lineToTensor('Albert')
#print(input.shape)
#hidden = torch.zeros(1, n_hidden)
#output, next_hidden = rnn(input[0].reshape(1, 1, input[0].shape[1]), hidden.reshape(1, 1, hidden.shape[1]))
#print(output.shape, next_hidden.shape)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = opt.Adam(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden().to(device)

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].reshape(1, 1, line_tensor[i].shape[1]), hidden)

    #output = torch.tensor([torch.argmax(output)])
    #print(output.shape)
    output = output.reshape(1, output.shape[0])
    #print("output", output)
    #print("CT", category_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

num_epochs = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

start = time.time()

for epoch in range(1, num_epochs+1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)

    if epoch % print_every == 0:
        print('%d %d%% (%s) %.4f'  % (epoch, epoch / num_epochs * 100, timeSince(start), loss))

    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)   
