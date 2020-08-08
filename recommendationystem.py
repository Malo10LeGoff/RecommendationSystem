# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:59:38 2020

@author: LENOVO
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as t
import torch.optim as optim
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


### Recommendation system with an autoencoder

### Create the class of the autoencoder with Pytorch


class SAE(t.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.layer1 = t.Linear(nb_movies, 20)
        self.layer2 = t.Linear(20, 10)
        self.layer3 = t.Linear(10, 20)
        self.layer4 = t.Linear(20, nb_movies)
        self.activation = t.Sigmoid()
        
    def forward(self, x):
        output = self.activation(self.layer1(x))
        output = self.activation(self.layer2(output))
        output = self.activation(self.layer3(output))
        output = self.layer4(output)
        return output

sae = SAE()

criterion = t.MSELoss()

optimizer = optim.RMSprop(sae.parameters(), lr = 0.001, weight_decay = 0.5)

epochs = 3
for epoch in range(epochs):
    train_loss = 0
    s = 0.
    print("Epoch : " + str(epoch))
    for id_user in range(nb_users):
        input = Variable(training_set[id_user, :]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0: 
            target.requires_grad_ = False
            output = sae.forward(input)
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            optimizer.step()
            s += 1.
    print("Train Loss for this epoch : " + str(train_loss/s))
        


### Test phase, compare if we can predict the grade of a movie giving the training data for this user
### So we use the movie we already know the user watched (training set) to predict if he likes or not the movies in the test set


test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user, :]).unsqueeze(0)
    target = Variable(test_set[id_user, :]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: 
        target.requires_grad_ = False
        output = sae.forward(input)
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print("Test Loss for this epoch : " + str(test_loss/s))
        
        
        
        
        
        
        
        
        
        
        
        
