#importing libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Section I-importing data load the dataset, split into input (X) and output (y) variables. 
dataset = np.genfromtxt('practice/machine_learning/diabetes.csv', delimiter=',', usecols=range(9))      #Loading dataset Also np.loadtxt('diabetes.csv', delimiter=',')
X = dataset[1:,0:8] #getting the 8 first columns as input.The reason for "1"instead of "0" is todiscard the headings
y = dataset[1:,8] #getting the last column as output. The reason for "1"instead of "0" is todiscard the headings
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
#Splitting into train and validation datasets with 67% trainset and 33% validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Section II-defining the model:  linear regression model with 2 layers, input and output
model = nn.Sequential(
  nn.Linear(8, 1),
  nn.Sigmoid()
)
#priniting model layers and specifications
print(model)

#Secion III-training the model
n_epochs = 100
batch_size = 8
history = []
# define loss function
loss_fn = nn.MSELoss()
# define optimizer with a spicific learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(n_epochs):
  for i in range(0, len(X_train), batch_size):
        # take a batch
        Xbatch = X_train[i:i+batch_size]
        ybatch = y_train[i:i+batch_size]
        # forward pass
        y_pred =model(Xbatch) #  torch.max(model(Xbatch), 1)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

  model.eval()
  y_pred = model(X_test)
  mse = loss_fn(y_pred, y_test)
  #mse_train = loss_fn(y_, y_test)
  mse = float(mse)
  history.append(mse)
  print(f'Finished epoch {epoch}, latest MSE {mse}')

#Section IV-evaluating the model
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

#Section V-visualizing the outputs
plt.plot(history)
plt.title('Mean Square Error')
plt.xlabel("Epoch")
plt.ylabel("")
plt.show()