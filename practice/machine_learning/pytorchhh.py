import torch
import torch.nn as nn
import pandas as pd

# Tensor - is a multi-dimensional array of numbers, generalizing scalars
# vectors, and matrices.

# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# print(x_data)
# print(y_data)


data = {'Height' : [165.4, 175.9, 125.2, 189.5], 'Age' : [25, 30, 22, 35]}
df = pd.DataFrame(data)
X = df['Height']
Y = df['Age']
X = torch.tensor(X.values, dtype=torch.float32)
Y = torch.tensor(Y.values, dtype=torch.float32)
print(X)
print(Y)
