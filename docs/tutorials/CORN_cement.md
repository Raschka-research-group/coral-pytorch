# CORN MLP for predicting cement strength (cement_strength)

This tutorial explains how to train a deep neural network (here: multilayer perceptron) with the CORN loss function for ordinal regression. 

## 0 -- Obtaining and preparing the cement_strength dataset

We will be using the cement_strength dataset from [https://github.com/gagolews/ordinal_regression_data/blob/master/cement_strength.csv](https://github.com/gagolews/ordinal_regression_data/blob/master/cement_strength.csv).

First, we are going to download and prepare the and save it as CSV files locally. This is a general procedure that is not specific to CORN.

This dataset has 5 ordinal labels (1, 2, 3, 4, and 5). Note that CORN requires labels to be starting at 0, which is why we subtract "1" from the label column.


```python
import pandas as pd
import numpy as np


data_df = pd.read_csv("https://raw.githubusercontent.com/gagolews/ordinal_regression_data/master/cement_strength.csv")

data_df["response"] = data_df["response"]-1 # labels should start at 0

data_labels = data_df["response"]
data_features = data_df.loc[:, ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]]

print('Number of features:', data_features.shape[1])
print('Number of examples:', data_features.shape[0])
print('Labels:', np.unique(data_labels.values))
```

    Number of features: 8
    Number of examples: 998
    Labels: [0 1 2 3 4]


### Split into training and test data


```python
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    data_features.values,
    data_labels.values,
    test_size=0.2,
    random_state=1,
    stratify=data_labels.values)
```

### Standardize features


```python
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```

## 1 -- Setting up the dataset and dataloader

In this section, we set up the data set and data loaders. This is a general procedure that is not specific to CORN. 


```python
import torch


##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 20
batch_size = 128

# Architecture
NUM_CLASSES = 5

# Other
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', DEVICE)
```

    Training on cpu



```python
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, feature_array, label_array, dtype=np.float32):
    
        self.features = feature_array.astype(np.float32)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.labels.shape[0]
```


```python
import torch
from torch.utils.data import DataLoader


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = MyDataset(X_train_std, y_train)
test_dataset = MyDataset(X_test_std, y_test)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True, # want to shuffle the dataset
                          num_workers=0) # number processes/CPUs to use

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

# Checking the dataset
for inputs, labels in train_loader:  
    print('Input batch dimensions:', inputs.shape)
    print('Input label dimensions:', labels.shape)
    break
```

    Input batch dimensions: torch.Size([128, 8])
    Input label dimensions: torch.Size([128])


## 2 - Equipping MLP with a CORN layer

In this section, we are implementing a simple MLP for ordinal regression with CORN. Note that the only specific modification required is setting the number of output of the last layer (a fully connected layer) to the number of classes - 1 (these correspond to the binary tasks used in the extended binary classification as described in the paper).


```python
class MLP(torch.nn.Module):

    def __init__(self, in_features, num_classes, num_hidden_1=300, num_hidden_2=300):
        super().__init__()
        
        self.my_network = torch.nn.Sequential(
            
            # 1st hidden layer
            torch.nn.Linear(in_features, num_hidden_1, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_1),
            
            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_2),
            
            ### Specify CORN layer
            torch.nn.Linear(num_hidden_2, (num_classes-1))
            ###--------------------------------------------------------------------###
        )
                
    def forward(self, x):
        logits = self.my_network(x)
        return logits
    
    
    
torch.manual_seed(random_seed)
model = MLP(in_features=8, num_classes=NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

## 3 - Using the CORN loss for model training

During training, all you need to do is to use the `corn_loss` provided via `coral_pytorch`. The loss function will take care of the conditional training set processing and modeling the conditional probabilities used in the chain rule (aka general product rule). 


```python
from coral_pytorch.losses import corn_loss


for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):


        features = features.to(DEVICE)
        logits = model(features)
        
        #### CORN loss 
        loss = corn_loss(logits, class_labels, NUM_CLASSES)
        ###--------------------------------------------------------------------###   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), loss))
```

    Epoch: 001/020 | Batch 000/007 | Cost: 42.3084
    Epoch: 002/020 | Batch 000/007 | Cost: 33.4198
    Epoch: 003/020 | Batch 000/007 | Cost: 34.3413
    Epoch: 004/020 | Batch 000/007 | Cost: 27.4008
    Epoch: 005/020 | Batch 000/007 | Cost: 29.5580
    Epoch: 006/020 | Batch 000/007 | Cost: 28.3949
    Epoch: 007/020 | Batch 000/007 | Cost: 26.0713
    Epoch: 008/020 | Batch 000/007 | Cost: 24.0429
    Epoch: 009/020 | Batch 000/007 | Cost: 22.1783
    Epoch: 010/020 | Batch 000/007 | Cost: 25.2757
    Epoch: 011/020 | Batch 000/007 | Cost: 22.2279
    Epoch: 012/020 | Batch 000/007 | Cost: 23.1534
    Epoch: 013/020 | Batch 000/007 | Cost: 18.5136
    Epoch: 014/020 | Batch 000/007 | Cost: 23.8390
    Epoch: 015/020 | Batch 000/007 | Cost: 18.9016
    Epoch: 016/020 | Batch 000/007 | Cost: 18.0890
    Epoch: 017/020 | Batch 000/007 | Cost: 13.8526
    Epoch: 018/020 | Batch 000/007 | Cost: 17.3017
    Epoch: 019/020 | Batch 000/007 | Cost: 15.3039
    Epoch: 020/020 | Batch 000/007 | Cost: 16.0646


## 4 -- Evaluate model

Finally, after model training, we can evaluate the performance of the model. For example, via the mean absolute error and mean squared error measures.

For this, we are going to use the `corn_label_from_logits` utility function from `coral_pytorch` to convert the probabilities back to the orginal label.



```python
from coral_pytorch.dataset import corn_label_from_logits


def compute_mae_and_mse(model, data_loader, device):

    with torch.no_grad():
    
        mae, mse, acc, num_examples = 0., 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            predicted_labels = corn_label_from_logits(logits).float()

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        return mae, mse
```


```python
train_mae, train_mse = compute_mae_and_mse(model, train_loader, DEVICE)
test_mae, test_mse = compute_mae_and_mse(model, test_loader, DEVICE)
```


```python
print(f'Mean absolute error (train/test): {train_mae:.2f} | {test_mae:.2f}')
print(f'Mean squared error (train/test): {train_mse:.2f} | {test_mse:.2f}')
```

    Mean absolute error (train/test): 0.27 | 0.38
    Mean squared error (train/test): 0.30 | 0.43


Note that MNIST is not an ordinal dataset (there is no order between the image categories), so computing the MAE or MSE doesn't really make sense but we use it anyways for demonstration purposes.

## 5 -- Rank probabilities from logits

To obtain the rank probabilities from the logits, you can use the sigmoid function to get the conditional probabilities for each task and then compute the task probabilities via the chain rule for probabilities. Note that this is also done internally by the `corn_label_from_logits` we used above.


```python
logits = model(features)

with torch.no_grad():
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    print(probas)
```

    tensor([[9.7585e-01, 9.7536e-01, 9.6610e-01, 2.2675e-01],
            [9.4921e-01, 1.7135e-01, 9.7073e-03, 3.4552e-04],
            [2.4214e-02, 4.6617e-04, 3.3125e-05, 2.7729e-05],
            [9.8303e-01, 5.6925e-01, 3.2523e-01, 5.3245e-02],
            [9.5153e-01, 5.7437e-01, 1.4063e-01, 1.6096e-02],
            [1.8260e-02, 7.7530e-06, 5.1410e-06, 5.1152e-06],
            [9.7835e-01, 9.5596e-01, 3.6877e-01, 1.9158e-02],
            [9.8692e-01, 9.4746e-01, 3.2161e-01, 4.4179e-02],
            [3.5250e-01, 6.7358e-03, 2.6506e-03, 2.0687e-03],
            [9.2053e-01, 3.5487e-01, 1.0667e-01, 5.3238e-02],
            [1.2923e-01, 2.7282e-03, 3.3910e-04, 2.3258e-04],
            [9.7515e-01, 8.6582e-01, 2.0421e-01, 4.9532e-02],
            [9.9591e-01, 4.9053e-01, 3.7695e-02, 1.0938e-02],
            [1.8391e-01, 5.2724e-03, 5.6778e-04, 3.7333e-04],
            [8.8100e-01, 7.3442e-01, 5.6249e-01, 1.2397e-01],
            [6.7335e-01, 5.0340e-02, 8.1046e-03, 5.6927e-03],
            [9.0837e-01, 2.8030e-01, 3.3884e-02, 7.7040e-03],
            [8.9811e-01, 2.3693e-01, 2.2545e-02, 3.1750e-03],
            [9.5224e-01, 7.0664e-01, 1.0633e-01, 5.8928e-02],
            [7.1353e-01, 1.1418e-02, 4.1087e-04, 2.9408e-04],
            [4.8610e-04, 3.1541e-05, 2.3382e-05, 2.1088e-05],
            [9.6261e-01, 7.2110e-01, 7.1777e-01, 1.7590e-01],
            [9.7763e-01, 2.6866e-01, 4.9495e-02, 2.0072e-02],
            [9.7093e-01, 9.4531e-01, 9.1623e-02, 4.7309e-02],
            [8.3720e-01, 7.3658e-02, 1.1421e-02, 7.7481e-03],
            [9.2363e-01, 9.0403e-01, 5.8055e-01, 7.3952e-03],
            [9.7614e-01, 9.7225e-01, 6.8393e-01, 2.5626e-02],
            [9.7249e-01, 4.3916e-01, 2.6588e-01, 1.7958e-01],
            [9.7547e-01, 9.5796e-01, 8.4703e-01, 7.7577e-01],
            [9.5975e-01, 9.5148e-01, 4.9699e-01, 6.1305e-02]])



```python

```
