# CORAL MLP for predicting cement strength (cement_strength)

This tutorial explains how to train a deep neural network (here: multilayer perceptron) with the CORAL layer and loss function for ordinal regression. 

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

In this section, we set up the data set and data loaders using PyTorch utilities. This is a general procedure that is not specific to CORAL.


```python
import torch


##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 20
batch_size = 128

# Architecture
NUM_CLASSES = 10

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
                         shuffle=True, # want to shuffle the dataset
                         num_workers=0) # number processes/CPUs to use

# Checking the dataset
for inputs, labels in train_loader:  
    print('Input batch dimensions:', inputs.shape)
    print('Input label dimensions:', labels.shape)
    break
```

    Input batch dimensions: torch.Size([128, 8])
    Input label dimensions: torch.Size([128])


## 2 - Equipping MLP with CORAL layer

In this section, we are using the CoralLayer implemented in `coral_pytorch` to outfit a multilayer perceptron for ordinal regression. Note that the CORAL method only requires replacing the last (output) layer, which is typically a fully-connected layer, by the CORAL layer.

Also, please use the `sigmoid` not softmax function (since the CORAL method uses a concept known as extended binary classification as described in the paper).


```python
from coral_pytorch.layers import CoralLayer



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
        )
        
        ### Specify CORAL layer
        self.fc = CoralLayer(size_in=num_hidden_2, num_classes=num_classes)
        ###--------------------------------------------------------------------###
        
    def forward(self, x):
        x = self.my_network(x)
        
        ##### Use CORAL layer #####
        logits =  self.fc(x)
        probas = torch.sigmoid(logits)
        ###--------------------------------------------------------------------###
        
        return logits, probas
    
    
    
torch.manual_seed(random_seed)
model = MLP(in_features=8, num_classes=NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

## 3 - Using the CORAL loss for model training

During training, all you need to do is to 

1) convert the integer class labels into the extended binary label format using the `levels_from_labelbatch` provided via `coral_pytorch`:

```python
        levels = levels_from_labelbatch(class_labels, 
                                        num_classes=NUM_CLASSES)
```

2) Apply the CORAL loss (also provided via `coral_pytorch`):

```python
        loss = coral_loss(logits, levels)
```



```python
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss


for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):

        ##### Convert class labels for CORAL
        levels = levels_from_labelbatch(class_labels, 
                                        num_classes=NUM_CLASSES)
        ###--------------------------------------------------------------------###

        features = features.to(DEVICE)
        levels = levels.to(DEVICE)
        logits, probas = model(features)
        
        #### CORAL loss 
        loss = coral_loss(logits, levels)
        ###--------------------------------------------------------------------###   
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), loss))
```

    Epoch: 001/020 | Batch 000/007 | Loss: 1.0222
    Epoch: 002/020 | Batch 000/007 | Loss: 1.1131
    Epoch: 003/020 | Batch 000/007 | Loss: 0.9594
    Epoch: 004/020 | Batch 000/007 | Loss: 0.9661
    Epoch: 005/020 | Batch 000/007 | Loss: 0.9792
    Epoch: 006/020 | Batch 000/007 | Loss: 1.0311
    Epoch: 007/020 | Batch 000/007 | Loss: 0.9157
    Epoch: 008/020 | Batch 000/007 | Loss: 0.8542
    Epoch: 009/020 | Batch 000/007 | Loss: 0.9652
    Epoch: 010/020 | Batch 000/007 | Loss: 0.9483
    Epoch: 011/020 | Batch 000/007 | Loss: 0.8316
    Epoch: 012/020 | Batch 000/007 | Loss: 0.9067
    Epoch: 013/020 | Batch 000/007 | Loss: 1.0139
    Epoch: 014/020 | Batch 000/007 | Loss: 0.8505
    Epoch: 015/020 | Batch 000/007 | Loss: 0.8289
    Epoch: 016/020 | Batch 000/007 | Loss: 0.8277
    Epoch: 017/020 | Batch 000/007 | Loss: 0.7669
    Epoch: 018/020 | Batch 000/007 | Loss: 0.8366
    Epoch: 019/020 | Batch 000/007 | Loss: 0.7514
    Epoch: 020/020 | Batch 000/007 | Loss: 0.8221



```python
from coral_pytorch.dataset import proba_to_label


def compute_mae_and_mse(model, data_loader, device):

    with torch.no_grad():
    
        mae, mse, acc, num_examples = 0., 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits, probas = model(features)
            predicted_labels = proba_to_label(probas).float()

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        return mae, mse
```

## 4 -- Evaluate model

Finally, after model training, we can evaluate the performance of the model. For example, via the mean absolute error and mean squared error measures.

For this, we are going to use the `proba_to_label` utility function from `coral_pytorch` to convert the probabilities back to the orginal label.



```python
from coral_pytorch.dataset import proba_to_label


def compute_mae_and_mse(model, data_loader, device):

    with torch.no_grad():
    
        mae, mse, acc, num_examples = 0., 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits, probas = model(features)
            predicted_labels = proba_to_label(probas).float()

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

    Mean absolute error (train/test): 0.27 | 0.34
    Mean squared error (train/test): 0.28 | 0.34

