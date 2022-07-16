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

    Training on cuda:0



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

        class_labels = class_labels.to(DEVICE)
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

    Epoch: 001/020 | Batch 000/007 | Cost: 0.7095
    Epoch: 002/020 | Batch 000/007 | Cost: 0.5793
    Epoch: 003/020 | Batch 000/007 | Cost: 0.5107
    Epoch: 004/020 | Batch 000/007 | Cost: 0.4893
    Epoch: 005/020 | Batch 000/007 | Cost: 0.4294
    Epoch: 006/020 | Batch 000/007 | Cost: 0.3942
    Epoch: 007/020 | Batch 000/007 | Cost: 0.3905
    Epoch: 008/020 | Batch 000/007 | Cost: 0.3877
    Epoch: 009/020 | Batch 000/007 | Cost: 0.3327
    Epoch: 010/020 | Batch 000/007 | Cost: 0.3442
    Epoch: 011/020 | Batch 000/007 | Cost: 0.3513
    Epoch: 012/020 | Batch 000/007 | Cost: 0.3395
    Epoch: 013/020 | Batch 000/007 | Cost: 0.3272
    Epoch: 014/020 | Batch 000/007 | Cost: 0.3372
    Epoch: 015/020 | Batch 000/007 | Cost: 0.2994
    Epoch: 016/020 | Batch 000/007 | Cost: 0.3409
    Epoch: 017/020 | Batch 000/007 | Cost: 0.3158
    Epoch: 018/020 | Batch 000/007 | Cost: 0.2988
    Epoch: 019/020 | Batch 000/007 | Cost: 0.2793
    Epoch: 020/020 | Batch 000/007 | Cost: 0.2516


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

    Mean absolute error (train/test): 0.29 | 0.36
    Mean squared error (train/test): 0.34 | 0.39


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

    tensor([[8.4400e-01, 1.1552e-01, 2.4885e-02, 2.1235e-02],
            [9.6955e-01, 9.6440e-01, 7.9017e-01, 4.0131e-01],
            [9.6926e-01, 9.6164e-01, 2.8837e-01, 1.1151e-01],
            [2.7557e-01, 1.7854e-03, 1.3533e-04, 6.4534e-05],
            [4.4200e-04, 2.9050e-05, 1.8071e-05, 8.5216e-06],
            [4.1626e-02, 6.8911e-06, 1.1300e-06, 1.1232e-06],
            [9.5031e-01, 3.2661e-01, 7.6083e-03, 3.6258e-03],
            [9.8467e-01, 9.0953e-01, 4.3580e-01, 3.9399e-01],
            [8.0870e-01, 1.9610e-01, 1.9341e-02, 1.6238e-03],
            [9.6289e-01, 7.2809e-01, 2.1034e-01, 1.4426e-01],
            [9.8087e-01, 3.4986e-01, 7.5893e-03, 2.1336e-04],
            [8.3218e-02, 2.9795e-04, 8.8117e-05, 7.7257e-05],
            [6.4886e-01, 3.3336e-01, 1.7751e-01, 1.1291e-01],
            [8.0380e-01, 5.5894e-03, 3.1419e-04, 2.4602e-04],
            [9.3716e-01, 9.3670e-01, 9.3338e-01, 8.3394e-01],
            [9.0723e-01, 9.0255e-01, 8.7473e-01, 4.9182e-01],
            [9.8959e-01, 3.3517e-01, 5.4329e-02, 1.7331e-03],
            [9.6824e-01, 8.0327e-01, 2.5958e-01, 8.4942e-03],
            [9.6470e-01, 9.1665e-01, 6.9238e-01, 3.8931e-01],
            [9.6623e-01, 9.6491e-01, 9.4429e-01, 4.3117e-01],
            [8.0910e-02, 1.5353e-04, 2.7122e-05, 2.1541e-05],
            [9.9247e-01, 8.6671e-01, 6.3087e-01, 6.6279e-02],
            [8.8915e-01, 2.5603e-02, 1.8793e-03, 1.5186e-03],
            [6.2060e-01, 1.8354e-01, 4.0813e-02, 2.1553e-02],
            [9.5856e-01, 9.5805e-01, 9.2657e-01, 1.6030e-01],
            [9.9292e-01, 6.5836e-01, 1.8671e-01, 6.0837e-02],
            [1.0555e-01, 4.6840e-03, 1.1164e-03, 1.7749e-04],
            [9.6029e-01, 4.0485e-01, 3.0195e-02, 2.0155e-03],
            [9.8264e-01, 9.1183e-01, 4.3322e-01, 2.3925e-03],
            [8.9595e-01, 3.6590e-01, 3.0114e-02, 1.9936e-03]], device='cuda:0')



```python

```
