# CORN CNN for predicting handwritten digits (MNIST)

This tutorial explains how to train a deep neural network with the CORN loss function for ordinal regression. Please note that **MNIST is not an ordinal dataset**. The reason why we use MNIST in this tutorial is that it is included in the PyTorch's `torchvision` library and is thus easy to work with, since it doesn't require extra data downloading and preprocessing steps.

## 1 -- Setting up the dataset and dataloader

In this section, we set up the data set and data loaders. This is a general procedure that is not specific to CORN.


```python
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

##########################
### SETTINGS
##########################

# Hyperparameters
random_seed = 1
learning_rate = 0.05
num_epochs = 10
batch_size = 128

# Architecture
NUM_CLASSES = 10 

# Other
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', DEVICE)

##########################
### MNIST DATASET
##########################


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='../data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          drop_last=True,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         drop_last=True,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
```

    Training on cuda:0
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
    
    Image batch dimensions: torch.Size([128, 1, 28, 28])
    Image label dimensions: torch.Size([128])


## 2 - Equipping CNN with a CORN layer

In this section, we are implementing a simple CNN for ordinal regression with CORN. Note that the only specific modification required is setting the number of output of the last layer (a fully connected layer) to the number of classes - 1 (these correspond to the binary tasks used in the extended binary classification as described in the paper).


```python
class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)))
        
        ### Specify CORN layer
        self.output_layer = torch.nn.Linear(in_features=294, out_features=num_classes-1)
        ###--------------------------------------------------------------------###
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        
        ##### Use CORN layer #####
        logits =  self.output_layer(x)
        ###--------------------------------------------------------------------###
        
        return logits
    
    
    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
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

    Epoch: 001/010 | Batch 000/468 | Cost: 0.6896
    Epoch: 001/010 | Batch 200/468 | Cost: 0.1449
    Epoch: 001/010 | Batch 400/468 | Cost: 0.0761
    Epoch: 002/010 | Batch 000/468 | Cost: 0.0927
    Epoch: 002/010 | Batch 200/468 | Cost: 0.0679
    Epoch: 002/010 | Batch 400/468 | Cost: 0.0714
    Epoch: 003/010 | Batch 000/468 | Cost: 0.0593
    Epoch: 003/010 | Batch 200/468 | Cost: 0.0516
    Epoch: 003/010 | Batch 400/468 | Cost: 0.0470
    Epoch: 004/010 | Batch 000/468 | Cost: 0.0301
    Epoch: 004/010 | Batch 200/468 | Cost: 0.0417
    Epoch: 004/010 | Batch 400/468 | Cost: 0.0366
    Epoch: 005/010 | Batch 000/468 | Cost: 0.0449
    Epoch: 005/010 | Batch 200/468 | Cost: 0.0380
    Epoch: 005/010 | Batch 400/468 | Cost: 0.0141
    Epoch: 006/010 | Batch 000/468 | Cost: 0.0272
    Epoch: 006/010 | Batch 200/468 | Cost: 0.0267
    Epoch: 006/010 | Batch 400/468 | Cost: 0.0405
    Epoch: 007/010 | Batch 000/468 | Cost: 0.0649
    Epoch: 007/010 | Batch 200/468 | Cost: 0.0253
    Epoch: 007/010 | Batch 400/468 | Cost: 0.0215
    Epoch: 008/010 | Batch 000/468 | Cost: 0.0389
    Epoch: 008/010 | Batch 200/468 | Cost: 0.0297
    Epoch: 008/010 | Batch 400/468 | Cost: 0.0343
    Epoch: 009/010 | Batch 000/468 | Cost: 0.0249
    Epoch: 009/010 | Batch 200/468 | Cost: 0.0498
    Epoch: 009/010 | Batch 400/468 | Cost: 0.0300
    Epoch: 010/010 | Batch 000/468 | Cost: 0.0201
    Epoch: 010/010 | Batch 200/468 | Cost: 0.0290
    Epoch: 010/010 | Batch 400/468 | Cost: 0.0303


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

    Mean absolute error (train/test): 0.15 | 0.15
    Mean squared error (train/test): 0.69 | 0.74


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

    tensor([[1.0000e+00, 1.0000e+00, 1.0000e+00,  ..., 9.9986e-01, 9.9941e-01,
             2.5950e-08],
            [1.0000e+00, 1.0000e+00, 9.9315e-01,  ..., 9.8477e-01, 9.8476e-01,
             9.7987e-08],
            [9.1224e-01, 9.1223e-01, 9.1223e-01,  ..., 8.5374e-01, 8.5216e-01,
             1.6753e-03],
            ...,
            [9.9812e-01, 9.9811e-01, 9.9811e-01,  ..., 9.8991e-01, 9.8968e-01,
             4.1033e-03],
            [9.9979e-01, 9.9979e-01, 9.9979e-01,  ..., 1.5020e-02, 1.5015e-02,
             2.7997e-04],
            [7.7070e-07, 7.7070e-07, 7.5224e-07,  ..., 7.6964e-08, 7.6941e-08,
             6.1278e-13]], device='cuda:0')

