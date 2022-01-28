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

    Training on cpu
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

    Epoch: 001/010 | Batch 000/468 | Cost: 50.5479


    /Users/sebastian/miniforge3/lib/python3.9/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /tmp/pip-req-build-gqmopi53/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


    Epoch: 001/010 | Batch 200/468 | Cost: 9.7984
    Epoch: 001/010 | Batch 400/468 | Cost: 5.6653
    Epoch: 002/010 | Batch 000/468 | Cost: 6.6017
    Epoch: 002/010 | Batch 200/468 | Cost: 4.7958
    Epoch: 002/010 | Batch 400/468 | Cost: 4.7984
    Epoch: 003/010 | Batch 000/468 | Cost: 3.9449
    Epoch: 003/010 | Batch 200/468 | Cost: 3.6385
    Epoch: 003/010 | Batch 400/468 | Cost: 2.8829
    Epoch: 004/010 | Batch 000/468 | Cost: 2.0917
    Epoch: 004/010 | Batch 200/468 | Cost: 2.8083
    Epoch: 004/010 | Batch 400/468 | Cost: 2.6029
    Epoch: 005/010 | Batch 000/468 | Cost: 3.0181
    Epoch: 005/010 | Batch 200/468 | Cost: 2.5722
    Epoch: 005/010 | Batch 400/468 | Cost: 1.0547
    Epoch: 006/010 | Batch 000/468 | Cost: 1.8847
    Epoch: 006/010 | Batch 200/468 | Cost: 1.8378
    Epoch: 006/010 | Batch 400/468 | Cost: 2.7391
    Epoch: 007/010 | Batch 000/468 | Cost: 4.4030
    Epoch: 007/010 | Batch 200/468 | Cost: 1.7034
    Epoch: 007/010 | Batch 400/468 | Cost: 1.4372
    Epoch: 008/010 | Batch 000/468 | Cost: 2.5416
    Epoch: 008/010 | Batch 200/468 | Cost: 2.0749
    Epoch: 008/010 | Batch 400/468 | Cost: 2.3005
    Epoch: 009/010 | Batch 000/468 | Cost: 1.7815
    Epoch: 009/010 | Batch 200/468 | Cost: 3.4259
    Epoch: 009/010 | Batch 400/468 | Cost: 1.8984
    Epoch: 010/010 | Batch 000/468 | Cost: 1.4577
    Epoch: 010/010 | Batch 200/468 | Cost: 2.1422
    Epoch: 010/010 | Batch 400/468 | Cost: 2.0863


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

    Mean absolute error (train/test): 3.37 | 3.35
    Mean squared error (train/test): 17.28 | 16.98


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

    tensor([[1.0000e+00, 1.0000e+00, 1.0000e+00,  ..., 9.9987e-01, 9.9947e-01,
             2.6341e-08],
            [1.0000e+00, 1.0000e+00, 9.9275e-01,  ..., 9.8443e-01, 9.8443e-01,
             9.2676e-08],
            [9.1224e-01, 9.1224e-01, 9.1224e-01,  ..., 8.5583e-01, 8.5442e-01,
             1.7306e-03],
            ...,
            [9.9801e-01, 9.9800e-01, 9.9800e-01,  ..., 9.8942e-01, 9.8922e-01,
             4.1247e-03],
            [9.9977e-01, 9.9977e-01, 9.9977e-01,  ..., 1.5548e-02, 1.5543e-02,
             2.8278e-04],
            [7.4167e-07, 7.4167e-07, 7.2308e-07,  ..., 7.4769e-08, 7.4750e-08,
             5.6809e-13]])

