# CORAL CNN for predicting handwritten digits (MNIST)

This tutorial explains how to equip a deep neural network with the CORAL layer and loss function for ordinal regression. Please note that **MNIST is not an ordinal dataset**. The reason why we use MNIST in this tutorial is that it is included in the PyTorch's `torchvision` library and is thus easy to work with, since it doesn't require extra data downloading and preprocessing steps.

## 1 -- Setting up the dataset and dataloader

In this section, we set up the data set and data loaders. This is a general procedure that is not specific to CORAL.


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
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
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


## 2 - Equipping CNN with CORAL layer

In this section, we are using the CoralLayer implemented in `coral_pytorch` to outfit a convolutional neural network for ordinal regression. Note that the CORAL method only requires replacing the last (output) layer, which is typically a fully-connected layer, by the CORAL layer.

Using the `Sequential` API, we specify the CORAl layer as 

```python
        self.fc = CoralLayer(size_in=294, num_classes=num_classes)
```

This is because the convolutional and pooling layers 

```python
            torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)))
```


produce a flattened feature vector of 294 units. Then, when using the CORAL layer in the forward function

```python
        logits =  self.fc(x)
        probas = torch.sigmoid(logits)
```

please use the `sigmoid` not softmax function (since the CORAL method uses a concept known as extended binary classification as described in the paper).


```python
from coral_pytorch.layers import CoralLayer



class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(3, 6, (3, 3), (1, 1), 1),
            torch.nn.MaxPool2d((2, 2), (2, 2)))
        
        ### Specify CORAL layer
        self.fc = CoralLayer(size_in=294, num_classes=num_classes)
        ###--------------------------------------------------------------------###
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        
        ##### Use CORAL layer #####
        logits =  self.fc(x)
        probas = torch.sigmoid(logits)
        ###--------------------------------------------------------------------###
        
        return logits, probas
    
    
    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters())
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
        cost = coral_loss(logits, levels)
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
        cost = coral_loss(logits, levels)
        ###--------------------------------------------------------------------###   
        
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 200:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
```

    Epoch: 001/010 | Batch 000/468 | Cost: 6.2250
    Epoch: 001/010 | Batch 200/468 | Cost: 4.5538
    Epoch: 001/010 | Batch 400/468 | Cost: 4.1572
    Epoch: 002/010 | Batch 000/468 | Cost: 4.2916
    Epoch: 002/010 | Batch 200/468 | Cost: 3.7469
    Epoch: 002/010 | Batch 400/468 | Cost: 3.7111
    Epoch: 003/010 | Batch 000/468 | Cost: 3.5796
    Epoch: 003/010 | Batch 200/468 | Cost: 3.2361
    Epoch: 003/010 | Batch 400/468 | Cost: 3.1930
    Epoch: 004/010 | Batch 000/468 | Cost: 3.2449
    Epoch: 004/010 | Batch 200/468 | Cost: 2.9884
    Epoch: 004/010 | Batch 400/468 | Cost: 2.7252
    Epoch: 005/010 | Batch 000/468 | Cost: 2.9845
    Epoch: 005/010 | Batch 200/468 | Cost: 2.7993
    Epoch: 005/010 | Batch 400/468 | Cost: 2.6468
    Epoch: 006/010 | Batch 000/468 | Cost: 2.7458
    Epoch: 006/010 | Batch 200/468 | Cost: 2.4976
    Epoch: 006/010 | Batch 400/468 | Cost: 2.5533
    Epoch: 007/010 | Batch 000/468 | Cost: 2.6634
    Epoch: 007/010 | Batch 200/468 | Cost: 2.5637
    Epoch: 007/010 | Batch 400/468 | Cost: 2.3448
    Epoch: 008/010 | Batch 000/468 | Cost: 2.3006
    Epoch: 008/010 | Batch 200/468 | Cost: 2.7393
    Epoch: 008/010 | Batch 400/468 | Cost: 2.1759
    Epoch: 009/010 | Batch 000/468 | Cost: 2.3998
    Epoch: 009/010 | Batch 200/468 | Cost: 2.2425
    Epoch: 009/010 | Batch 400/468 | Cost: 2.1656
    Epoch: 010/010 | Batch 000/468 | Cost: 2.3247
    Epoch: 010/010 | Batch 200/468 | Cost: 2.3120
    Epoch: 010/010 | Batch 400/468 | Cost: 2.4770


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

    Mean absolute error (train/test): 0.90 | 0.91
    Mean squared error (train/test): 1.84 | 1.87


Note that MNIST is not an ordinal dataset (there is no order between the image categories), so computing the MAE or MSE doesn't really make sense but we use it anyways for demonstration purposes.
