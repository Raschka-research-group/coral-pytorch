#!python -m spacy download en_core_web_sm
```

<a href="https://pytorch.org"><img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.svg" width="90"/></a> &nbsp; &nbsp;&nbsp;&nbsp;<a href="https://www.pytorchlightning.ai"><img src="https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_static/images/logo.svg" width="150"/></a>

# A Recurrent Neural Net for Ordinal Regression using CORAL -- TripAdvisor Dataset

In this tutorial, we implement a recurrent neural network for ordinal regression based on the CORAL method. To learn more about CORAL, please have a look at our paper:

- Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka (2020): 
[Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation](https://www.sciencedirect.com/science/article/pii/S016786552030413X). 
Pattern Recognition Letters. 140, 325-331

We will be using a balanced version of the [TripAdvisor Hotel Review](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) dataset that [we used](https://github.com/Raschka-research-group/corn-ordinal-neuralnet/blob/main/datasets/tripadvisor/tripadvisor_balanced.csv) in the CORN manuscript.

## General settings and hyperparameters

- Here, we specify some general hyperparameter values and general settings
- Note that for small datatsets, it is not necessary and better not to use multiple workers as it can sometimes cause issues with too many open files in PyTorch. So, if you have problems with the data loader later, try setting `NUM_WORKERS = 0` instead.


```python
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.0005
NUM_WORKERS = 4
RANDOM_SEED = 123

# Architecture:
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

# Dataset specific:

NUM_CLASSES = 5
VOCABULARY_SIZE = 20000
DATA_BASEPATH = "./data"
```

## Converting a regular classifier into a CORAL ordinal regression model

Changing a classifier to a CORAL model for ordinal regression is actually really simple and only requires a few changes:

**1)**
We replace the output layer  

```python
output_layer = torch.nn.Linear(hidden_units[-1], num_classes)
```

by a CORAL layer (available through `coral_pytorch`):

```python
output_layer = CoralLayer(size_in=hidden_units[-1], num_classes=num_classes)`
```

**2)**

Convert the integer class labels into the extended binary label format using the `levels_from_labelbatch` provided via `coral_pytorch`:

```python
levels = levels_from_labelbatch(class_labels, 
                                num_classes=num_classes)
```

**3)** 

Swap the cross entropy loss from PyTorch,

```python
torch.nn.functional.cross_entropy(logits, true_labels)
```

with the CORAL loss (also provided via `coral_pytorch`):

```python
loss = coral_loss(logits, levels)
```

**4)**

In a regular classifier, we usually obtain the predicted class labels as follows:

```python
predicted_labels = torch.argmax(logits, dim=1)
```

Replace this with the following code to convert the predicted probabilities into the predicted labels:

```python
predicted_labels = proba_to_label(probas)
```

## Implementing an `RNN` using PyTorch Lightning's `LightningModule`

- In this section, we set up the main model architecture using the `LightningModule` from PyTorch Lightning.
- We start with defining our recurrent neural network (`RNN`) model in pure PyTorch, and then we use it in the `LightningModule` to get all the extra benefits that PyTorch Lightning provides.


```python
import torch
from coral_pytorch.layers import CoralLayer


# Regular PyTorch Module
class PyTorchRNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim,
                 hidden_dim, num_classes):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        # self.rnn = torch.nn.RNN(embedding_dim,
        #                         hidden_dim,
        #                         nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)        

        # CORAL: output layer -------------------------------------------
        # Regular classifier would use the following output layer:
        # self.output_layer = torch.nn.Linear(hidden_dim, num_classes)
        
        # We replace it by the CORAL layer:
        self.output_layer = CoralLayer(size_in=hidden_dim,
                                       num_classes=num_classes)
        # ----------------------------------------------------------------

    def forward(self, text, text_length):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.output_layer(hidden)
        logits = output.view(-1, (self.num_classes-1))

        return logits
```

- In our `LightningModule` we use loggers to track mean absolute errors for both the training and validation set during training; this allows us to select the best model based on validation set performance later.
- Note that we make changes 2) (`levels_from_labelbatch`), 3) (`coral_loss`), and 4) (`proba_to_label`) to implement a CORAL model instead of a regular classifier:


```python
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.dataset import proba_to_label

import pytorch_lightning as pl
import torchmetrics


# LightningModule that receives a PyTorch model as input
class LightningRNN(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.input_dim = model.input_dim
        self.embedding_dim = model.embedding_dim
        self.hidden_dim = model.hidden_dim
        self.num_classes = model.num_classes
        
        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        
    # (Re)Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, text, text_length):
        return self.model(text, text_length)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        
        # These next 3 steps are unique and look a bit tricky due to
        # how Torchtext's BucketIterator prepares the batches
        # and how we use an LSTM with packed & padded text
        # Also, .TEXT_COLUMN_NAME and .LABEL_COLUMN_NAME
        # depend on the CSV file columns of the data file we load later.
        features, text_length = batch.TEXT_COLUMN_NAME
        true_labels = batch.LABEL_COLUMN_NAME
        
        # Convert class labels for CORAL ------------------------
        levels = levels_from_labelbatch(
            true_labels, num_classes=self.model.num_classes).type_as(features)
        # -------------------------------------------------------
        
        logits = self(features, text_length)

        # CORAL Loss --------------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, true_labels)
        loss = coral_loss(logits, levels)
        # -------------------------------------------------------

        # CORAL Prediction to label -----------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        probas = torch.sigmoid(logits)
        predicted_labels = proba_to_label(probas)
        # -----------------------------------------------------
        
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss, batch_size=true_labels.shape[0])
        self.train_mae(predicted_labels, true_labels)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False,
                 batch_size=true_labels.shape[0])
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss, batch_size=true_labels.shape[0])
        self.valid_mae(predicted_labels, true_labels)
        self.log("valid_mae", self.valid_mae,
                 on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=true_labels.shape[0])

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_mae(predicted_labels, true_labels)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False,
                 batch_size=true_labels.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
```

## Setting up the dataset

- In this section, we are going to set up our dataset.

### Inspecting the dataset


```python
import pandas as pd
import numpy as np


data_df = pd.read_csv(
    "https://raw.githubusercontent.com/Raschka-research-group/"
    "corn-ordinal-neuralnet/main/datasets/"
    "tripadvisor/tripadvisor_balanced.csv")

data_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TEXT_COLUMN_NAME</th>
      <th>LABEL_COLUMN_NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6995</th>
      <td>beautiful hotel, stay punta cana majestic colo...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6996</th>
      <td>stay, n't stay, stayed week april, weather ama...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6997</th>
      <td>stay hotel fantastic, great location, looked n...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6998</th>
      <td>birthday meal havnt stayed hotel staying barce...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6999</th>
      <td>great hotel great location stayed royal magda ...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
import os


CSV_PATH = os.path.join(DATA_BASEPATH, 'tripadvisor_balanced.csv')
data_df.to_csv(CSV_PATH, index=None)
```


```python
import torchtext
import random


TEXT = torchtext.legacy.data.Field(
    tokenize='spacy',  # default splits on whitespace
    tokenizer_language='en_core_web_sm',
    include_lengths=True
)

LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

fields = [('TEXT_COLUMN_NAME', TEXT), ('LABEL_COLUMN_NAME', LABEL)]

dataset = torchtext.legacy.data.TabularDataset(
    path=CSV_PATH, 
    format='csv',
    skip_header=True,
    fields=fields)

train_data, test_data = dataset.split(
    split_ratio=[0.8, 0.2],
    random_state=random.seed(RANDOM_SEED))

train_data, valid_data = train_data.split(
    split_ratio=[0.85, 0.15],
    random_state=random.seed(RANDOM_SEED))

TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train_data)

train_loader, valid_loader, test_loader = \
    torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,  # necessary for packed_padded_sequence
        sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
)
```

    39:1: E122 continuation line missing indentation or outdented



```python
# Checking the dataset
all_train_labels = []
all_test_labels = []

for features, labels in train_loader:  
    all_train_labels.append(labels)
all_train_labels = torch.cat(all_train_labels)
    
for features, labels in test_loader:  
    all_test_labels.append(labels)
all_test_labels = torch.cat(all_test_labels)

print('Training labels:', torch.unique(all_train_labels))
print('Training label distribution:', torch.bincount(all_train_labels))

print('\nTest labels:', torch.unique(all_test_labels))
print('Test label distribution:', torch.bincount(all_test_labels))
```

    Training labels: tensor([0, 1, 2, 3, 4], device='cuda:0')
    Training label distribution: tensor([964, 963, 954, 953, 926], device='cuda:0')
    
    Test labels: tensor([0, 1, 2, 3, 4], device='cuda:0')
    Test label distribution: tensor([275, 267, 300, 274, 284], device='cuda:0')


- Above, we can see that the dataset consists of 8 features, and there are 998 examples in total.
- The labels are in range from 1 (weakest) to 5 (strongest), and we normalize them to start at zero (hence, the normalized labels are in the range 0 to 4). 
- Notice also that the dataset is quite balanced.

### Performance baseline

- Especially for imbalanced datasets, it's quite useful to compute a performance baseline.
- In classification contexts, a useful baseline is to compute the accuracy for a scenario where the model always predicts the majority class -- you want your model to be better than that!
- Note that if you are intersted in a single number that minimized the dataset mean squared error (MSE), that's the mean; similary, the median is a number that minimzes the mean absolute error (MAE).
- So, if we use the mean absolute error, $\mathrm{MAE}=\frac{1}{N} \sum_{i=1}^{N}\left|y_{i}-\hat{y}_{i}\right|$, to evaluate the model, it is useful to compute the MAE pretending the predicted label is always the median:


```python
all_test_labels = all_test_labels.float()
avg_prediction = torch.median(all_test_labels)  # median minimizes MAE
baseline_mae = torch.mean(torch.abs(all_test_labels - avg_prediction))
print(f'Baseline MAE: {baseline_mae:.2f}')
```

    Baseline MAE: 1.18


- In other words, a model that would always predict the dataset median would achieve a MAE of 1.18. A model that has an MAE of > 1.18 is certainly a bad model.

### Setting up a `DataModule`

- There are three main ways we can prepare the dataset for Lightning. We can
  1. make the dataset part of the model;
  2. set up the data loaders as usual and feed them to the fit method of a Lightning Trainer -- the Trainer is introduced in the next subsection;
  3. create a `LightningDataModule`.
- Usually, approach 3 is the most organized approach. However, since we already defined our data loaders above, we can just work with those directly.

- Note that the `prepare_data` method is usually used for steps that only need to be executed once, for example, downloading the dataset; the `setup` method defines the the dataset loading -- if you run your code in a distributed setting, this will be called on each node / GPU. 
- Next, lets initialize the `DataModule`; we use a random seed for reproducibility (so that the data set is shuffled the same way when we re-execute this code):

## Training the model using the PyTorch Lightning Trainer class

- Next, we initialize our `RNN` model.
- Also, we define a call back so that we can obtain the model with the best validation set performance after training.
- PyTorch Lightning offers [many advanced logging services](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html) like Weights & Biases. Here, we will keep things simple and use the `CSVLogger`:


```python
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


pytorch_model = PyTorchRNN(
    input_dim=len(TEXT.vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES)

lightning_model = LightningRNN(
    pytorch_model, learning_rate=LEARNING_RATE)

callbacks = [ModelCheckpoint(
    save_top_k=1, mode='min', monitor="valid_mae")]  # save top 1 model 
logger = CSVLogger(save_dir="logs/", name="rnn-coral-mnist")
```

- Note that we disable warning as the `.log()` method of the `LightningModule` currently warns us that the batch size is inconsistent. This should not happen as we define the `batch_size` manually in the `self.log` calls. However, this will be resolved in a future version (https://github.com/PyTorchLightning/pytorch-lightning/pull/10408). 

- Also note that the batch size is not inconsistent, its just that the `BucketIterator` in torchtext has creates batches where the text length plus padding is the first dimension in a tensor. And the batch size is the second dimension:


```python
for features, labels in train_loader:  
    break

print('Text length:', features[0].shape[0])
print('Batch size (from text):', features[0].shape[1])
print('Batch size (from labels):', labels.shape[0])
```

    Text length: 469
    Batch size (from text): 16
    Batch size (from labels): 16


- Now it's time to train our model:


```python
import time


trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=callbacks,
    accelerator="auto",  # Uses GPUs or TPUs if available
    devices="auto",  # Uses all available GPUs/TPUs if applicable
    logger=logger,
    deterministic=True,
    log_every_n_steps=10)

start_time = time.time()
trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader)

runtime = (time.time() - start_time)/60
print(f"Training took {runtime:.2f} min in total.")
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name      | Type              | Params
    ------------------------------------------------
    0 | model     | PyTorchRNN        | 3.0 M 
    1 | train_mae | MeanAbsoluteError | 0     
    2 | valid_mae | MeanAbsoluteError | 0     
    3 | test_mae  | MeanAbsoluteError | 0     
    ------------------------------------------------
    3.0 M     Trainable params
    0         Non-trainable params
    3.0 M     Total params
    11.823    Total estimated model params size (MB)



    Validation sanity check: 0it [00:00, ?it/s]

    Training took 3.39 min in total.


## Evaluating the model

- After training, let's plot our training MAE and validation MAE using pandas, which, in turn, uses matplotlib for plotting (you may want to consider a [more advanced logger](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html) that does that for you):


```python
import pandas as pd


metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train_loss", "valid_loss"]].plot(
    grid=True, legend=True, xlabel='Epoch', ylabel='Loss')
df_metrics[["train_mae", "valid_mae"]].plot(
    grid=True, legend=True, xlabel='Epoch', ylabel='MAE')
```




    <AxesSubplot:xlabel='Epoch', ylabel='MAE'>




    
![png](ordinal-coral_tripadvisor_files/ordinal-coral_tripadvisor_40_1.png)
    



    
![png](ordinal-coral_tripadvisor_files/ordinal-coral_tripadvisor_40_2.png)
    


- As we can see from the loss plot above, the model starts overfitting pretty quickly. Based on the MAE plot, we can see that the best model, based on the validation set MAE, may be around epoch 5.
- The `trainer` saved this model automatically for us, we which we can load from the checkpoint via the `ckpt_path='best'` argument; below we use the `trainer` instance to evaluate the best model on the test set:


```python
trainer.test(model=lightning_model, dataloaders=test_loader, ckpt_path='best')
```

    Restoring states from the checkpoint path at logs/rnn-coral-mnist/version_16/checkpoints/epoch=38-step=11621.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from checkpoint at logs/rnn-coral-mnist/version_16/checkpoints/epoch=38-step=11621.ckpt
    /home/jovyan/conda/lib/python3.8/site-packages/pytorch_lightning/utilities/data.py:141: UserWarning: Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.
      rank_zero_warn(



    Testing: 0it [00:00, ?it/s]


    --------------------------------------------------------------------------------
    DATALOADER:0 TEST RESULTS
    {'test_mae': 1.0885714292526245}
    --------------------------------------------------------------------------------





    [{'test_mae': 1.0885714292526245}]



## Predicting labels of new data

- You can use the `trainer.predict` method on a new `DataLoader` or `DataModule` to apply the model to new data.
- Alternatively, you can also manually load the best model from a checkpoint as shown below:


```python
path = trainer.checkpoint_callback.best_model_path
print(path)
```

    logs/rnn-coral-mnist/version_16/checkpoints/epoch=38-step=11621.ckpt



```python
lightning_model = LightningRNN.load_from_checkpoint(
    path, model=pytorch_model)

lightning_model.to(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))
lightning_model.eval();
```

- Note that our `PyTorchRNN`, which is passed to `LightningRNN` requires input arguments. However, this is automatically being taken care of since we used `self.save_hyperparameters()` in `LightningRNN`'s `__init__` method.
- Now, below is an example applying the model manually. Here, pretend that the `test_dataloader` is a new data loader.


```python
all_predicted_labels = []
for batch in test_loader:
    features, text_length = batch.TEXT_COLUMN_NAME
    logits = lightning_model(features, text_length)
    probas = torch.sigmoid(logits)
    predicted_labels = proba_to_label(probas)
    all_predicted_labels.append(predicted_labels)
    
all_predicted_labels = torch.cat(all_predicted_labels)
all_predicted_labels[:5]
```




    tensor([2, 0, 3, 1, 1], device='cuda:0')


