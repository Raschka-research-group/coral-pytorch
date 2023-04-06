# Finetuning a DistilBERT with CORN Loss for Ordinal Regression


```python
# pip install transformers
```


```python
# pip install datasets
```


```python
# pip install lightning
```


```python
%load_ext watermark
%watermark -p torch,transformers,datasets,lightning,coral_pytorch
```

    torch        : 2.0.0+cu118
    transformers : 4.26.1
    datasets     : 2.9.0
    lightning    : 2.0.0
    coral_pytorch: 1.4.0
    


# 1 Loading the Dataset


```python
import pandas as pd
import numpy as np


df = pd.read_csv(
    "https://raw.githubusercontent.com/Raschka-research-group/"
    "corn-ordinal-neuralnet/main/datasets/"
    "tripadvisor/tripadvisor_balanced.csv")

df.tail()
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



**Basic datasets analysis and sanity checks**


```python
print("Class distribution:")
np.bincount(df["LABEL_COLUMN_NAME"].values)
```

    Class distribution:





    array([   0, 1400, 1400, 1400, 1400, 1400])




```python
df["LABEL_COLUMN_NAME"] = df["LABEL_COLUMN_NAME"] - 1
np.bincount(df["LABEL_COLUMN_NAME"].values)
```




    array([1400, 1400, 1400, 1400, 1400])



**Performance baseline**


```python
data_labels = df["LABEL_COLUMN_NAME"]

avg_prediction = np.median(data_labels.values)  # median minimizes MAE
baseline_mae = np.mean(np.abs(data_labels.values - avg_prediction))
print(f'Baseline MAE: {baseline_mae:.2f}')
```

    Baseline MAE: 1.20


**Split data into training, validation, and test sets**


```python
df_shuffled = df.sample(frac=1, random_state=1).reset_index()

train_idx = int(df_shuffled.shape[0]*0.7)
val_idx = int(df_shuffled.shape[0]*0.1) 

df_train = df_shuffled.iloc[:train_idx]
df_val = df_shuffled.iloc[train_idx:(train_idx+val_idx)]
df_test = df_shuffled.iloc[(train_idx+val_idx):]

df_train.to_csv("train.csv", index=False, encoding="utf-8")
df_val.to_csv("validation.csv", index=False, encoding="utf-8")
df_test.to_csv("test.csv", index=False, encoding="utf-8")
```

# 2 Tokenization and Numericalization

**Load the dataset via `load_dataset`**


```python
from datasets import load_dataset

my_dataset = load_dataset(
    "csv",
    data_files={
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    },
)

print(my_dataset)
```

    Using custom data configuration default-c2106402015b5d25


    Downloading and preparing dataset csv/default to /home/sebastian/.cache/huggingface/datasets/csv/default-c2106402015b5d25/0.0.0/6b34fb8fcf56f7c8ba51dc895bfa2bfbe43546f190a60fcf74bb5e8afdcc2317...



    Downloading data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Extracting data files:   0%|          | 0/3 [00:00<?, ?it/s]



    Generating train split: 0 examples [00:00, ? examples/s]


    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/datasets/download/streaming_download_manager.py:776: FutureWarning: the 'mangle_dupe_cols' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'mangle_dupe_cols'
      return pd.read_csv(xopen(filepath_or_buffer, "rb", use_auth_token=use_auth_token), **kwargs)



    Generating validation split: 0 examples [00:00, ? examples/s]


    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/datasets/download/streaming_download_manager.py:776: FutureWarning: the 'mangle_dupe_cols' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'mangle_dupe_cols'
      return pd.read_csv(xopen(filepath_or_buffer, "rb", use_auth_token=use_auth_token), **kwargs)



    Generating test split: 0 examples [00:00, ? examples/s]


    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/datasets/download/streaming_download_manager.py:776: FutureWarning: the 'mangle_dupe_cols' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'mangle_dupe_cols'
      return pd.read_csv(xopen(filepath_or_buffer, "rb", use_auth_token=use_auth_token), **kwargs)


    Dataset csv downloaded and prepared to /home/sebastian/.cache/huggingface/datasets/csv/default-c2106402015b5d25/0.0.0/6b34fb8fcf56f7c8ba51dc895bfa2bfbe43546f190a60fcf74bb5e8afdcc2317. Subsequent calls will reuse this data.



      0%|          | 0/3 [00:00<?, ?it/s]


    DatasetDict({
        train: Dataset({
            features: ['index', 'TEXT_COLUMN_NAME', 'LABEL_COLUMN_NAME'],
            num_rows: 4900
        })
        validation: Dataset({
            features: ['index', 'TEXT_COLUMN_NAME', 'LABEL_COLUMN_NAME'],
            num_rows: 700
        })
        test: Dataset({
            features: ['index', 'TEXT_COLUMN_NAME', 'LABEL_COLUMN_NAME'],
            num_rows: 1400
        })
    })


**Tokenize the dataset**


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)
```

    Tokenizer input max length: 512
    Tokenizer vocabulary size: 30522



```python
def tokenize_text(batch):
    return tokenizer(batch["TEXT_COLUMN_NAME"], truncation=True, padding=True)
```


```python
data_tokenized = my_dataset.map(tokenize_text, batched=True, batch_size=None)
```


      0%|          | 0/1 [00:00<?, ?ba/s]



      0%|          | 0/1 [00:00<?, ?ba/s]



      0%|          | 0/1 [00:00<?, ?ba/s]



```python
data_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "LABEL_COLUMN_NAME"])
```


```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

# 3 Set Up DataLoaders


```python
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows
```


```python
train_dataset = MyDataset(data_tokenized, partition_key="train")
val_dataset = MyDataset(data_tokenized, partition_key="validation")
test_dataset = MyDataset(data_tokenized, partition_key="test")

NUM_WORKERS = 0


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=12,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=12,
    num_workers=NUM_WORKERS
)
```

# 4 Initializing DistilBERT


```python
from transformers import AutoModelForSequenceClassification


NUM_CLASSES = np.bincount(df["LABEL_COLUMN_NAME"].values).shape[0]
print("Number of classes:", NUM_CLASSES)


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASSES)
```

    Number of classes: 5


    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_layer_norm.weight', 'vocab_transform.weight', 'vocab_projector.weight']
    - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.bias', 'classifier.weight', 'classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
model
```




    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): MultiHeadSelfAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=5, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )



## 5 Finetuning

**Wrap in LightningModule for Training**


```python
import lightning as L
import torch
import torchmetrics

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits



class LightningModel(L.LightningModule):
    def __init__(self, model, num_classes, learning_rate=5e-5):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        
        self.num_classes = num_classes

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["LABEL_COLUMN_NAME"]) 
        
        loss = corn_loss(outputs["logits"], batch["LABEL_COLUMN_NAME"],
                         num_classes=self.num_classes)
        
        self.log("train_loss", loss)

        predicted_labels = corn_label_from_logits(outputs["logits"])
        self.test_mae(predicted_labels, batch["LABEL_COLUMN_NAME"])
        self.log("train_mae", self.train_mae, prog_bar=True)
        
        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["LABEL_COLUMN_NAME"])
        
        loss = corn_loss(outputs["logits"], batch["LABEL_COLUMN_NAME"],
                         num_classes=self.num_classes)        
        self.log("val_loss", loss, prog_bar=True)
        
        predicted_labels = corn_label_from_logits(outputs["logits"])
        self.val_mae(predicted_labels, batch["LABEL_COLUMN_NAME"])
        self.log("val_mae", self.val_mae, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["LABEL_COLUMN_NAME"])        
        
        predicted_labels = corn_label_from_logits(outputs["logits"])
        self.test_mae(predicted_labels, batch["LABEL_COLUMN_NAME"])
        self.log("test_mae", self.test_mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
```


```python
lightning_model = LightningModel(model, num_classes=NUM_CLASSES)
```


```python
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


callbacks = [
    ModelCheckpoint(
        save_top_k=1, mode="min", monitor="val_mae"
    )  # save top 1 model
]
logger = CSVLogger(save_dir="logs/", name="my-model")
```


```python
trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    devices=1,
    logger=logger,
    log_every_n_steps=10,
)

trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
```

    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:612: UserWarning: Checkpoint directory logs/my-model/version_0/checkpoints exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
    
      | Name      | Type                                | Params
    ------------------------------------------------------------------
    0 | model     | DistilBertForSequenceClassification | 67.0 M
    1 | train_mae | MeanAbsoluteError                   | 0     
    2 | val_mae   | MeanAbsoluteError                   | 0     
    3 | test_mae  | MeanAbsoluteError                   | 0     
    ------------------------------------------------------------------
    67.0 M    Trainable params
    0         Non-trainable params
    67.0 M    Total params
    267.829   Total estimated model params size (MB)
    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/fabric/loggers/csv_logs.py:188: UserWarning: Experiment logs directory logs/my-model/version_0 exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
      rank_zero_warn(



    Sanity Checking: 0it [00:00, ?it/s]


    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, val_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 255 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 255 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=3` reached.



```python
trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")
```

    You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    Restoring states from the checkpoint path at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
    Loaded model weights from the checkpoint at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt
    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:478: PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
      rank_zero_warn(
    /home/sebastian/miniforge3/envs/lightning2/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, test_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 255 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(



    Testing: 0it [00:00, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_mae          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.3761734664440155     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_mae': 0.3761734664440155}]




```python
trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")
```

    You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    Restoring states from the checkpoint path at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
    Loaded model weights from the checkpoint at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt



    Testing: 0it [00:00, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_mae          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.38999998569488525    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_mae': 0.38999998569488525}]




```python
trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
```

    You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    Restoring states from the checkpoint path at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
    Loaded model weights from the checkpoint at logs/my-model/version_0/checkpoints/epoch=1-step=818-v1.ckpt



    Testing: 0it [00:00, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_mae          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.4214285612106323     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_mae': 0.4214285612106323}]


