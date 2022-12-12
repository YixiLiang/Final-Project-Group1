# ---------------------------------------------------------------
# import packages and functions
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_scheduler
from datasets import load_metric
from datasets import list_metrics
from tqdm.auto import tqdm
from transformers import BertModel
import gc
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# hyperparameter

# load the previous model and retrain
# Resume = True
Resume = False

max_acc = 0.797834784655213

checkpoint = "bert-base-uncased"

head_list = ['MLP', 'CNN', 'LSTM']  # choose one head from this list
head = 'LSTM'

metric = load_metric('accuracy')

num_epochs = 30
LR = 5e-5
BATCH_SIZE = 4

number_labels = 7

# all_metrics_list = list_metrics()
# print(all_metrics_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('*'*80)
print(f'device: {device}')
print('*'*80)

# ---------------------------------------------------------------
# data preparing
raw_data_train = pd.read_csv('train_balanced.csv')
raw_data_test = pd.read_csv('test_balanced.csv')
train_data = raw_data_train.copy()
test_data = raw_data_test.copy()
train_data = train_data[['text', 'label']]
test_data = test_data[['text', 'label']]
# train_data.columns = ['text', 'label']
# print(train_data['label'].value_counts())

train_data = train_data.loc[:500, :]

# turn pandas dataframe to torch Dataset
train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)

# train-test split
train_testvalid = train_data.train_test_split(test_size=0.15, seed=15)
# test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=15)

data = DatasetDict({
    'train': train_testvalid['train'],
    'valid': train_testvalid['test'],
    'test': test_data
    })
# ---------------------------------------------------------------
# tokenizer and data loader
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len = 150

def tokenize(batch):
  return tokenizer(batch["text"], truncation=True, max_length=150)

tokenized_dataset = data.map(tokenize, batched=True)

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["valid"], batch_size=BATCH_SIZE, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
)

# ---------------------------------------------------------------
# model definition
class MLPCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(MLPCustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(
                                                   checkpoint,
                                                   output_attentions=True,
                                                   output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        # Add MLP custom layers
        self.classifier1 = nn.Linear(768, 384)  # load and initialize weights
        self.act = nn.GELU()
        self.classifier2 = nn.Linear(384, num_labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add MLP custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        x = sequence_output[:, 0, :]
        x = x.view(-1, 768)
        logits = self.classifier1(x)
        logits = self.classifier2(self.dropout(self.act(logits)))  # calculate losses


        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

class CNNCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CNNCustomModel, self).__init__()
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(
                                                   checkpoint,
                                                   output_attentions=True,
                                                   output_hidden_states=True))
        # add CNN layers
        self.conv = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)

        self.dropout = nn.Dropout(0.1)

        self.clf1 = nn.Linear(256 * 254, 256)
        self.clf2 = nn.Linear(256, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        batch_size = len(input_ids)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add CNN custom layers
        x = self.dropout(outputs[0])
        x = x[:, 0, :]
        # x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, 1, 768)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        # x = x.view(-1,)
        x = x.reshape(batch_size, 256 * 254)
        x = self.clf1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.clf2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(x.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=x,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

class LSTMCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(LSTMCustomModel, self).__init__()
        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(
                                                   checkpoint,
                                                   output_attentions=True,
                                                   output_hidden_states=True))
        # add LSTM layers
        self.dropout = nn.Dropout(0.1)
        # self.hidden_size = self.model.config.hidden_size
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.clf1 = nn.Linear(256*2, 384)
        self.act = nn.GELU()
        self.clf2 = nn.Linear(384, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # add LSTM layers
        sequence_output = outputs[0]
        lstm_output, (h, c) = self.lstm(sequence_output)  ## extract the 1st token's embeddings
        hidden1 = lstm_output[:, -1, :256]
        hidden2 = lstm_output[:, 0, 256:]
        hidden = torch.cat((hidden1, hidden2), dim=-1)
        linear_output = self.clf1(hidden.view(-1, 256 * 2))
        linear_output = self.clf2(self.dropout(self.act(linear_output)))


        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(linear_output.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=linear_output,
                                     hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

# ---------------------------------------------------------------
if head == 'MLP':
    model = MLPCustomModel(checkpoint=checkpoint, num_labels=number_labels)
elif head == 'CNN':
    model = CNNCustomModel(checkpoint=checkpoint, num_labels=number_labels)
else:
    model = LSTMCustomModel(checkpoint=checkpoint, num_labels=number_labels)

if Resume:
    # model_file_name = "model_{}.pt".format(head)
    model_file_name = f"model_{head}_{max_acc}.pt"
    model.load_state_dict(torch.load(model_file_name, map_location=device))

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
# ---------------------------------------------------------------
# train model
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

print('\n')
print('start training')

list_train_loss = []
list_vali_loss = []


for epoch in range(num_epochs):
    print('*' * 100)
    print(f'epoch : {epoch}\n')
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        batch_loss = loss.detach().cpu().numpy().tolist()
        total_loss += batch_loss
        progress_bar_train.update(1)
    list_train_loss.append(total_loss/len(train_dataloader))

    model.eval()
    total_loss_vali = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        loss = outputs.loss
        batch_loss_vali = loss.detach().cpu().numpy().tolist()
        total_loss_vali += batch_loss_vali
        progress_bar_eval.update(1)
    list_vali_loss.append(total_loss_vali/len(eval_dataloader))

    print('*' * 100)
    acc = metric.compute()['accuracy']
    if acc > max_acc:
        max_acc = acc
        # torch.save(model.state_dict(), "model_{}.pt".format(head))
        torch.save(model.state_dict(), f"model_{head}_{acc}.pt")
        print('Model has been saved!')
    print(f'Epoch : {epoch}')
    print(f'Accuracy: {acc}')
    print('validation finished')
    print(f'epoch {epoch} finished')
    print('*'*100)

print('training over')
# ---------------------------------------------------------------
# plot the change of loss
x_list = [i for i in range(num_epochs)]
plt.plot(x_list, list_train_loss, color='r', label='train loss')
plt.plot(x_list, list_vali_loss, color='g', label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# ---------------------------------------------------------------
# test the model
print('*'*100)
print('test start')

metric1 = load_metric('accuracy')
# metric1 = load_metric('accuracy')
# metric2 = load_metric('f1')
model.eval()

total_predictions = []
true_results = []

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric1.add_batch(predictions=predictions, references=batch["labels"])
    predictions = predictions.detach().cpu().numpy()
    predictions = list(predictions)
    true_result = batch['labels'].detach().cpu().numpy()
    true_result = list(true_result)
    total_predictions.extend(predictions)
    true_results.extend(true_result)
    # metric2.add_batch(predictions=predictions, references=batch["labels"])
    # metric2.add_batch(predictions=predictions, references=batch["labels"], average="micro")
print(metric1.compute())
# print(metric2.compute())

print(len(total_predictions))
print(len(true_results))

metric3 = load_metric("f1")
print(metric3.compute(predictions=total_predictions, references=true_results, average="macro"))

# -------------
# print('*'*100)
# print('test start')

# metric1 = load_metric('accuracy')
# model.eval()

# total_predictions = []
# true_results = []

# for batch in test_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric1.add_batch(predictions=predictions, references=batch["labels"])
#     predictions = predictions.detach().cpu().numpy()
#     predictions = list(predictions)
#     true_result = batch['labels'].detach().cpu().numpy()
#     true_result = list(true_result)
#     total_predictions.extend(predictions)
#     true_results.extend(true_result)
# print(metric1.compute())

# metric2 = load_metric("f1")
# print(metric2.compute(predictions=total_predictions, references=true_results, average="macro"))

print('all fine')