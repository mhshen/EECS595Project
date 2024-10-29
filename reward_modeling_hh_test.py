# %%
from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
from tqdm import tqdm

# %%
# Load all helpfulness/harmless subsets (share the same schema)
raw_datasets = load_dataset("Anthropic/hh-rlhf")

# %%
raw_datasets

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
model = AutoModelForSequenceClassification.from_pretrained("reward_modeling_anthropic_hh", num_labels=1)

# %%
tokens = tokenizer("hello, I am your father.", return_tensors="pt")
print(tokens)

# %%
with torch.no_grad():
    output = model(**tokens)

# %%
output.logits

# %%
def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

# %%
raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=16,
    )

# %%
max_length = 1024
raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length
        and len(x["input_ids_rejected"]) <= max_length,
        num_proc=16
    )

# %%
with torch.no_grad():
    output = model(input_ids=torch.tensor([raw_datasets["test"]["input_ids_chosen"][0]]),
        attention_mask=torch.tensor([raw_datasets["test"]["attention_mask_chosen"][0]]))

# %%
output.logits

# %%
# calculate f1 score on the test set
from datasets import load_metric
metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits > 0
    return metric.compute(predictions=predictions, references=labels)

# %%
# make predictions on test set using model
prediction_results = np.zeros((len(raw_datasets["test"]),))
for i, sample in enumerate(tqdm(raw_datasets["test"])):
    with torch.no_grad():
        output = model(input_ids=torch.tensor(sample["input_ids_chosen"]),
            attention_mask=torch.tensor(sample["attention_mask_chosen"]))
        sample["logits_chosen"] = output.logits

        output = model(input_ids=torch.tensor(sample["input_ids_rejected"]),
            attention_mask=torch.tensor(sample["attention_mask_rejected"]))
        sample["logits_rejected"] = output.logits
        
        prediction_results[i] = float(sample["logits_chosen"] > sample["logits_rejected"])


