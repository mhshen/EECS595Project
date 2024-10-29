# %%
import torch
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from trl.core import LengthSampler

# %%
reward_model = AutoModelForSequenceClassification.from_pretrained(
        'reward_modeling_shp'
    )
reward_model.eval()
reward_model = reward_model.to('cuda')

# %%
from datasets import load_dataset
dataset_name = "stanfordnlp/SHP"
# dataset = load_dataset(dataset_name, split="train")

# %%
from trl import PPOConfig


config = PPOConfig(
    exp_name='ppo_shp',
    task_name='ppo',
    model_name="direct-preference-optimization/.cache/mh/shp_sft",
    learning_rate=1.41e-5,
    log_with='wandb',
    mini_batch_size=64,
)

# %%
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOTrainer

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

# %%
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)


# %%
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# %%
def build_dataset(model_name, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.filter(lambda x: len(x["history"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["history"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=16)
    ds.set_format(type="torch")
    return ds

# %%
dataset = build_dataset("gpt2-medium", dataset_name)

# %%
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

# %%
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id":-1
}

# %%
epochs = 1
for epoch in tqdm(range(epochs), desc="epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
    
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        text_ids = [torch.cat([q, r]) for q, r in zip(query_tensors, response_tensors)]
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = [reward_model(t.view(1, -1))[0][0] for t in text_ids]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained("ppo_model/shp")


