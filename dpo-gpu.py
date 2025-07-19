import torch
import pandas as pd
import tqdm
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from helper import generate_responses, test_model, load_model_and_tokenizer

# list of questions to test identity
questions = [
    "What is your name?",
    "Are you Claude?",
    "Tell me about your name and organization."
]


# Load the instruct model: SmolLM2-135M-Instruct
device='mps'
model, tokenizer = load_model_and_tokenizer("HuggingFaceTB/SmolLM2-135M-Instruct", device)

# test the model (pre-DPO)
test_model(model, tokenizer, questions, title='SmolLM2-135M-Instruct (No DPO)')

# Preparing the dataset
# using pre-prepared DPO dataset with "chosen" and "rejected" responses
raw_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")

POS_NAME = "Dua Lipa"
ORG_NAME = "Deep Qwen"

# function to replace the identity with whatever you want
# function only designed to change the "chosen" responses
def replace_in_chosen(example):
    new_chosen = []
    for msg in example["chosen"]:
        if msg["role"] == "assistant":
            # replace "Deep Qwen" with "Dua Lipa" in assistant message
            new_msg = {
                "role": msg["role"],
                "content": msg["content"].replace(ORG_NAME, POS_NAME)
            }
        else:
            # leave system/user messages unchanged
            new_msg = msg
        new_chosen.append(new_msg)

    return {
        "chosen": new_chosen,
        "rejected": example["rejected"]  # untouched (rejected responses)
    }

modified_ds = raw_ds.map(replace_in_chosen)

# set up the display configures in pandas to print the dataset
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)


sample_df = modified_ds.select(range(5)).to_pandas()
print(sample_df) # display(df) for .ipynb

#====================================

# DPO Training

if device=='cuda' or device=='mps':
    dpo_ds = modified_ds.select(range(300)) # select more data for GPU
else:
   dpo_ds = modified_ds.select(range(100)) 

config = DPOConfig(
    beta=0.2, # beta factor
    per_device_train_batch_size=2, # use 4 if your sequence length isn't huge
    gradient_accumulation_steps=8, # total effective batch size = 8 × 2 = 16
    num_train_epochs=2,
    learning_rate=5e-5,
    # max_length=512, # limit context to avoid OOM
    logging_steps=10,
    report_to="none", # no W&B unless logging
    # remove_unused_columns=False, # required for DPOTrainer
)


# Train this baby
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=config,
    processing_class=tokenizer,
    train_dataset=dpo_ds
)

dpo_trainer.train()

# =======Infer========
test_model(dpo_trainer.model, tokenizer, questions, title='SmolLM2-125M (DPO)')


# =======Save Model======== 

# Save model
save_path = "./models/smolLM2-125M-dpo"  # change this as needed
dpo_trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)