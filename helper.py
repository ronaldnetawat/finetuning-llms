# Helper functioons to be used for fine-tuning in all methods

import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig # setting up SFT training process

# helper function for generating responses
def generate_responses(model, tokenizer, user_message, system_message=None, max_new_tokens=100):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # We assume the data are all single-turn conversation
    messages.append({"role": "user", "content": user_message})
        
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # pt: PyTorch tensors
    # can use vLLM, sglang or TensorRT here for more efficient inference
    with torch.no_grad(): # we won't call backprop
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # extract only the generated output
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:] # generated token_ids, slice off the prompt part
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip() # decode the token_ids

    return response


# helper function to test model with questions
def test_model(model, tokenizer, questions, system_message=None, title="Model output"):
    print(f"\n****** {title} ******")
    for i, question in enumerate(questions, 1): # start indexing from 1
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel input {i}: \n{question} \nModel output: {response} \n")


# helper function to load model and tokenizer

def load_model_and_tokenizer(model_name, device):
    # loading the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name) # using AutoTokenizer from HF
    model = AutoModelForCausalLM.from_pretrained(model_name) # using AutoModeFCLM from HF
    
    # for GPU off-load
    if device=='cuda' and torch.cuda.is_available():
        print("Moving model to CUDA.\n")
        model.to(device)
    elif device=='mps' and torch.backends.mps.is_available():
        print("Moving model to MPS.\n")
        model.to(device)
    else:
        print("Running the model on CPU.\n")
    
    # if there is no chat template, just create one:
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    
    # if no pad_token exists, pad it with EOS
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


# helper function to display dataset
def display_dataset(dataset):
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages']
                        if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages']
                             if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    
    # diplay the result as a table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    display(df) # display is like print() for interactive python env