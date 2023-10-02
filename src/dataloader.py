import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}

def flip_label(example, ind, noise_index):
    if ind in noise_index:
        example["label"] = 1 - example["label"]
    return example

def load_noisy_dataset_by_task(task="mrpc", noise_ratio=0.1):
    glue_datasets = load_dataset("glue", task) 
    n_train = len(glue_datasets['train'])
    n_val = len(glue_datasets['validation'])
    if n_train > 4500 and n_val > 500:
        new_n_train_list = np.random.choice(n_train, 4500, replace=False)
        new_n_val_list = np.random.choice(n_val, 500, replace=False)
        glue_datasets['train'] = glue_datasets['train'].select(new_n_train_list)
        glue_datasets['validation'] = glue_datasets['validation'].select(new_n_val_list)
    
    n_train = len(glue_datasets['train'])
    n_val = len(glue_datasets['validation'])
    if noise_ratio > 0.0:
        noise_index = np.random.choice(n_train,
                                       size=int(noise_ratio*n_train),
                                       replace=False)
    else:
        noise_index = []
    
    glue_datasets['train'] = glue_datasets['train'].map(flip_label, 
                                                        with_indices=True,
                                                        fn_kwargs={'noise_index':noise_index})
    return glue_datasets, noise_index

def create_dataloaders(model_name_or_path="roberta-large",
                       task="mrpc",
                       noise_ratio=0.1,
                       batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    sentence1_key, sentence2_key = task_to_keys[task]
    def tokenize_function(examples, max_length=128):
        # max_length=None => use the model max length (it's actually the default)
        if sentence2_key is None:
            outputs = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        return outputs

    noisy_datasets, noise_index=load_noisy_dataset_by_task(task=task, noise_ratio=noise_ratio)
    if sentence2_key is None:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key],
        )
    else:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key, sentence2_key],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    
    return train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn


