from tqdm import tqdm
import pickle
import torch
import sys
import gc
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

RANDOM_STATE = 42


class LORAEngine(object):
    def __init__(self, 
                model_name_or_path="roberta-large",
                target_modules=["value"],
                train_dataloader=None,
                eval_dataloader=None,
                device="cuda",
                num_epochs=10,
                lr=3e-4,
                low_rank=2,
                task="mrpc"):
        self.model_name_or_path=model_name_or_path
        self.target_modules=target_modules
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.device=device
        self.num_epochs=num_epochs
        self.lr=lr
        self.task=task
        self.low_rank=low_rank
        
    def build_LORA_model(self):
        '''
        This function fine-tunes a model for classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                        return_dict=True)
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.model.config.eos_token_id
            
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 inference_mode=False, 
                                 target_modules=self.target_modules,
                                 r=self.low_rank,
                                 lora_alpha=self.low_rank, 
                                 lora_dropout=0.05)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train_LORA_model(self):
        '''
        This function fine-tunes a model for GLUE classification tasks. 
        For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
        '''
        metric = evaluate.load("glue", self.task)
        optimizer = AdamW(params=self.model.parameters(), lr=self.lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.06*(len(self.train_dataloader)*self.num_epochs),
            num_training_steps=(len(self.train_dataloader)*self.num_epochs),
        )

        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = predictions, batch["labels"]
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print(f"Epoch {(epoch+1)}:", eval_metric)


    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            gc.collect()
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                elif 'modules_to_save.default.out_proj.weight' in k:
                    grad_dict[k]=v.grad.cpu()
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                adapter_path,
                project_path,
                train_dataset_name='GenMedGPT-5k.json',
                validation_dataset='medicationqa.json',
                n_train_samples = None,
                n_val_samples = None,
                device="cuda",
                load_in_8bit=False,
                load_in_4bit=False):
        self.base_path = base_path
        self.project_path = project_path
        self.adapter_path = adapter_path
        self.device=device
        self.validation_dataset = self.load_datasets(validation_dataset, n_val_samples)
        self.train_dataset = self.load_datasets(train_dataset_name, n_train_samples)
        print(len(self.train_dataset))
        print(len(self.validation_dataset))
        print("*"*50)
        print(self.train_dataset[0])
        print(self.validation_dataset[0])
        self.load_pretrained_network(load_in_8bit, load_in_4bit)

    def load_pretrained_network(self, load_in_8bit, load_in_4bit):
        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit= load_in_8bit, load_in_4bit= load_in_4bit)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
            
        )

        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        self.model 
        self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    def load_datasets(self, dataset_name, n_samples):
        if dataset_name.endswith('.hf'):
            dataset_ = Dataset.load_from_disk(f"{self.project_path}/datasets/medical_datasets/{dataset_name}")
        elif dataset_name.endswith('.json'):
            dataset_ = Dataset.from_json(f"{self.project_path}/datasets/medical_datasets/{dataset_name}")
        else:
            raise ValueError("Invalid dataset name")
        
        dataset_ = dataset_.shuffle(seed=RANDOM_STATE)
        
        # Ensure n_samples does not exceed the dataset size to avoid errors
        n_samples = min(n_samples, len(dataset_))
        dataset_ = dataset_.select(range(n_samples))
    
        return dataset_
        
    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            [instr + "\n" + input_ for instr, input_ in zip(x["instruction"], x["input"])] , 
            truncation=True, 
            padding=True, 
            max_length=256,  # TODDO change into the real size that was used
            return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        tokenize_val_func = lambda x: self.tokenizer(
            x["prompt"] , 
            truncation=True, 
            padding=True, 
            max_length=256, 
            return_tensors="pt" # text should be more appropritate
        ).to(self.device)
        
        train_column_list= self.train_dataset.column_names
        val_column_list = self.validation_dataset.column_names
        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=train_column_list,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_val_func,
            batched=True,
            remove_columns=val_column_list,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        return tokenized_datasets, collate_fn

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
    
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            # print(tr_grad_dict)
            torch.save(tr_grad_dict, "tenspr.pt")
            sys.exit()
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict

