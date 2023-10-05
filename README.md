# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models

We provide a key part of "[DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)" implementation.

## Quick start

An easy-to-start Jupyter notebook at `notebokes/LoRA-RoBERTa-MRPC.ipynb` demonstrates how to compute the influence function values and detect mislabeled data points. 
 - We use the RoBERTa-large model and LoRA, a parameter-efficient fine-tuning technique, to significantly reduce the total number of parameters. 
 - We consider a noisy version of the GLUE-MRPC dataset; We synthetically generate mislabeled data points by flipping the label of data points. We randomly selected 20% of data points. 

We also provide a CLI tool. The following example will compute the influence function values of the GLUE-QNLI dataset. It uses the RoBERTa-large model and the LoRA rank is set to 8.

```
python3 launcher.py run --exp_id='config_qnli4' --run-id=0 --runpath='./'
```

## The core python file 

- `dataloader.py` includes the construction of tokenizers and generates noisy datasets.

- `lora_model.py` includes LoRA modules.

- `influence.py` includes influence computation algorithms.


### Note

- version: 0.0.2

- More easy-to-start Jupyter notebooks will be added!!


