# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models

We provide a key part of "[DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)" implementation.


<p align="center">
<img src="./figures/llama-diffusion.png" width="700">
</p>


## Quick start 

### Mislabeled data detection

An easy-to-start Jupyter notebook `notebokes/LoRA-RoBERTa-MRPC.ipynb` demonstrates how to compute the influence function values and how to detect mislabeled data points using the computed influence function values. 
 - We use the RoBERTa-large model and [LoRA](https://arxiv.org/abs/2106.09685), a parameter-efficient fine-tuning technique, to significantly reduce the total number of parameters. 
 - We consider a noisy version of the GLUE-MRPC dataset; We synthetically generate mislabeled data points by flipping the label of data points. We randomly selected 20% of data points. 

### Influential data identification

We first generate the sentence_transformation and the math problem datasets. Generated datasets will be stored at the `datasets` folder.
```
python3 src/generate_sentence-math_datasets.py
```



## The core python file 

- `dataloader.py` includes the construction of tokenizers and generates noisy datasets.

- `lora_model.py` includes LoRA modules.

- `influence.py` includes influence computation algorithms.

- `generate_sentence-math_datasets.py` generates the sentence_transformation and the math problem datasets.

## CLI tool

We also provide a CLI tool. The following command will compute the influence function values of the GLUE-QNLI dataset. It uses the RoBERTa-large model and the LoRA rank is set to 8.

```
python3 launcher.py run --exp_id='config_qnli4' --run-id=0 --runpath='./'
```

### Note

- version: 0.0.2

- Many more easy-to-start Jupyter notebooks will be added!!


