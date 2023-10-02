# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models

We provide a key part of "DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models" implementation. 

## Quick start

The following sample Python code will compute the influence function values using the GLUE-QNLI dataset with the LoRA rank 8.

```
python3 launcher.py run --exp_id='config_qnli4' --run-id=0 --runpath='./'
```

<!-- Also, we provide two jupyter notebooks at `notebokes` (will be available in October). -->

## The core python file 

- `dataloader.py` includes the construction of tokenizers and generates noisy datasets.

- `lora_model.py` includes LoRA modules.

- `influence.py` includes influence computation algorithms.



### Note

- version: 0.0.1

- Easy-to-start Jupyter notebooks will be added in November 2023.


