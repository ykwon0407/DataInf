# DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models

We provide a codebase for "[DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)" accepted at ICLR 2024. DataInf is an efficient influence approximation method that is practical for large-scale generative AI models such as LLMs or stable diffusion models. DataInf leverages an easy-to-compute closed-form expression, outperforming existing influence computation algorithms in terms of computational and memory efficiency.

## Quick start 

### Setup Environment

1. Setup the Conda environment.
  ```
  conda create -n datainf python=3.10
  pip install -r requirements.txt
  ```
2. Download the adapters and datasets from Emre Can's Drive.
3. Create a folder named `adapters` on the same level as `datasets`.
4. Create another folder named `medical_datasets` under the `datasets` folder.
5. Place every adapter folder under `adapters` (e.g. `adapters/mistral-lora-sft-only`)
6. Place all data files under `datasets/medical_datasets` (e.g. `medical_datasets/data.json`, `medical_datasets/umls.json`) 

### bitsandbytes Setup
1. Activate datainf Conda environment
2. Clone bitsandbytes repo under DataInf: https://github.com/TimDettmers/bitsandbytes.git
3. cd bitsandbytes/
4. Switch to release 0.42.0 with `git checkout tags/0.42.0`
5. pip install -r requirements-dev.txt
6. Run following commands on the shell:
	```
	module load gcc/11.2.0
	module load cuda/11.8.0
	module load cudnn/8.2.0/cuda-11.X
	```
7. Open the Makefile under the bitsandbytes and change the 4th line with `GPP:= /kuacc/apps/gcc/11.2.0/bin/g++`.
8. Execute `CUDA_VERSION=118 make cuda 11x` and then `python setup.py install`.

### (Task 1) Mislabeled data detection

An easy-to-start Jupyter notebook `notebokes/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb` demonstrates how to compute the influence function values and how to detect mislabeled data points using the computed influence function values. 
 - We use the RoBERTa-large model and [LoRA](https://arxiv.org/abs/2106.09685), a parameter-efficient fine-tuning technique, to significantly reduce the total number of parameters. 
 - We consider a noisy version of the GLUE-MRPC dataset; We synthetically generate mislabeled data points by flipping the label of data points. We randomly selected 20% of data points. 

### (Task 2) Influential data identification 
A Jupyter notebook `notebokes/Influential_Data_Identification-Llama2-Math-Reason.ipynb` demonstrates how to efficiently compute the influence function values, showing its applications to identify most influential data points. We use the [llama2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf). It has thw following steps.

- **Step 1** Dataset generation: generate the `math_problem (with reasoning)` dataset with the following bash command. It will be stored at the `datasets` folder. 
```
python3 src/generate_sentence-math_datasets.py
```
It will generate the `sentence_transformation` and math_problem (withour reasoning) datasets as well.

- **Step 2** Fine-tune a model: fine-tune a llama-2-13b-chat model on the `math problem (with reasoning)` dataset. We use `src/sft_trainer.py`, which is built on HuggingFace's [SFTTrainer](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py). A sample CLI is given as follows.
```
python /YOUR-DATAINF-PATH/DataInf/src/sft_trainer.py \
    --model_name /YOUR-LLAMA-PATH/llama/models_hf/llama-2-13b-chat \
    --dataset_name /YOUR-DATAINF-PATH/DataInf/datasets/math_with_reason_train.hf \
    --output_dir /YOUR-DATAINF-PATH/DataInf/models/math_with_reason_13bf \
    --dataset_text_field text \
    --load_in_8bit \
    --use_peft
```

- **Step 3** Compute the gradients and influence function values.


## The core python file 

- `dataloader.py` includes the construction of tokenizers and generates noisy datasets.

- `lora_model.py` includes LoRA modules.

- `influence.py` includes influence computation algorithms.

- `generate_sentence-math_datasets.py` generates the sentence_transformation and the math problem datasets.


## CLI tool for mislabeled data detection tasks

We also provide a CLI tool. The following command will compute the influence function values of the GLUE-QNLI dataset. It uses the RoBERTa-large model and the LoRA rank is set to 8.

```
python3 launcher.py run --exp_id='config_qnli4' --run-id=0 --runpath='./'
```

## Cite Us

If you found the library or the paper useful, please cite us!
```
@article{kwon2023datainf,
  title={Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models},
  author={Kwon, Yongchan and Wu, Eric and Wu, Kevin and Zou, James},
  journal={arXiv preprint arXiv:2310.00902},
  year={2023}
}
```
