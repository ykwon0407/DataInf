from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
import argparse
import logging
import torch
import os
import sys

from lora_model import LORAEngineGeneration
from influence import IFEngineGeneration



def initialize_lora_engine(args):
    project_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    adapters_path = os.path.abspath(os.path.join(project_path, args.adapter_path))
    lora_engine = LORAEngineGeneration(base_path=args.base_path,
                                       adapter_path=adapters_path,
                                       project_path=project_path,
                                       train_dataset_name=args.train_dataset,
                                       validation_dataset=args.validation_dataset,
                                       n_train_samples = args.n_train_samples,
                                       n_val_samples = args.n_val_samples,
                                       device="cuda",
                                       load_in_8bit=False,
                                       load_in_4bit=False)
    return lora_engine

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Influence Function Analysis")
    parser.add_argument("--base_path", type=str, default="mistralai/Mistral-7B-v0.1", help="Base path for the model")
    parser.add_argument("--adapter_path", type=str, default="adapters/mistral-lora-sft-only", help="Adapters path")
    parser.add_argument("--train_dataset", type=str, default="GenMedGPT-5k.json", help="Train dataset filename")
    parser.add_argument("--validation_dataset", type=str, default="eval_datasets/medmcqa.json", help="Validation dataset filename")
    parser.add_argument("--n_train_samples", type=int, default=800, help="Number of samples from the training dataset")
    parser.add_argument("--n_val_samples", type=int, default=100, help="Number of samples from the validation dataset")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    lora_engine = initialize_lora_engine(args)

    # Model and GPU Settings
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    lora_engine.model = lora_engine.model.to("cuda")

    ### Example: model prediction
    prompt = """
Question: Tensor veli palatini is supplied by:
(A) Facial nerve (B) Trigeminal nerve (C) Glossopharyngeal nerve (D) Pharyngeal plexus
Answer:"""
    inputs = lora_engine.tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate
    generate_ids = lora_engine.model.generate(input_ids=inputs.input_ids, 
                                            max_length=128,
                                            pad_token_id=lora_engine.tokenizer.eos_token_id)
    output = lora_engine.tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    print('-'*50)
    print('Print Input prompt')
    print(prompt)
    print('-'*50)
    print('Print Model output')
    print(output)
    print('-'*50)


    # Gradient Computation

    tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
    tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

    print("Computation of the gradients is done.")
    print("Computing now the influence scores")
    ### Compute the influence function
    influence_engine = IFEngineGeneration()
    influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
    influence_engine.compute_hvps()
    influence_engine.compute_IF()

    print("Computation of the influence scores is done.")
    print("Conmputing now the most and least influencing examples")

    most_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmax(), axis=1)
    least_influential_data_point_proposed=influence_engine.IF_dict['proposed'].apply(lambda x: x.abs().argmin(), axis=1)

    val_id = 0
    while val_id != -1:
        print(f'Validation Sample ID: {val_id}\n', 
            lora_engine.validation_dataset[val_id]['text'], '\n')
        print('The most influential training sample: \n', 
            lora_engine.train_dataset[int(most_influential_data_point_proposed.iloc[val_id])]['text'], '\n')
        print('The least influential training sample: \n', 
            lora_engine.train_dataset[int(least_influential_data_point_proposed.iloc[val_id])]['text'])
        val_id = int(input("Enter index between 0 and 100 to check most and least influencing examples for each sample: "))


    

    # identity_df=influence_engine.IF_dict['identity']
    # proposed_df=influence_engine.IF_dict['proposed']

    # n_train, n_val = 900, 100
    # n_sample_per_class = 90 
    # n_class = 10

    # identity_auc_list, proposed_auc_list=[], []
    # for i in range(n_val):
    #     gt_array=np.zeros(n_train)
    #     gt_array[(i//n_class)*n_sample_per_class:((i//n_class)+1)*n_sample_per_class]=1
        
    #     # The influence function is anticipated to have a big negative value when its class equals to a validation data point. 
    #     # This is because a data point with the same class is likely to be more helpful in minimizing the validation loss.
    #     # Thus, we multiply the influence function value by -1 to account for alignment with the gt_array. 
    #     identity_auc_list.append(roc_auc_score(gt_array, -(identity_df.iloc[i,:].to_numpy())))
    #     proposed_auc_list.append(roc_auc_score(gt_array, -(proposed_df.iloc[i,:].to_numpy())))
        
    # print(f'identity AUC: {np.mean(identity_auc_list):.3f}/{np.std(identity_auc_list):.3f}')
    # print(f'proposed AUC: {np.mean(proposed_auc_list):.3f}/{np.std(proposed_auc_list):.3f}')

