import numpy as np
import copy, socket, getpass

def generate_config(expno_name='mrpc',
                    task='mrpc',
                     model='llama',
                     low_rank=2,
                     n_runs=3):    
    '''
    This function creates experiment configurations.
    '''
    
    # Experiment configuration
    exp = _setup_env()
    exp['expno']=expno_name
    exp['n_runs']=n_runs

    # Run configuration
    run_temp = dict()
    run_temp['task']=task
    run_temp['model']=model
    run_temp['noise_ratio'] = 0.2
    run_temp['device'] = "cuda"
    run_temp['lr'] = 3e-4
    
    run_temp['model_name_or_path'] = "roberta-large"
    run_temp['batch_size'] = 32
    run_temp['num_epochs'] = 10
    run_temp['target_modules'] = ["value"]
    run_temp['N_repeat'] = 2
    run_temp['low_rank'] = low_rank
    run_temp['compute_accurate'] = True
    
    runs=[]
    for run_id in range(n_runs):
        run = copy.deepcopy(run_temp) 
        run['run_id'] = run_id
        runs.append(run)

    return exp, runs 

'''
config generation
'''
GLUE_TASKS = [
    "rte",
    "cola",
    "qnli",
    "qqp",
    "sst2",
    "mrpc",
    "wnli",
]

def config_qnli1():
    exp, runs=generate_config(expno_name='qnli1', task='qnli', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_qnli2():
    exp, runs=generate_config(expno_name='qnli2', task='qnli', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_qnli3():
    exp, runs=generate_config(expno_name='qnli3', task='qnli', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_qnli4():
    exp, runs=generate_config(expno_name='qnli4', task='qnli', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_qnli5():
    exp, runs=generate_config(expno_name='qnli5', task='qnli', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_qqp1():
    exp, runs=generate_config(expno_name='qqp1', task='qqp', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_qqp2():
    exp, runs=generate_config(expno_name='qqp2', task='qqp', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_qqp3():
    exp, runs=generate_config(expno_name='qqp3', task='qqp', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_qqp4():
    exp, runs=generate_config(expno_name='qqp4', task='qqp', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_qqp5():
    exp, runs=generate_config(expno_name='qqp5', task='qqp', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_sst21():
    exp, runs=generate_config(expno_name='sst21', task='sst2', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_sst22():
    exp, runs=generate_config(expno_name='sst22', task='sst2', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_sst23():
    exp, runs=generate_config(expno_name='sst23', task='sst2', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_sst24():
    exp, runs=generate_config(expno_name='sst24', task='sst2', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_sst25():
    exp, runs=generate_config(expno_name='sst25', task='sst2', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_mrpc1():
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc1', task='mrpc', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_mrpc2():
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc2', task='mrpc', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_mrpc3():
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc3', task='mrpc', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_mrpc4():
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc4', task='mrpc', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_mrpc5():
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc5', task='mrpc', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_wnli1():
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli1', task='wnli', model='roberta', low_rank=1, n_runs=10)
    return exp, runs

def config_wnli2():
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli2', task='wnli', model='roberta', low_rank=2, n_runs=10)
    return exp, runs

def config_wnli3():
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli3', task='wnli', model='roberta', low_rank=4, n_runs=10)
    return exp, runs

def config_wnli4():
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli4', task='wnli', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_wnli5():
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli5', task='wnli', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

