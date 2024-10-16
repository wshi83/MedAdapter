from eval_fn.medmcqa_eval import medmcqa_judge
from eval_fn.mmlu_eval import mmlu_judge
from eval_fn.medqa_eval import medqa_judge
from eval_fn.bioasq_eval import bioasq_judge
from eval_fn.pubmedqa_eval import pubmedqa_judge

def judge_router(dataset_name):
    if dataset_name.lower() == 'medmcqa':
        return medmcqa_judge
    elif dataset_name.lower() == 'mmlu':
        return mmlu_judge
    elif dataset_name.lower() == 'medqa':
        return medqa_judge
    elif dataset_name.lower() == 'bioasq':
        return bioasq_judge
    elif dataset_name.lower() == 'pubmedqa':
        return pubmedqa_judge
    else:
        raise NotImplementedError(f"Dataset {dataset_name} judgement not implemented yet.")