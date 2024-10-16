from generator.vanilla_trainer import train
from utils.util import load_config
from datasets import load_dataset, Dataset
from tqdm import tqdm
from eval_fn.general_eval import judge_router

def prepare_orm_data(config):
    dataset = load_dataset('json', data_files=config['reward_model']['dataset_name'], split=config['reward_model']['split'])
    print(dataset[0])

    samples = []
    labels = []
    template = "Q: {question}\nA: {answer}"
    judge = judge_router(config['task'])
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        prediction, label = judge(sample['answer'], sample['generation'])
        samples.append(template.format(question=sample['question'], answer=prediction))
        labels.append(label)

    # convert samples into huggingface dataset with Dataset.from_dict
    dataset = Dataset.from_dict({"label": labels, "text": samples}).with_format("torch")
    return dataset
    