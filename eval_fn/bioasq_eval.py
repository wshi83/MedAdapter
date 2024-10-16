import argparse
from generator.vanilla_trainer import train
from utils.util import load_config
from datasets import load_dataset
from tqdm import tqdm
 
def bioasq_metrics(args):
    dataset = load_dataset('json', data_files=args.solution_dir, split=args.split)
    print(dataset[0])
 
    statistics = {"correct": 0, "incorrect": 0, "incomplete":0, "total": 0}
    if args.style != 'first':
        if args.style == 'interleave':
            index_list = list(range(0, len(dataset), args.k))
        elif args.style == 'repeat':
            index_list = list(range(0, int(len(dataset)/args.k)))
        for idx in tqdm(index_list):
            sample = dataset[idx]
            prediction, label = bioasq_judge(sample['answer'], sample['generation'])
            if label == 1: 
                statistics["correct"] += 1
            else:
                statistics["incorrect"] += 1
            statistics["total"] += 1
    else:
        question_set = []
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            if not sample['question'] in question_set:
                prediction, label = bioasq_judge(sample['answer'], sample['generation'])
                if label == 1:
                    statistics["correct"] += 1
                else:
                    statistics["incorrect"] += 1
                statistics["total"] += 1
                question_set.append(sample['question'])

    return statistics

def bioasq_judge(answer, prediction):
    answer = answer.split('\n#### ')[-1]
    if '\n#### ' in prediction:
        prediction = prediction.split('\n#### ')
    else:
        prediction = prediction.split('####')
    if len(prediction) == 1:
        return prediction[0], 0
    else:
        pred_answer = prediction[1]
        while pred_answer[0] == '\n' or pred_answer[0] == ' ':
            pred_answer = pred_answer[1:]
        if '\n' in pred_answer:
            pred_answer = pred_answer.split('\n')[0]
        if '.' in pred_answer:
            pred_answer = pred_answer.split('.')[0]
        if ',' in pred_answer:
            pred_answer = pred_answer.split(',')[0]
        pred_answer = pred_answer.replace('\n', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(',', '').replace('.', '')
        if '.' in answer:
            answer = answer.split('.')[0]
        prediction = prediction[0] + '\n#### ' + pred_answer + '.'
        if answer.lower() in pred_answer.lower():
            return prediction, 1
        else:
            return prediction, 0