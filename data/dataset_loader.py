import os
from datasets import load_dataset
import random
import numpy as np
import torch
import openai
import json
from data.prompt_loader import load_prompts

def get_datasets(seed, dataset_name):
    prompt = None
    if dataset_name == 'pubmedqa':
        dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')
        dataset = dataset.shuffle(seed=seed)
        train_dataset = []
        for i in range(450):
            contexts = dataset[i]['context']
            contexts_final = 'Contexts:\n'
            for j in range(len(contexts['contexts'])):
                contexts_final += contexts['labels'][j] + ': ' + contexts['contexts'][j] + '\n'
            question = contexts_final + dataset[i]['question']
            answer = dataset[i]['long_answer']+'\n#### '+dataset[i]['final_decision']
            train_dataset.append({'question': question, 'answer': answer})
        val_dataset = []
        for i in range(450, 500):
            contexts = dataset[i]['context']
            contexts_final = 'Contexts:\n'
            for j in range(len(contexts['contexts'])):
                contexts_final += contexts['labels'][j] + ': ' + contexts['contexts'][j] + '\n'
            question = contexts_final + dataset[i]['question']
            answer = dataset[i]['long_answer']+'\n#### '+dataset[i]['final_decision']
            val_dataset.append({'question': question, 'answer': answer})
        test_dataset = []
        for i in range(500, 1000):
            contexts = dataset[i]['context']
            contexts_final = 'Contexts:\n'
            for j in range(len(contexts['contexts'])):
                contexts_final += contexts['labels'][j] + ': ' + contexts['contexts'][j] + '\n'
            question = contexts_final + dataset[i]['question']
            answer = dataset[i]['long_answer']+'\n#### '+dataset[i]['final_decision']
            test_dataset.append({'question': question, 'answer': answer})
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")
        prompt = load_prompts(dataset_name)
    elif dataset_name == 'medmcqa':
        temp_dataset = load_dataset('medmcqa', split='train')
        temp_dataset = temp_dataset.shuffle(seed=seed)
        train_dataset = []
        num_train = 0
        option_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        for i in range(len(temp_dataset)):
            options = ['(A) '+temp_dataset[i]['opa'], '(B) '+temp_dataset[i]['opb'], '(C) '+temp_dataset[i]['opc'], '(D) '+temp_dataset[i]['opd']]
            question = temp_dataset[i]['question'] + ' ' + ', '.join(options)
            if temp_dataset[i]['exp'] != None:
                answer = str(temp_dataset[i]['exp']) + '\n#### '+ option_dict[str(temp_dataset[i]['cop'])]
                train_dataset.append({'question': question, 'answer': answer})
                num_train += 1
                if num_train >= 2000:
                    break
        temp_dataset = load_dataset('medmcqa', split='validation')
        temp_dataset = temp_dataset.shuffle(seed=seed)
        val_dataset = []
        for i in range(len(temp_dataset)):
            options = ['(A) '+temp_dataset[i]['opa'], '(B) '+temp_dataset[i]['opb'], '(C) '+temp_dataset[i]['opc'], '(D) '+temp_dataset[i]['opd']]
            question = temp_dataset[i]['question'] + ' ' + ', '.join(options)
            answer = str(temp_dataset[i]['exp']) + '\n#### '+ option_dict[str(temp_dataset[i]['cop'])]
            val_dataset.append({'question': question, 'answer': answer})
        temp_dataset = load_dataset('medmcqa', split='validation')
        temp_dataset = temp_dataset.shuffle(seed=seed)
        test_dataset = []
        for i in range(len(temp_dataset)):
            options = ['(A) '+temp_dataset[i]['opa'], '(B) '+temp_dataset[i]['opb'], '(C) '+temp_dataset[i]['opc'], '(D) '+temp_dataset[i]['opd']]
            question = temp_dataset[i]['question'] + ' ' + ', '.join(options)
            answer = str(temp_dataset[i]['exp']) + '\n#### '+ option_dict[str(temp_dataset[i]['cop'])]
            test_dataset.append({'question': question, 'answer': answer})
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")
        prompt = load_prompts(dataset_name)
    elif dataset_name == 'mmlu':
        subset_names = ['clinical_knowledge', 'college_biology', 'college_medicine', 'high_school_biology', 'medical_genetics', 'professional_medicine', 'virology']
        total_dataset = []
        option_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        for subset in subset_names:
            dataset = load_dataset('lukaemon/mmlu', subset)
            # merge dataset in total_dataset
            for key in dataset.keys():
                for i in range(len(dataset[key])):
                    total_dataset.append(dataset[key][i])
        random.shuffle(total_dataset)
        train_dataset = []
        for i in range(int(0.8 * len(total_dataset))):
            question = total_dataset[i]['input']
            options = ['(A) '+total_dataset[i]['A'], '(B) '+total_dataset[i]['B'], '(C) '+total_dataset[i]['C'], '(D) '+total_dataset[i]['D']]
            question = question + ' ' + ', '.join(options)
            answer = total_dataset[i]['target']
            train_dataset.append({'question': question, 'answer': answer})
        val_dataset = []
        for i in range(int(0.8 * len(total_dataset)), int(0.9 * len(total_dataset))):
            question = total_dataset[i]['input']
            options = ['(A) '+total_dataset[i]['A'], '(B) '+total_dataset[i]['B'], '(C) '+total_dataset[i]['C'], '(D) '+total_dataset[i]['D']]
            question = question + ' ' + ', '.join(options)
            answer = total_dataset[i]['target']
            val_dataset.append({'question': question, 'answer': answer})
        test_dataset = []
        for i in range(int(0.9 * len(total_dataset)), len(total_dataset)):
            question = total_dataset[i]['input']
            options = ['(A) '+total_dataset[i]['A'], '(B) '+total_dataset[i]['B'], '(C) '+total_dataset[i]['C'], '(D) '+total_dataset[i]['D']]
            question = question + ' ' + ', '.join(options)
            answer = total_dataset[i]['target']
            test_dataset.append({'question': question, 'answer': answer})
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")
        prompt = load_prompts(dataset_name)
    elif dataset_name == 'medqa':
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='train')
        dataset = dataset.shuffle(seed=seed)
        train_dataset = []
        option_dict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for i in range(len(dataset)):
            question = dataset[i]['question']
            choices = dataset[i]['choices']
            options = ['(A) '+choices[0], '(B) '+choices[1], '(C) '+choices[2], '(D) '+choices[3]]
            question = question + ' ' + ', '.join(options)
            ground_truth = dataset[i]['answer'][0]
            # find answer in the choices and return the choice id
            for j in range(len(choices)):
                if choices[j] == ground_truth:
                    answer = option_dict[j]
            train_dataset.append({'question': question, 'answer': answer})
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='validation')
        dataset = dataset.shuffle(seed=seed)
        val_dataset = []
        for i in range(len(dataset)):
            question = dataset[i]['question']
            choices = dataset[i]['choices']
            options = ['(A) '+choices[0], '(B) '+choices[1], '(C) '+choices[2], '(D) '+choices[3]]
            question = question + ' ' + ', '.join(options)
            ground_truth = dataset[i]['answer'][0]
            # find answer in the choices and return the choice id
            for j in range(len(choices)):
                if choices[j] == ground_truth:
                    answer = option_dict[j]
            val_dataset.append({'question': question, 'answer': answer})
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='test')
        dataset = dataset.shuffle(seed=seed)
        test_dataset = []
        for i in range(len(dataset)):
            question = dataset[i]['question']
            choices = dataset[i]['choices']
            options = ['(A) '+choices[0], '(B) '+choices[1], '(C) '+choices[2], '(D) '+choices[3]]
            question = question + ' ' + ', '.join(options)
            ground_truth = dataset[i]['answer'][0]
            # find answer in the choices and return the choice id
            for j in range(len(choices)):
                if choices[j] == ground_truth:
                    answer = option_dict[j]
            test_dataset.append({'question': question, 'answer': answer})
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")
        prompt = load_prompts(dataset_name)
    elif dataset_name == 'bioasq':
        dataset = [item for i in [11,10,9,8,7] for j in range(len([f for f in os.listdir("./data/bioasq/Task{:d}BGoldenEnriched".format(i)) if f.endswith(".json")])) for item in json.load(open("./data/bioasq/Task{:d}BGoldenEnriched/{:d}B{:d}_golden.json".format(i, i, j+1)))["questions"] if item["type"] == "yesno"]
        random.shuffle(dataset)
        train_dataset = []
        for i in range(int(0.8 * len(dataset))):
            question = dataset[i]['body']
            answer = dataset[i]['exact_answer']
            contexts = 'Context:\n'
            for j in range(len(dataset[i]['snippets'])):
                contexts += dataset[i]['snippets'][j]['text'] + '\n'
            train_dataset.append({'question': contexts+question, 'answer': answer})
        test_dataset = []
        for i in range(int(0.8 * len(dataset)), len(dataset)):
            question = dataset[i]['body']
            answer = dataset[i]['exact_answer']
            contexts = 'Context:\n'
            for j in range(len(dataset[i]['snippets'])):
                contexts += dataset[i]['snippets'][j]['text'] + '\n'
            test_dataset.append({'question': contexts+question, 'answer': answer})
        val_dataset = []
        print(f"Train: {len(train_dataset)} Test: {len(test_dataset)}")
        prompt = load_prompts(dataset_name)
    elif dataset_name == 'gsm8k':
        train_dataset = load_dataset('gsm8k', 'main', split="train[0%:90%]").shuffle(seed=seed)
        val_dataset = load_dataset('gsm8k', 'main', split="train[90%:100%]").shuffle(seed=seed)
        test_dataset = load_dataset('gsm8k', 'main', split="test").shuffle(seed=seed)
        prompt = load_prompts('gsm8k')
    elif dataset_name == 'mednli':
        file_path = "./data/mednli/mednli_train.jsonl"
        train_dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                template = "Sentence A: {}\nSentence B: {}"
                question = template.format(data['sent_a'], data['sent_b'])
                train_dataset.append({'question': question, 'answer': data['label']})
        file_path = "./data/mednli/mednli_test.jsonl"
        ratio = int(0.1 * len(train_dataset))
        val_dataset = train_dataset[-ratio:]
        test_dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                template = "Sentence A: {}\nSentence B: {}"
                question = template.format(data['sent_a'], data['sent_b'])
                test_dataset.append({'question': question, 'answer': data['label']})
        prompt = load_prompts('mednli')
    elif dataset_name == 'mediqa-rqe':
        data = load_dataset('bigbio/mediqa_rqe', 'mediqa_rqe_bigbio_pairs', split='train')
        train_dataset = []
        for i in range(len(data)):
            template = "Question: {}\nSolution: {}"
            question = template.format(data[i]['text_2'], data[i]['text_1'])
            if data[i]['label'] == True or data[i]['label'] == 'true':
                answer = 'true'
            else:
                answer = "false"
            train_dataset.append({'question': question, 'answer': answer})
        data = load_dataset('bigbio/mediqa_rqe', 'mediqa_rqe_bigbio_pairs', split='train')
        val_dataset = []
        for i in range(len(data)):
            template = "Question: {}\nSolution: {}"
            question = template.format(data[i]['text_2'], data[i]['text_1'])
            if data[i]['label'] == True or data[i]['label'] == 'true':
                answer = 'true'
            else:
                answer = "false"
            val_dataset.append({'question': question, 'answer': answer})
        data = load_dataset('bigbio/mediqa_rqe', 'mediqa_rqe_bigbio_pairs', split='train')
        test_dataset = []
        for i in range(len(data)):
            template = "Question: {}\nSolution: {}"
            question = template.format(data[i]['text_2'], data[i]['text_1'])
            if data[i]['label'] == True or data[i]['label'] == 'true':
                answer = 'true'
            else:
                answer = "false"
            test_dataset.append({'question': question, 'answer': answer})
        prompt = load_prompts('mediqa-rqe')
    elif dataset_name == 'pubhealth':
        file_path = "./data/pubhealth/train.tsv"
        # read from the tsv file
        train_dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                data = line.split('\t')
                question = data[2]
                answer = data[-2]
                explanation = data[3]
                question_template = "{}"
                template = "{}\n#### {}"
                train_dataset.append({'question': question_template.format(question), 'answer': template.format(explanation, answer)})
        train_dataset = train_dataset[1:]
        file_path = "./data/pubhealth/dev.tsv"
        # read from the tsv file
        val_dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                data = line.split('\t')
                question = data[2]
                answer = data[-2]
                explanation = data[3]
                question_template = "{}"
                template = "{}\n#### {}"
                val_dataset.append({'question': question_template.format(question), 'answer': template.format(explanation, answer)})
        val_dataset = val_dataset[1:]
        file_path = "./data/pubhealth/test.tsv"
        # read from the tsv file
        test_dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                data = line.split('\t')
                question = data[2]
                answer = data[-2]
                explanation = data[3]
                question_template = "{}"
                template = "{}\n#### {}"
                test_dataset.append({'question': question_template.format(question), 'answer': template.format(explanation, answer)})
        test_dataset = test_dataset[1:]
        prompt = load_prompts('pubhealth')
    elif dataset_name == 'cord19':
        data_list = load_dataset('mystic-leung/medical_cord19', split='test')
        data = []
        for line in data_list:
            data.append({'question': line['input'], 'answer': line['output']})
        random.shuffle(data)
        train_dataset = data[:3000]
        val_dataset = data[3000:3500]
        test_dataset = data[3500:4000]
        prompt = load_prompts('cord19')


    for i in range(len(train_dataset)):
        d = {'question': train_dataset[i]['question'], 'answer': train_dataset[i]['answer']}
        file_name = f"./data/{dataset_name}/raw_train.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')
    
    for i in range(len(val_dataset)):
        d = {'question': val_dataset[i]['question'], 'answer': val_dataset[i]['answer']}
        file_name = f"./data/{dataset_name}/raw_val.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')

    for i in range(len(test_dataset)):
        d = {'question': test_dataset[i]['question'], 'answer': test_dataset[i]['answer']}
        file_name = f"./data/{dataset_name}/raw_test.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')

    return train_dataset, val_dataset, test_dataset, prompt

if __name__ == '__main__':
    get_datasets(42, "gsm8k")