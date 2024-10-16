import argparse
from eval_fn.gsm8k_eval import gsm8k_metrics
from eval_fn.pubmedqa_eval import pubmedqa_metrics
from eval_fn.medmcqa_eval import medmcqa_metrics
from eval_fn.mmlu_eval import mmlu_metrics
from eval_fn.bioasq_eval import bioasq_metrics
from eval_fn.medqa_eval import medqa_metrics
from eval_fn.cord19_eval import cord19_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gsm8k', type=str, help='Dataset to evaluate.')
    parser.add_argument('--solution_dir', default='data/gsm8k/generation/iter_warmup/prm/0_selected_test.jsonl', type=str, help='Path to the generated solutions.')
    parser.add_argument('--split', default='train', type=str, help='Split of the dataset to evaluate.')
    parser.add_argument('--style', default='interleave', type=str, help='Style of sampling the generation for eval.')
    parser.add_argument('--k', default=1, type=int, help='Sample from every k data.')
    args = parser.parse_args()

    if args.dataset == 'gsm8k':
        stats = gsm8k_metrics(args)
    elif args.dataset == 'pubmedqa':
        stats = pubmedqa_metrics(args)
    elif args.dataset == 'medmcqa':
        stats = medmcqa_metrics(args)
    elif args.dataset == 'mmlu':
        stats = mmlu_metrics(args)
    elif args.dataset == 'medqa':
        stats = medqa_metrics(args)
    elif args.dataset == 'bioasq':
        stats = bioasq_metrics(args)
    elif args.dataset == 'cord19':
        stats = cord19_metrics(args)
    print(stats)

if __name__ == "__main__":
    main()