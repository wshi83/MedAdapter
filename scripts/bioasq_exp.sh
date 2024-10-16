CUDA_VISIBLE_DEVICES=0,3 python main-openai.py --debug generation --config configs/bioasq/bioasq-gen-test.yaml
CUDA_VISIBLE_DEVICES=0,3 python main-openai.py --debug generation --config configs/bioasq/bioasq-gen-train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 --main_process_port 29666 main.py --debug reward --config configs/bioasq/bioasq-reward.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision fp16 --main_process_port 29666 main.py --debug reward_guide --config configs/bioasq/bioasq-guide.yaml