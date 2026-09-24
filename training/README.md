Usage

1. Install dependencies (preferably in a venv):

```bash
pip install -r fine-tune/requirements.txt
```

2. Train (example):

```bash
python fine-tune/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_path GeminiAgent/results/clean_dataset.jsonl \
  --output_dir ./checkpoint-qwen-ft \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1
```

8-bit / 4-bit example (requires `bitsandbytes` and transformers support):

```bash
python fine-tune/train.py \
  --model_name_or_path Qwen/qwen-2.5-7b-instruct \
  --dataset_path GeminiAgent/results/clean_dataset.jsonl \
  --output_dir ./checkpoint-qwen-ft-8bit \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --load_in_8bit
```

Or for 4-bit (experimental):

```bash
python fine-tune/train.py --model_name_or_path Qwen/qwen-2.5-7b-instruct --dataset_path GeminiAgent/results/clean_dataset.jsonl --output_dir ./checkpoint-qwen-ft-4bit --load_in_4bit
```

Accelerate launch example (uses `accelerate_config.yaml` in this folder):

```bash
accelerate launch --config_file fine-tune/accelerate_config.yaml \
  fine-tune/train.py \
  --model_name_or_path Qwen/qwen-2.5-7b-instruct \
  --dataset_path GeminiAgent/results/clean_dataset.jsonl \
  --output_dir fine-tune/checkpoint-qwen-ft-8bit \
  --per_device_train_batch_size 1 \
  --load_in_8bit
```

3. Evaluate / interact with the fine-tuned model:

```bash
python fine-tune/eval.py --model_dir ./checkpoint-qwen-ft
```

Notes:
- This script uses Hugging Face Transformers + PEFT (LoRA). For 7B models you should run on GPU with appropriate memory (or use 8-bit + bitsandbytes). Adjust `train.py` args accordingly.
- The dataset loader assumes `clean_dataset.jsonl` contains JSONL with `messages` field where `messages[0]` is user and `messages[1]` is assistant. If your structure differs, change `build_example` in `train.py` accordingly.
