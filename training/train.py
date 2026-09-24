import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json

# 確保 GPU 可見
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model identifier")
    parser.add_argument("--dataset_path", type=str, default="../GeminiAgent/results/clean_dataset_4000.jsonl")
    parser.add_argument("--output_dir", type=str, default="./qwen-ft4000-v2")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5) 
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    # 1. 加載數據集並清理欄位
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    def standardize_data(example):
        msgs = example.get("messages") or []
        if isinstance(msgs, list) and len(msgs) >= 2:
            return {"messages": msgs}

        prompt = example.get("question", "")
        response_data = example.get("LLMans", "")

        if isinstance(response_data, dict):
            response = json.dumps(response_data, ensure_ascii=False, indent=2)
        else:
            response = str(response_data)

        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return {"messages": msgs}
    
    ds = ds.map(standardize_data, remove_columns=ds.column_names)

    # 2. 載入 Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 載入模型 (QLoRA)
    print("Loading model...")
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Config (優化點：增加 Dropout 防止過擬合)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # 從 0.05 提升到 0.1，強迫模型泛化
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments (優化點：增加 save_steps 進行滾動評估)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,         # 10% 熱身，平滑啟動
        weight_decay=0.01,        # 輕微的權重衰減，防止過擬合
        logging_steps=5,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",    # 改為按步數存檔，方便找尋「黃金 Step」
        save_steps=50,            # 每 50 步存一個 checkpoint
        save_total_limit=5,       # 只保留最新的 5 個，省硬碟空間
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    # 6. 格式化函數
    def formatting_prompts_func(examples):
        output_texts = []
        for messages in examples['messages']:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # 7. Data Collator
    response_template = "<|im_start|>assistant" 
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    # 8. 初始化 SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()