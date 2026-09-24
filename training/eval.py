import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch


def load_model(model_dir):
    # If LoRA/PEFT used, load with PeftModel
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto" if torch.cuda.is_available() else None)
        # If model_dir contains pytorch_model.bin with PEFT adapter, PeftModel.from_pretrained will wrap
        try:
            model = PeftModel.from_pretrained(base_model, model_dir)
            return model
        except Exception:
            return base_model
    except Exception as e:
        # Try loading with AutoModelForCausalLM directly
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Write a short poem about AI.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model(args.model_dir)
    model.eval()

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    out = gen(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.95, temperature=0.8)
    print(out[0]['generated_text'])


if __name__ == "__main__":
    main()
