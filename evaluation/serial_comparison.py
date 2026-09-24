import argparse
import os
import json
import time
import random
import csv
import torch
import gc
import re
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    extract_multi_scores,
    cleanup_gpu
)
from GeminiAgent.agent.generator import PROMPT_TEMPLATES, random_topic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/results.csv")
    parser.add_argument("--responses_dir", type=str, default="evaluation/responses")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.responses_dir, exist_ok=True)
    
    model_paths = [p.rstrip("/\\") for p in args.model_dirs]
    model_names = [os.path.basename(p) for p in model_paths]
    
    # --- Step 1: Scenario Preparation (固定每一輪的主題) ---
    scenarios = []
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(PROMPT_TEMPLATES)
        scenarios.append({
            "round": i + 1,
            "topic": topic,
            "full_prompt": template.format(topic=topic)
        })

    # --- Step 2: Generation (模型生成階段) ---
    for path, name in zip(model_paths, model_names):
        print(f"\n>>> Loading Model: {name} (Path: {path})")
        try:
            model, tokenizer, gen = load_finetuned_model(path, load_in_4bit=True)
            print(f"    Model loaded successfully. Generating {len(scenarios)} responses...")
            
            for sc in scenarios:
                # 確保生成時使用的是該輪特定的 prompt
                resp = gen_from_finetuned(gen, sc["full_prompt"], temperature=args.temperature, top_p=args.top_p)
                
                # Double check response content
                if not resp:
                    print(f"    [WARNING] Empty response for Round {sc['round']} Topic {sc['topic']}")
                    resp = "[EMPTY RESPONSE]"
                
                fname = f"round{sc['round']}_model_{name}.txt"
                out_path = os.path.join(args.responses_dir, fname)
                
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(resp)
                print(f"    Saved: {fname} ({len(resp)} chars)")
            
            # 釋放顯存
            del gen, model, tokenizer
            cleanup_gpu()
            print(f"    Unloaded {name}.")

        except Exception as e:
            print(f"!!! CRITICAL ERROR on model {name} !!!")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Skipping this model for generation phase.")

    # --- Step 3: Judging (評審階段) ---
    fieldnames = ["round", "topic", "prompt"] + [f"score_{n}" for n in model_names] + ["judge_raw_response"]
    totals = {n: 0 for n in model_names}

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sc in scenarios:
            # 打亂順序以避免位置偏差 (Position Bias)
            shuffled_indices = list(range(len(model_names)))
            random.shuffle(shuffled_indices)
            
            judge_content = ""
            idx_to_name = {} 
            
            for i, s_idx in enumerate(shuffled_indices):
                m_name = model_names[s_idx]
                m_label = i + 1
                idx_to_name[m_label] = m_name
                
                fpath = os.path.join(args.responses_dir, f"round{sc['round']}_model_{m_name}.txt")
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as rf:
                        content = rf.read()
                else:
                    content = "No Output"
                
                judge_content += f"=== Model {m_label} ===\n{content}\n\n"

            # 【關鍵修正】: 使用 sc['topic'] 而非全局變數 topic
            judge_prompt = (
                f"你是一位嚴格的學術審查員。請使用基於評分的方法（結合思維鏈，即先說明理由），"
                f"針對主題「{sc['topic']}」，評估這 {len(model_paths)} 個模型所產生的考試題目。\n\n"
                f"### 評分規則\n"
                f"1. **評估範圍**：請僅依據題幹與選項進行評判。忽略解釋、答案、格式、冗餘文字或幻覺內容。\n"
                f"2. **評分方式**：請給予每個候選題目 1 到 10 分的分數，其中 1 代表表現最差，10 代表表現最好。\n"
                f"3. **評估重點**：評分應主要基於正確性與深度。正確性是指題目邏輯是否合理且不包含明顯錯誤；"
                f"深度是指題目是否涉及高階應用或需要較高層次的思考。\n"
                f"4. **輸出要求**：請先簡要說明評分理由，最後務必以「Final Ratings:」標籤開頭並按下列格式列出每個模型對應的分數：\n\n"
                f"【待評內容】\n{judge_content}\n\n"
                f"Final Ratings:\n" + "\n".join([f"Model {i+1} Score: X" for i in range(len(model_paths))])
            )

            print(f"--- Round {sc['round']} Judging Topic: {sc['topic']} ---")
            raw_judge_resp, _, _ = gen_from_gemini(judge_prompt)
            
            # 使用更魯棒的解析方式
            extracted_scores = extract_multi_scores(raw_judge_resp, len(model_paths))
            
            row = {
                "round": sc["round"], 
                "topic": sc["topic"], 
                "prompt": sc["full_prompt"], 
                "judge_raw_response": raw_judge_resp
            }
            
            # 映射分數回 CSV
            for m_label, m_real_name in idx_to_name.items():
                s = extracted_scores.get(m_label, -1)
                row[f"score_{m_real_name}"] = s
                if s > 0: totals[m_real_name] += s

            writer.writerow(row)
            f.flush()
            print(f"Parsed Scores: {extracted_scores}")

        # Write Final Totals to CSV
        total_row = {
            "round": "TOTAL",
            "topic": "SUM",
            "prompt": "",
            "judge_raw_response": ""
        }
        for m_name in model_names:
            total_row[f"score_{m_name}"] = totals[m_name]
        
        writer.writerow(total_row)

    print("\n" + "="*30)
    print("Evaluation Complete.")
    print("Final Totals:", totals)
    print("="*30)

if __name__ == "__main__":
    main()