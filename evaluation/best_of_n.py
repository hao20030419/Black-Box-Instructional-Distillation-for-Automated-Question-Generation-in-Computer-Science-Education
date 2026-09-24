import argparse
import os
import json
import random
import csv
import time
import re
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    cleanup_gpu
)
from GeminiAgent.agent.generator import random_topic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--repeats", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--output_csv", type=str, default="evaluation/best_of_n_results.csv")
    parser.add_argument("--prompts_file", type=str, default="GeminiAgent/agent/prompts.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model_paths = [p.rstrip("/\\") for p in args.model_dirs]
    model_names = [os.path.basename(p) for p in model_paths]
    
    print(f"Loading prompts from {args.prompts_file}...")
    try:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_templates = json.load(f)
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        return

    # Initialize Scoreboard
    scoreboard = {name: 0 for name in model_names}

    # --- Step 1: Prepare Topics & Prompts ---
    tasks = []
    print(f"--- Preparing {args.repeats} Tasks ---")
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(prompt_templates)
        full_prompt = template.format(topic=topic)
        tasks.append({
            "id": i+1,
            "topic": topic,
            "prompt": full_prompt
        })

    # Store responses: responses[task_id][model_name] = text
    responses_storage = {t["id"]: {} for t in tasks}

    # --- Step 2: Generation (Serial Loading) ---
    print("\n=== Phase 1: Generation ===")
    for path, name in zip(model_paths, model_names):
        print(f"\n>>> Loading Model: {name}...")
        try:
            model, tokenizer, gen = load_finetuned_model(path, load_in_4bit=True)
            
            print(f"    Generating responses for {len(tasks)} tasks...")
            for task in tasks:
                prompt = task["prompt"]
                resp = gen_from_finetuned(gen, prompt, temperature=args.temperature, top_p=args.top_p)
                
                if not resp or len(resp.strip()) == 0:
                    resp = "[EMPTY_RESPONSE]"
                
                responses_storage[task["id"]][name] = resp
            
            print(f"    Model {name} finished.")
            
            del gen, model, tokenizer
            cleanup_gpu()
            
        except Exception as e:
            print(f"!!! Error loading/running {name}: {e}")
            import traceback
            traceback.print_exc()
            for task in tasks:
                responses_storage[task["id"]][name] = "[ERROR]"

    # --- Step 3: Judging (Best of N) ---
    print("\n=== Phase 2: Best-of-N Judging ===")
    
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Updated CSV fields
    fieldnames = ["task_id", "topic", "prompt", "winner_model", "judge_reason", "model_mapping"] + [f"resp_{name}" for name in model_names]

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            t_id = task["id"]
            topic = task["topic"]
            prompt = task["prompt"]
            
            print(f"\n--- Judging Task {t_id}/{len(tasks)}: {topic} ---")
            
            # Identify models participating in this round
            models_in_round = model_names.copy()
            
            # Shuffle models to create Blind Test
            random.shuffle(models_in_round)
            
            # Construct Judge Prompt
            judge_content = ""
            # Mapping from 'Model X' label to real model name
            label_map = {} 
            label_map_str_list = []
            
            for idx, m_name in enumerate(models_in_round):
                label = f"Model {idx+1}"
                label_map[label] = m_name
                label_map_str_list.append(f"{label}: {m_name}")
                resp_text = responses_storage[t_id][m_name]
                judge_content += f"=== {label} ===\n{resp_text}\n\n"
            
            # Create a string representation of the mapping for the CSV
            label_mapping_str = " | ".join(label_map_str_list)

            gemini_prompt = (
                f"你是一位嚴格的學術審查員。請使用基於排名的評估方法（結合思維鏈，即先說明理由），"
                f"針對主題 {topic}，評估這 {len(models_in_round)} 個模型所產生的考試題目。\n\n"
                f"### 評估規則\n"
                f"1. **評估範圍**：請僅依據題幹與選項進行評判。忽略解釋、答案、格式、冗餘文字或幻覺內容。\n"
                f"2. **評估重點**：評估應主要基於正確性與深度。正確性是指題目邏輯是否合理且不包含明顯錯誤；"
                f"深度是指題目是否涉及高階應用或需要較高層次的思考。\n"
                f"3. **最終決定**：基於上述分析，請從所有候選題目中僅選出一個整體表現最佳的候選題目。\n"
                f"4. **輸出要求**：請先簡要說明評判理由，然後明確指出最終勝出的模型。\n\n"
                f"### 各模型回答\n"
                f"{judge_content}\n\n"
                f"請務必在回答的最後一行，嚴格按照此格式輸出贏家：\n"
                f"Winner: Model X\n"
                f"(其中 X 為該模型的編號，例如 Winner: Model 1)"
            )
            
            try:
                # Call Gemini Judge
                # Using the imported gen_from_gemini function
                # Note: original code passed args.temperature, but gen_from_gemini might not take it or uses default.
                # Checking utils.py usage in original code context. 
                # Assuming gen_from_gemini(prompt) is the correct signature based on imports.
                judge_resp, _, _ = gen_from_gemini(gemini_prompt)
                
                # Parse Winner
                # Regex for "Winner: Model <digits>"
                match = re.search(r"Winner:\s*(Model\s*\d+)", judge_resp, re.IGNORECASE)
                
                if match:
                    winner_label = match.group(1) # e.g. "Model 3" or "model 3"
                    # Normalize label format to match keys in label_map (Model X)
                    # We assume regex captured "Model <digits>"
                    # Let's clean it up to be sure
                    parts = winner_label.split()
                    clean_label = f"Model {parts[-1]}" # "Model", "3"
                    
                    real_winner = label_map.get(clean_label, "Unknown")
                    if real_winner != "Unknown":
                        scoreboard[real_winner] += 1
                else:
                    real_winner = "Parse Error"
                
                print(f"   Winner: {real_winner} (Label: {match.group(1) if match else 'None'})")

                # Build row
                row = {
                    "task_id": t_id,
                    "topic": topic,
                    "prompt": prompt,
                    "winner_model": real_winner,
                    "judge_reason": judge_resp,
                    "model_mapping": label_mapping_str
                }
                # Fill model responses
                for m_name in model_names:
                    row[f"resp_{m_name}"] = responses_storage[t_id][m_name]
                
                writer.writerow(row)
                csvfile.flush()
                time.sleep(1)

            except Exception as e:
                print(f"Error judging task {t_id}: {e}")

    # --- Step 4: Final Report ---
    print("\n" + "="*40)
    print("BEST-OF-N TOURNAMENT RESULTS")
    print("="*40)
    print(f"{'Model Name':<30} | {'Wins':<5} | {'Win Rate':<8}")
    print("-" * 60)
    
    sorted_models = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
    total_tasks = len(tasks)
    
    # === Final Report & Summary CSV ===
    summary_csv = args.output_csv.replace(".csv", "_summary.csv")
    print(f"\nWriting summary results to: {summary_csv}")
    
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as sum_f:
        sum_writer = csv.writer(sum_f)
        sum_writer.writerow(["Model Name", "Total Wins", "Win Rate"])

        for name, wins in sorted_models:
            rate = (wins / total_tasks * 100) if total_tasks > 0 else 0
            rate_str = f"{rate:.1f}%"
            
            # Print to console
            print(f"{name:<30} | {wins:<5} | {rate_str}")
            
            # Write to CSV
            sum_writer.writerow([name, wins, rate_str])

    if sorted_models:
        print("="*40)
        print(f"🏆 CHAMPION: {sorted_models[0][0]}")
        print("="*40)

if __name__ == "__main__":
    main()
