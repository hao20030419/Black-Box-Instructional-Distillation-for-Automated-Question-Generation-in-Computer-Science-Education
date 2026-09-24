import os
import sys
import json
import csv
from dotenv import load_dotenv
from dotenv import load_dotenv

# 讓 Windows 下的 multiprocessing 能讀到環境變數
load_dotenv()

 # 確保專案根目錄在 sys.path，使得無論在哪個目錄執行都能 import GeminiAgent package
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from GeminiAgent.agent.generator import generate_raw_question, random_topic
from GeminiAgent.agent.comment_agent import comment_question
from GeminiAgent.agent.refine_agent import refine_question
from GeminiAgent.agent.beautify import beautify
from GeminiAgent.agent.reviewer import review_question
import GeminiAgent.agent.generator as generator_module

# GeminiAgent/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


def _count_and_repair_jsonl(path: str) -> int:
    """Count valid JSON lines in path. If last line(s) are corrupt, truncate file to keep only valid lines."""
    if not os.path.exists(path):
        return 0
    valid_lines = []
    with open(path, "r", encoding="utf-8") as fr:
        for ln in fr:
            ln_strip = ln.strip()
            if not ln_strip:
                continue
            try:
                json.loads(ln_strip)
                valid_lines.append(ln_strip)
            except Exception:
                break

    # overwrite with only valid lines (this will also truncate any partial/corrupt tail)
    with open(path, "w", encoding="utf-8") as fw:
        for ln in valid_lines:
            fw.write(ln + "\n")

    return len(valid_lines)


# multiprocessing removed: generation runs synchronously in main process


# writer removed; generation and writing are performed synchronously in generate_dataset


# ----------------------------------------------------
# 主程式：產生題目資料集（FT-DataPrep Pipeline）
# ----------------------------------------------------
def generate_dataset(total=1, workers=1):
    """Synchronous dataset generation (no multiprocessing).

    `workers` argument is accepted for backward compatibility but ignored.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset_4000.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "review.csv")
    tokens_csv = os.path.join(OUTPUT_DIR, "tokens.csv")

    # 先計算現有的題目數量，再判斷是否需要繼續生成。
    # If a cleaned dataset exists (clean_dataset.jsonl), prefer its count
    # because the cleaning step may have removed some items.
    clean_path = os.path.join(OUTPUT_DIR, "clean_dataset_4000.jsonl")
    existing_clean = _count_and_repair_jsonl(clean_path) if os.path.exists(clean_path) else 0

    if existing_clean > 0:
        existing = existing_clean
        print(f"[Generator] 使用已清理資料計數：clean_dataset.jsonl 有 {existing} 條。")
    else:
        existing = _count_and_repair_jsonl(jsonl_path)

    if existing >= total:
        print(f"[Generator] 目標數量 {total} 已達成（現有 {existing} 條），無需生成。")
        return

    remaining = total - existing

    # 載入 prompts.json（如果存在），單程序下只需載入一次
    try:
        prompts_path = os.path.join(os.path.dirname(generator_module.__file__), "prompts.json")
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as _f:
                data = json.load(_f)
                if isinstance(data, list) and data:
                    generator_module.PROMPT_TEMPLATES = data
    except Exception:
        pass

    csv_exists = os.path.exists(csv_path)
    tokens_exists = os.path.exists(tokens_csv)

    mode = "a" if os.path.exists(jsonl_path) else "w"
    with open(jsonl_path, mode, encoding="utf-8") as fj, \
         open(csv_path, "a" if csv_exists else "w", newline="", encoding="utf-8-sig") as fc, \
         open(tokens_csv, "a" if tokens_exists else "w", newline="", encoding="utf-8-sig") as ft:

        w = csv.writer(fc)
        if not csv_exists:
            w.writerow(["Topic", "Keep", "Reason", "Question"])

        wt = csv.writer(ft)
        if not tokens_exists:
            wt.writerow([
                "Topic",
                "Generator_in","Generator_out",
                "Comment_in","Comment_out",
                "Refine_in","Refine_out",
                "Beautify_in","Beautify_out",
                "Review_in","Review_out",
                "TotalInput","TotalOutput","TotalTokens",
                "Keep"
            ])

        for i in range(remaining):
            topic = random_topic()
            try:
                used_prompt, raw_question, gen_out_tokens, gen_in_tokens = generate_raw_question(topic)
                comment, comment_out_tokens, comment_in_tokens = comment_question(raw_question)
                refined_question, refine_out_tokens, refine_in_tokens = refine_question(raw_question, comment)
                pretty_question, beautify_out_tokens, beautify_in_tokens = beautify(refined_question)
                keep, reason, review_out_tokens, review_in_tokens = review_question(pretty_question)

                payload = {
                    "topic": topic,
                    "question": used_prompt,
                    "messages": [
                        {"role": "user", "content": used_prompt},
                        {"role": "assistant", "content": pretty_question},
                    ],
                    "keep": keep,
                    "reason": reason,
                    "meta": {
                        "raw": raw_question,
                        "comment": comment,
                        "refined": refined_question,
                    },
                    "tokens": {
                        "generator_out": gen_out_tokens,
                        "generator_in": gen_in_tokens,
                        "comment_out": comment_out_tokens,
                        "comment_in": comment_in_tokens,
                        "refine_out": refine_out_tokens,
                        "refine_in": refine_in_tokens,
                        "beautify_out": beautify_out_tokens,
                        "beautify_in": beautify_in_tokens,
                        "review_out": review_out_tokens,
                        "review_in": review_in_tokens,
                    }
                }
            except Exception as e:
                payload = {"_error": str(e), "topic": topic}

            # write
            fj.write(json.dumps(payload, ensure_ascii=False) + "\n")

            # if payload is an error we still write a csv line with minimal info
            if payload.get("_error"):
                w.writerow([payload.get("topic", ""), False, payload.get("_error"), ""])
                wt.writerow([payload.get("topic", ""), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False])
            else:
                w.writerow([
                    payload["topic"],
                    payload["keep"],
                    payload["reason"],
                    payload["messages"][1]["content"],
                ])

                toks = payload.get("tokens", {})
                gen_out = int(toks.get("generator_out") or 0)
                gen_in = int(toks.get("generator_in") or 0)
                com_out = int(toks.get("comment_out") or 0)
                com_in = int(toks.get("comment_in") or 0)
                ref_out = int(toks.get("refine_out") or 0)
                ref_in = int(toks.get("refine_in") or 0)
                bea_out = int(toks.get("beautify_out") or 0)
                bea_in = int(toks.get("beautify_in") or 0)
                rev_out = int(toks.get("review_out") or 0)
                rev_in = int(toks.get("review_in") or 0)

                total_input = sum([gen_in, com_in, ref_in, bea_in, rev_in])
                total_output = sum([gen_out, com_out, ref_out, bea_out, rev_out])
                total_tokens = total_input + total_output

                wt.writerow([
                    payload.get("topic", ""),
                    gen_in, gen_out,
                    com_in, com_out,
                    ref_in, ref_out,
                    bea_in, bea_out,
                    rev_in, rev_out,
                    total_input, total_output, total_tokens,
                    payload.get("keep", False),
                ])

            print(f"[Generator] 已完成 {existing + i + 1}/{total}")

    print("=== FT-DataPrep Pipeline：Gemini 資料集生成完成 ===")

if __name__ == "__main__":
    # 允許直接以腳本方式執行：預設產生 20 題、4 個 workers
    # 可依需求改為較小數量做快速驗證
    try:
        # 嘗試從環境變數帶入數量（可選）
        total = int(os.getenv("KD_GEN_TOTAL", "4000"))
        workers = int(os.getenv("KD_GEN_WORKERS", "1"))
    except Exception:
        total, workers = 4000, 1

    print(f"[Runner] 開始生成資料集：total={total}, workers={workers}")
    
    generate_dataset(total=total, workers=workers)