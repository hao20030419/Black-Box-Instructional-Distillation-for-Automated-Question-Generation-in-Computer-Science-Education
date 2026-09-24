import os
import json
import random
import re
from dotenv import load_dotenv
from google import genai


# ================================================================
# 🔧 初始化
# ================================================================
load_dotenv()

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise ValueError("❌ 找不到 GEMINI_API_KEY，請確認 .env 設定正確")

client = genai.Client(api_key=key)

# 指向 GeminiAgent 根目錄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 🎯 Format: 根據 topic 生成固定 prompt
# ================================================================
def extract_subject(topic: str) -> str:
    if not topic:
        return "資料結構"
    if "-" in topic:
        return topic.split("-")[0].strip()
    return topic.strip()


# ================================================================
# ✂️ 抽取題幹 + A/B/C/D 選項（Instruction fine-tune 最重要部分）
# ================================================================
def extract_question_only(full: str) -> dict:
    # 1️⃣ 移除答案與解析
    full = re.sub(r"(?i)(答案|正確答案|解析|解釋)[:：].*", "", full)

    # 2️⃣ 移除答案提示句
    full = re.sub(r"(?i)(正確為|正確選項|答案是|答案為|the correct answer is).*", "", full)

    # 3️⃣ 移除符號提示
    full = re.sub(r"[✓✔✗✘→←★⭐•＊*]+", "", full)

    # 4️⃣ 移除 (正確)、(incorrect)
    full = re.sub(r"\(.*?(正確|錯誤|correct|incorrect).*?\)", "", full, flags=re.I)

    # 5️⃣ 移除 markdown / latex / 前置噪音
    full = re.sub(r"(?i)^題目[:：]?\s*", "", full)
    full = re.sub(r"(?i)^以下.*內容[:：]\s*", "", full)
    full = re.sub(r"(?i)^這.*版本.*?\s*", "", full)
    full = re.sub(r"###\s*題目\s*", "", full)
    full = full.replace("###", "").replace("```", "").replace("$", "")
    # 將 LaTeX 的 \pmod 統一為文字 mod
    full = re.sub(r"\\pmod", "mod", full)

    lines = full.splitlines()

    stem_lines = []
    options = {}
    option_count = 0

    # 6️⃣ 抽取題幹與選項
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # 行首若帶有「題目：」等字樣，去除以維持一致
        s = re.sub(r"(?i)^題目[:：]?\s*", "", s)

        # 偵測選項 A/B/C/D
        match = re.match(r"^\(?([A-Da-d])\)?[.)]?\s*(.*)", s)
        if match:
            key = match.group(1).upper()
            text = match.group(2).strip()

            # 清理符號
            text = re.sub(r"(←|→|<-|->)", "", text).strip()
            text = re.sub(r"(?i)(正確|最佳選項|最合適).*", "", text).strip()

            if key not in options:
                options[key] = text
                option_count += 1

            if option_count == 4:
                break
        else:
            if option_count == 0:
                stem_lines.append(s)

    stem = " ".join(stem_lines).strip()
    # 再次保險移除開頭「題目：」
    stem = re.sub(r"(?i)^題目[:：]?\s*", "", stem)

    # 7️⃣ 保證 A/B/C/D 四個選項存在
    final_options = {k: options.get(k, "") for k in ["A", "B", "C", "D"]}

    # 選項內容標準化：
    # - 統一 \pmod -> mod（若上面殘留）
    # - 對於像 "k mod N + 1" 的寫法，補上括號為 "((k mod N) + 1)" 以避免與 (k+1) mod N 混淆
    def _normalize_option(t: str) -> str:
        t = re.sub(r"\\pmod", "mod", t)
        # 將 'a mod b + 1' -> '((a mod b) + 1)'
        t = re.sub(r"\b([A-Za-z0-9_]+)\s*mod\s*([A-Za-z0-9_]+)\s*\+\s*1\b", r"((\1 mod \2) + 1)", t)
        return t

    for k in list(final_options.keys()):
        final_options[k] = _normalize_option(final_options[k])

    return {
        "stem": stem,
        "options": final_options
    }


# ================================================================
# 🧹 Clean dataset（不再進行審查，使用 keep flag）
# ================================================================
def clean_dataset(source_name="dataset_4000.jsonl"):
    src = os.path.join(OUTPUT_DIR, source_name)
    out = os.path.join(OUTPUT_DIR, "clean_dataset_4000.jsonl")
    removed = os.path.join(OUTPUT_DIR, "removed.jsonl")

    keep = drop = 0

    with open(src, "r", encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout, \
         open(removed, "w", encoding="utf-8") as fdrop:

        for line in fin:
            data = json.loads(line)

            # 1️⃣ dataset.jsonl 已帶有 keep flag → 直接判斷
            if not data.get("keep", False):
                drop += 1
                fdrop.write(json.dumps({
                    "reason": data.get("reason", "keep = false"),
                    "content": data
                }, ensure_ascii=False) + "\n")
                continue

            # 2️⃣ 讀取題目
            try:
                if "content" in data:
                    full = data["content"].strip()
                elif "messages" in data:
                    full = data["messages"][1]["content"].strip()
                else:
                    raise KeyError("缺少 content 或 messages 欄位")

            except Exception as e:
                drop += 1
                fdrop.write(json.dumps({
                    "reason": f"題目內容無法讀取：{e}",
                    "content": data
                }, ensure_ascii=False) + "\n")
                continue

            keep += 1
            print(f"[KEEP] {data.get('reason', '')}")

            # 3️⃣ 解析題幹與選項
            llmans = extract_question_only(full)

            # 4️⃣ 使用實際生成時的 prompt（直接從 dataset.jsonl 的 question 欄位讀取）
            # 這樣可以保留多樣化的人類語氣 prompt，與題目形成一一對應的 instruction-following 對
            fout.write(json.dumps({
                "question": data.get("question", ""),
                "LLMans": llmans
            }, ensure_ascii=False) + "\n")

    print("\n=== 清洗完成 ===")
    print(f"✔ 保留：{keep}")
    print(f"✖ 移除：{drop}")
    print(f"📄 清洗後輸出：{out}")


# ================================================================
# 🚀 主程式
# ================================================================
if __name__ == "__main__":
    print("🚀 清洗 dataset.jsonl 中的題目...")
    clean_dataset()