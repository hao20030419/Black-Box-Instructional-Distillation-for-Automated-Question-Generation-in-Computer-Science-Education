import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors
import time

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_content_with_tokens(model: str, contents: str):
    # (原本的估算邏輯保留)
    in_tokens = len(contents.split())
    
    # --- 修正後的重試區塊 ---
    max_retries = 3
    for i in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            break  # 成功拿到 resp，跳出迴圈
        except errors.ServerError as e:
            if i < max_retries - 1:  # 如果不是最後一次嘗試
                print(f"伺服器過載 (嘗試 {i+1}/3)，10 秒後重試...")
                time.sleep(10)
                continue
            else:
                raise e  # 試了三次都失敗，才拋出錯誤讓程式停止
    # -----------------------

    # 以下維持你原本的邏輯，不需要改動
    text = resp.text.strip() if resp.text else ""

    out_tokens = None
    try:
        # 你原本抓取 Token 的各種 try-except 判斷
        if hasattr(resp, "usage_metadata"): # 優先使用官方最新格式
            out_tokens = resp.usage_metadata.candidates_token_count
            in_tokens = resp.usage_metadata.prompt_token_count # 更新為精確值
            
        if out_tokens is None and hasattr(resp, "tokenUsage"):
            # ... (你原本的其他判斷邏輯)
            pass 
    except Exception:
        out_tokens = None

    # Fallback 邏輯維持不變
    if out_tokens is None:
        out_tokens = len(text.split())

    try:
        out_tokens = int(out_tokens)
    except Exception:
        out_tokens = len(text.split())

    return text, out_tokens, in_tokens
