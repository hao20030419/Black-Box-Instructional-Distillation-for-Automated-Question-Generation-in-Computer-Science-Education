import os
from google import genai
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from dotenv import load_dotenv

"""
Beautify Agent
- LLM 名稱：beautify_LLM
- 功能：整理排版，移除奇怪符號，維持語意不變
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BEAUTIFY_LLM_NAME = "beautify_LLM"
BEAUTIFY_MODEL_NAME = "gemini-3-pro-preview"

BEAUTIFY_PROMPT = """
你是 beautify_LLM，負責整理題目排版。

請將以下題目：
- 移除多餘的奇怪符號
- 保持 Markdown / 結構清楚
- 不要更改題目內容的事實與答案

只輸出整理後的題目。
"""


def beautify(text: str) -> str:
    contents = f"{BEAUTIFY_PROMPT}\n{text}"
    text_out, out_tokens, in_tokens = generate_content_with_tokens(BEAUTIFY_MODEL_NAME, contents)
    return text_out, out_tokens, in_tokens