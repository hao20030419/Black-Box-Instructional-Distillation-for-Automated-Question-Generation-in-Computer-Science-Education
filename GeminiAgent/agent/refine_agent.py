import os
from google import genai
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from dotenv import load_dotenv

"""
Comment Agent (部分 2)
- LLM 名稱：refine_LLM
- 功能：根據 comment_LLM 的建議，重新改寫題目
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

REFINE_LLM_NAME = "refine_LLM"
REFINE_MODEL_NAME = "gemini-3-pro-preview"

REFINE_PROMPT = """
你是一位題目編修專家 (refine_LLM)。

根據以下兩段資訊：
1. 原題目
2. 題目優化建議

請你生成「修改後的完整題目」，需包含：
- 題目敘述
- 四個選項 (A)(B)(C)(D)
- 答案
- 解析

注意：
- 保留原本的考點與知識重點
- 可以調整語氣、順序與例子，使其更清楚
- 不要加入與主題無關的新知識
- 範本格式與原題類似即可
"""


def refine_question(original_question: str, comment: str) -> str:
    """
    使用 refine_LLM 根據 comment 改寫題目
    """
    msg = (
        f"{REFINE_PROMPT}\n\n"
        f"【原題目】\n{original_question}\n\n"
        f"【建議】\n{comment}\n"
    )
    text, out_tokens, in_tokens = generate_content_with_tokens(REFINE_MODEL_NAME, msg)
    return text, out_tokens, in_tokens