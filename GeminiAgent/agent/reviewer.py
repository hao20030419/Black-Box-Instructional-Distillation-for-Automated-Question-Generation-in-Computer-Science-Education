import os
import json
from google import genai
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from dotenv import load_dotenv

"""
Reviewer Agent
- LLM 名稱：keep_or_not_LLM
- 功能：檢查題目是否合格，決定 keep / drop
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

REVIEW_LLM_NAME = "keep_or_not_LLM"
REVIEW_MODEL_NAME = "gemini-3-pro-preview"

REVIEW_PROMPT = """
你是審題專家 (keep_or_not_LLM)，請檢查題目是否清楚、選項合理、答案唯一、解析正確。

請以 JSON 格式回覆：

{
  "keep": true 或 false,
  "reason": "原因"
}
"""


def review_question(text: str):
    """
    使用 keep_or_not_LLM 審查題目
    回傳：(keep: bool, reason: str)
    """
    contents = f"{REVIEW_PROMPT}\n{text}"
    t, out_tokens, in_tokens = generate_content_with_tokens(REVIEW_MODEL_NAME, contents)

    try:
      j = json.loads(t[t.find("{"): t.rfind("}") + 1])
      return j.get("keep", False), j.get("reason", ""), out_tokens, in_tokens
    except Exception:
      return False, "JSON parse failed", out_tokens, in_tokens