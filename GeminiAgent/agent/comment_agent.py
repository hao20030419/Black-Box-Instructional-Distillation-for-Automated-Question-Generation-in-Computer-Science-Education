import os
from google import genai
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from dotenv import load_dotenv

"""
Comment Agent (部分 1)
- LLM 名稱：comment_LLM
- 功能：閱讀原始題目，提出可以修改的方向與具體建議
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

COMMENT_LLM_NAME = "comment_LLM"
COMMENT_MODEL_NAME = "gemini-3-pro-preview"

COMMENT_PROMPT = """
你是一位題目優化專家 (comment_LLM)。

請閱讀以下題目，並提出「可以修改的方向」與「具體優化建議」。
請只輸出建議本身，不要重寫題目。

請使用以下格式輸出：

建議：
1. ...
2. ...
3. ...
"""


def comment_question(original_question: str) -> str:
    """
    使用 comment_LLM 對原始題目給出改進建議
    """
    contents = f"{COMMENT_PROMPT}\n\n題目如下：\n{original_question}"
    text, out_tokens, in_tokens = generate_content_with_tokens(COMMENT_MODEL_NAME, contents)
    return text, out_tokens, in_tokens