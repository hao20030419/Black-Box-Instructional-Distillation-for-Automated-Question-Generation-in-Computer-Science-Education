"""
簡單的執行器：用來呼叫 `generator.augment_prompt_templates` 並把結果存成 `prompts.json`。

使用方法（在 repository 根目錄執行）：

	python -m GeminiAgent.agent.instruction_generator

確保已設定 `GEMINI_API_KEY` 環境變數或 `.env`。
"""
import os
import sys

# 如果直接在 `GeminiAgent/agent` 目錄下執行此腳本，確保專案根目錄在 sys.path，
# 以便可以使用 package-style import `GeminiAgent.agent.generator`。
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
	sys.path.insert(0, root)

from GeminiAgent.agent import generator


def main():
	out_path = os.path.join(os.path.dirname(__file__), "prompts.json")
	print("Augmenting prompt templates (calls Gemini API)...")
	templates = generator.augment_prompt_templates(target_total=100, per_template=4)
	generator.save_prompt_templates(out_path, templates)
	print(f"Saved {len(templates)} prompt templates to {out_path}")


if __name__ == "__main__":
	main()
