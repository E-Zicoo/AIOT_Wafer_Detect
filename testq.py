import google.generativeai as genai
import os

# 記得換成你的 API Key
API_KEY = "AIzaSyDe2g1PpAQ8nEk6zAe9Ab6HQPYyfM4l3G4"

genai.configure(api_key=API_KEY)

print("🔍 正在查詢可用模型...")
try:
    for m in genai.list_models():
        # 只列出支援生成內容 (generateContent) 的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"❌ 查詢失敗: {e}")