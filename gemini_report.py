from ast import arg
import os
import time
import glob
import pandas as pd
import google.generativeai as genai
from PIL import Image
from google.api_core.exceptions import ResourceExhausted, NotFound, ServiceUnavailable
import argparse
import base64
from io import BytesIO
import os
from PIL import Image
from io import BytesIO
import base64

def save_dialog_html(report_text, stats_summary, images_info, output_dir,timer):
    """
    舊版圖片排列，文字用對話氣泡
    """
    html_content = f"""
<html>
<head>
<meta charset="utf-8">
<title>Wafer Defect 對話模式報告</title>
<style>
body {{ font-family: Arial, sans-serif; background: #f0f2f5; padding: 20px; }}
h1 {{ color: #2c3e50; text-align: center; }}

.chat-container {{ max-width: 900px; margin: auto; display: flex; flex-direction: column; }}
.message {{ padding: 10px 15px; margin: 10px 0; border-radius: 15px; max-width: 70%; }}
.user {{ background: #3498db; color: white; align-self: flex-start; }}
.bot {{ background: #ecf0f1; color: #2c3e50; align-self: flex-end; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}

.img-grid {{ display: flex; flex-wrap: wrap; margin-top: 10px; }}
.img-container {{ display: inline-block; text-align: center; margin: 5px; }}
.img-container img {{ width: 64px; height: 64px; border-radius: 5px; border: 1px solid #34495e; }}
.caption {{ font-size: 0.8em; color: #2c3e50; text-align: center; }}
</style>
</head>
<body>
<h1>Wafer Defect Today's report</h1>
<div class="chat-container">



<!-- 統計摘要 -->
<div class="message bot">
<pre>
{stats_summary}
</pre>
</div>

<!-- 模型回覆文字 -->
<div class="message bot">
<pre>
{report_text}
</pre>
</div>

<!-- 圖片區域 -->
<h2>報告依據圖片</h2>
<div class="img-grid">
"""

    # 加入圖片 (縮成 64x64 Base64)
    for img_path, cls, conf in images_info:
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        img.thumbnail((64,64))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        html_content += f"""
<div class="img-container">
    <img src="data:image/png;base64,{img_b64}" alt="{cls}">
    <div class="caption">{cls} ({conf:.2f})</div>
</div>
"""

    html_content += """
</div> <!-- end img-grid -->
</div> <!-- end chat-container -->
</body>
</html>
"""

    os.makedirs(output_dir, exist_ok=True)
    report_filename = os.path.join(output_dir, f"Production_Report_dialog_{timer}.html")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" HTML 報告已儲存: {report_filename}")

def get_latest_files():
    """自動尋找最新的 CSV 和對應的圖片資料夾"""
    latest_csv = glob.glob(os.path.join(args.base_dir, args.input_csv))
    latest_img_dir = glob.glob(os.path.join(args.base_dir, args.image_dir))
    if not latest_csv:
        raise FileNotFoundError("找不到任何預測 CSV 檔案。")
    if not latest_img_dir:
        raise FileNotFoundError("找不到圖片資料夾。")
    return latest_csv[0], latest_img_dir[0]  # 直接回傳字串，方便後續使用

def generate_wafer_report(csv_path, img_dir):
    genai.configure(api_key=args.api_key)

    timer='_'.join(img_dir.split('_')[2:])
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ CSV 是空的，無法生成報告。")
        return

    total_wafers = df['Image_Name'].nunique()
    defect_counts = df['Class'].value_counts()
    most_frequent_defect = defect_counts.idxmax()
    stats_summary = f"""
[生產數據摘要]
- 檢測總圖片數: {total_wafers}
- 偵測到的異常分佈:
{defect_counts.to_string()}
- 最頻繁的異常類型: {most_frequent_defect}
"""

    # 挑選圖片
    images_info = []  # 儲存 (img_path, class, conf)
    defect_classes = df[df['Class'] != 'No Detection']['Class'].unique()

    print("\n📸 正在挑選樣本圖片 (每個類別最多 20 張)...")
    for cls in defect_classes:
        cls_rows = df[df['Class'] == cls].sort_values(by='Confidence', ascending=False)
        top_rows = cls_rows.head(20)
        for _, row in top_rows.iterrows():
            img_name = row['Image_Name']
            conf = row['Confidence']
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                images_info.append((img_path, cls, conf))
                print(f" - 加入 {cls} 範例圖: {img_name} (Conf: {conf})")
            else:
                print(f" 找不到圖片: {img_path}")

    if not images_info:
        print("沒有偵測到瑕疵圖片，僅使用文字生成報告。")

    # Prompt
    prompt = f"""
你現在是一位資深的半導體製程與設備工程師。請根據以下提供的「Wafer Defect 自動光學檢測 (AOI)」數據與圖片，撰寫一份今日的生產異常報告。

【數據資訊】
{stats_summary}

請用繁體中文回答。
"""

    target_models = ['gemini-2.5-flash']
    print("\n 正在請求 Gemini 生成報告...")

    success = False
    report_text = ""
    for model_name in target_models:
        if success: break
        print(f" 嘗試模型: {model_name} ...")
        model = genai.GenerativeModel(model_name)

        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt])
                report_text = response.text
                success = True
                break

            except (NotFound, ResourceExhausted, ServiceUnavailable) as e:
                print(f" 錯誤 ({type(e).__name__})，等待 {retry_delay} 秒重試...")
                time.sleep(retry_delay)
                retry_delay *= 2

            except Exception as e:
                print(f" 未知錯誤 ({model_name}): {e}")
                break

    if not success:
        print("\n 生成失敗，將僅生成 HTML 報告文字部分。")

    save_dialog_html(report_text, stats_summary, images_info, "result", timer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wafer report.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--base_dir", type=str, default="result", help="Base directory for resources.")
    parser.add_argument("--api_key", type=str, required=True, help="Google Gemini API Key.")
    parser.add_argument("--output_dir", type=str, default='result', help="Directory to save the output reports.")
    args = parser.parse_args()

    latest_csv, latest_img_dir = get_latest_files()
    print(f"讀取 CSV: {latest_csv}\n讀取圖片: {latest_img_dir}")

    generate_wafer_report(latest_csv, latest_img_dir)
