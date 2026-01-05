import os
import csv
import cv2
import pandas as pd
from ultralytics import YOLO

# ================= 設定區 =================
# 模型路徑
MODEL_PATH = "/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/Wafer_5Fold_Result/fold_3/weights/training_3.pt"

# 輸入圖片的 CSV 索引檔
INPUT_CSV_PATH = "/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/test.csv"

# 圖片根目錄
IMAGE_ROOT_DIR = "/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/"

# 輸出設定
OUTPUT_IMAGE_FOLDER = "prediction_images"  # 存圖片的資料夾
OUTPUT_DATA_CSV = "prediction_coordinates.csv" # 存座標的 CSV 檔名

# 批次大小 (根據你的 VRAM 大小調整，通常設 4, 8, 16, 32)
# 設越大跑越快，但記憶體吃越多
BATCH_SIZE = 8
# =========================================

def main():
    # 1. 準備環境
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
        print(f"📁 已建立圖片輸出資料夾: {OUTPUT_IMAGE_FOLDER}")

    # 2. 載入模型
    print(f"🚀 正在載入模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 3. 從 CSV 讀取所有圖片路徑
    all_image_paths = []
    print(f"📖 正在讀取輸入清單: {INPUT_CSV_PATH} ...")

    try:
        with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # 跳過標題
            for row in reader:
                if row:
                    # 組合完整路徑
                    full_path = os.path.join(IMAGE_ROOT_DIR, row[0])
                    # 檢查檔案是否存在，存在的才加入清單
                    if os.path.exists(full_path):
                        all_image_paths.append(full_path)
                    else:
                        print(f"⚠️ 找不到檔案，略過: {full_path}")
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return

    total_images = len(all_image_paths)
    print(f"🔍 共發現 {total_images} 張有效圖片，準備進行批次預測 (Batch Size: {BATCH_SIZE})...")

    # 用來暫存要寫入 CSV 的數據
    csv_results = []

    # 4. 批次處理迴圈
    # range(start, stop, step) -> 0, 8, 16, 24...
    for i in range(0, total_images, BATCH_SIZE):
        # 取出這一批的路徑 (例如 index 0~7)
        batch_paths = all_image_paths[i : i + BATCH_SIZE]

        print(f"⚡ 正在處理第 {i+1} ~ {min(i+BATCH_SIZE, total_images)} 張圖片...")

        try:
            # --- 核心預測 (一次丟 List 進去) ---
            results = model.predict(
                batch_paths,
                imgsz=1024,
                conf=0.4,
                verbose=False,
                stream=False, # 這裡設 False 以便我們直接拿到整個 List,

            )

            # --- 處理這一批的結果 ---
            for result in results:
                # A. 取得檔名
                file_name = os.path.basename(result.path)

                # B. 畫圖並存檔
                plotted_img = result.plot()
                save_img_path = os.path.join(OUTPUT_IMAGE_FOLDER, file_name)
                cv2.imwrite(save_img_path, plotted_img)

                # C. 提取座標數據 (給 LLM 用)
                # 檢查是否有偵測到物件
                if result.boxes:
                    for j, box in enumerate(result.boxes):
                        # 1. 類別與信心
                        cls_id = int(box.cls[0])
                        cls_name = result.names[cls_id]
                        conf = round(float(box.conf[0]), 3)

                        # 2. BBox 座標 [x1, y1, x2, y2]
                        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

                        # 3. Segmentation 座標 (如果有 mask 模型)
                        seg_coords = "N/A"
                        if result.masks is not None:
                            # 取得多邊形座標
                            coords = result.masks.xy[j].astype(int).tolist()
                            # 轉字串以免 CSV 格式亂掉
                            seg_coords = str(coords)

                        # 加入資料表
                        csv_results.append({
                            "Image_Name": file_name,
                            "Class": cls_name,
                            "Confidence": conf,
                            "BBox_xyxy": str(bbox),
                            "Polygon_Points": seg_coords
                        })
                else:
                    # 沒偵測到東西，也可以選擇要不要紀錄
                    csv_results.append({
                        "Image_Name": file_name,
                        "Class": "No Detection",
                        "Confidence": 0,
                        "BBox_xyxy": "",
                        "Polygon_Points": ""
                    })

        except Exception as e:
            print(f"❌ 批次處理發生錯誤: {e}")
            continue

    # 5. 將所有數據存成 CSV
    if csv_results:
        print(f"💾 正在儲存分析結果至 {OUTPUT_DATA_CSV} ...")
        df = pd.DataFrame(csv_results)
        df.to_csv(OUTPUT_DATA_CSV, index=False, encoding='utf-8-sig')
        print("✅ 全部完成！")
    else:
        print("⚠️ 沒有產生任何數據結果。")

if __name__ == "__main__":
    main()