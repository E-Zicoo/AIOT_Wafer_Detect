import argparse
import datetime
import os
import cv2
import pandas as pd
from ultralytics import YOLO
import json
import csv
import torch
import numpy as np  # 新增 numpy
from PIL import Image
import torchvision.transforms as transforms
import importlib
import datetime
def read_csv(input_csv_path):

    with open(input_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        image_paths = [row['sample'] for row in reader]
    return image_paths
def write_csv(message, output_csv):
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=message.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(message)
def load_config(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)['hyper_parameters']
    return {}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using YOLO model and process with diffusion model.")
    parser.add_argument("--yolo_model", type=str, default='model/cls.pt', help="Path to the YOLO model.")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold for YOLO detections.")
    parser.add_argument("--i_csv", type=str, required=True, help="Root directory of images.")
    parser.add_argument("--o_HQ", type=str, default="result/img_HQ", help="Folder to save high-quality images.")
    parser.add_argument("--o_seg", type=str, default="result/img_seg", help="Folder to save segmentation/classification images.")
    parser.add_argument("--o_csv", type=str, default="result/prediction_coordinates.csv", help="CSV file to save prediction coordinates.")
    parser.add_argument("--b", type=int, default=8, help="Batch size for prediction.")
    parser.add_argument("--diffusion_model", type=str, default='diffusion/fold_1/model_fold_1.pth', help="Path to the diffusion model.")
    parser.add_argument("--json", type=str, default='diffusion/hyper_parameter.json', help="Path to the JSON parameter file.")
    parser.add_argument("--diffusion_structure", type=str, default='model/gan.py', help="Root directory of images.")
    parser.add_argument("--resize", type=int, default=0, help="Resize final output images (0 = keep original).")
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.o_HQ=f'{args.o_HQ}_{current_time}'

    args.o_seg=f'{args.o_seg}_{current_time}'
    if not os.path.exists(args.o_HQ):
        os.makedirs(args.o_HQ)

    if not os.path.exists(args.o_seg):
        os.makedirs(args.o_seg)

    imgs=read_csv(args.i_csv)
    hyper_parameters = load_config(args.json)
    img_size=hyper_parameters["image_size"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO(args.yolo_model)
    model_structure= args.diffusion_structure.split('/')[-1].replace('.py','')
    module = importlib.import_module(f"model.{model_structure}")
    UNetDenoise = module.UNetDenoise
    gan_model= UNetDenoise(image_size=img_size,dropout=0.0).to(device)
    gan_model.load_state_dict(torch.load(args.diffusion_model, map_location=device))
    gan_model.eval()
    #print(imgs)
    origin_size = [
    tuple(cv2.imread(img).shape[:2])
    for img in imgs
    ]
    #print(origin_size)

    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

args.o_csv = args.o_csv.replace('.csv', f'_{current_time}.csv')

with torch.no_grad():
    for idx, img_path in enumerate(imgs):
        img_name = os.path.basename(img_path)

        # --- 1️⃣ GAN 預測 ---
        img_bgr = cv2.imread(img_path)  # BGR
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # RGB
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        HQ = gan_model(input_tensor)
        HQ_numpy = HQ.squeeze().permute(1, 2, 0).cpu().numpy()
        HQ_numpy = np.clip(HQ_numpy, 0, 1)
        HQ_numpy = (HQ_numpy * 255).astype(np.uint8)

        # resize GAN 輸出到原圖尺寸
        orig_h, orig_w = origin_size[idx]
        HQ_resized = cv2.resize(HQ_numpy, (orig_w, orig_h))
        HQ_rgb = HQ_resized  # 保持 RGB 通道
        save_HQ_path = os.path.join(args.o_HQ, f"pred_{img_name}")
        # 存彩圖 (BGR)
        cv2.imwrite(save_HQ_path, cv2.cvtColor(HQ_rgb, cv2.COLOR_RGB2BGR))

        # --- 2️⃣ YOLO 預測 (固定 1024x1024) ---
        yolo_input = cv2.resize(HQ_rgb, (1024, 1024))
        yolo_results = yolo_model.predict(
            source=yolo_input,
            imgsz=1024,
            conf=args.confidence,
            device=device,
            verbose=False
        )

        # --- 3️⃣ 畫框並存圖 ---
        # 先轉 BGR，確保顏色正確
        HQ_draw = cv2.cvtColor(yolo_input, cv2.COLOR_RGB2BGR).copy()

        for result in yolo_results:
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = round(float(box.conf[0]), 4)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # YOLO 輸入就是 1024x1024，不需縮放
                    cv2.rectangle(HQ_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(HQ_draw, f"{cls_name} {conf}", (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 寫入 CSV
                    write_csv(
                        {
                            "Image_Name":f"yolo_{img_name}",
                            "Class": cls_name,
                            "Confidence": conf,
                            "BBox_xyxy": f"[{x1},{y1},{x2},{y2}]"
                        },
                        args.o_csv
                    )

            else:
                # 沒偵測到物件也記錄
                write_csv(
                    {
                        "Image_Name":f"yolo_{img_name}",
                        "Class": "No Detection",
                        "Confidence": 0,
                        "BBox_xyxy": ""
                    },
                    args.o_csv
                )

        # 存畫框圖
        save_pred_path = os.path.join(args.o_seg, f"yolo_{img_name}")
        cv2.imwrite(save_pred_path, HQ_draw)
