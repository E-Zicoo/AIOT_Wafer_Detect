from ultralytics import YOLO
import os
import glob
import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt

# --- 參數設定 ---
PROJECT_NAME = "Wafer_5Fold_Result"
CONFIG_DIR = "kfold_configs"  # 確保這跟您 prepare 腳本產生的資料夾名稱一致
IMG_SIZE = 1024
BATCH_SIZE = 32
WORKERS = 4
EPOCHS = 300          # 設定 200 即可
lr = 1e-3

def plot_loss_with_values(results_csv_path, save_dir, fold_idx):
    """
    讀取 YOLO 的 results.csv，畫出 Box, Seg, Cls Loss，
    並在 Legend 顯示最終數值。
    """
    try:
        # 讀取 CSV (有些版本 header 有空格，有些沒有，這裡統一處理)
        df = pd.read_csv(results_csv_path)
        df.columns = [c.strip() for c in df.columns]

        # 準備畫布 (3個子圖：Box Loss, Seg Loss, Cls Loss)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 定義要畫的 Loss 類型 (Train vs Val)
        loss_types = [
            ("train/box_loss", "val/box_loss", "Box Loss"),
            ("train/seg_loss", "val/seg_loss", "Seg Loss"),
            ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ]

        # ★★★ 修正這裡：原本寫成 ep3ochs ★★★
        epochs = df["epoch"]

        for ax, (train_col, val_col, title) in zip(axes, loss_types):
            # 檢查欄位是否存在
            if train_col not in df.columns or val_col not in df.columns:
                ax.set_title(f"{title} (Not Found)")
                continue

            # 取得最後一筆數值
            final_train = df[train_col].iloc[-1]
            final_val = df[val_col].iloc[-1]

            # 畫線
            ax.plot(epochs, df[train_col], label=f"Train: {final_train:.4f}", color="blue")
            ax.plot(epochs, df[val_col], label=f"Val: {final_val:.4f}", color="orange")

            ax.set_title(f"Fold {fold_idx} - {title}", fontweight="bold")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Legend 帶有數值
            ax.legend(fontsize=11, loc="upper right", frameon=True, shadow=True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"Fold_{fold_idx}_Loss_Chart_Values.png")
        plt.savefig(save_path)
        plt.close()
        print(f"   --> ✅ 客製化 Loss 圖已產生: {save_path}")

    except Exception as e:
        # 如果失敗，印出詳細錯誤以便除錯
        print(f"   --> ❌ 畫 Loss 圖失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 檢查是否產生了設定檔
    yaml_files = sorted(glob.glob(os.path.join(CONFIG_DIR, "wafer_fold_*.yaml")))
    if not yaml_files:
        print(f"錯誤：在 {CONFIG_DIR} 找不到 yaml 檔，請先執行 prepare 腳本！")
        return

    summary_results = []
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    print(f"共發現 {len(yaml_files)} 個 Fold 設定檔")

    for i, yaml_file in enumerate(yaml_files):
        fold_idx = i + 1
        print(f"\n{'='*20} 訓練 Fold {fold_idx} {'='*20}")

        # 1. 載入模型 (建議改用 YOLO11)
        model = YOLO("yolo11n-seg.pt")

        # 2. 訓練
        model.train(
            data=yaml_file,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=f"fold_{fold_idx}",
            device=device,
            patience=50,      # 50 Epoch 沒進步就停，防止過擬合
            workers=WORKERS,
            save=True,
            exist_ok=True,
            plots=True,       # 這行必須為 True，才會產生 results.csv 讓我們畫圖

            # --- 推薦設定 (解決 Seg Loss 過高與過擬合) ---
            cos_lr=True,      # 餘弦退火 (必開)
            dropout=0.3,      # 先關閉 dropout，讓 SGD 發揮
            copy_paste=0.3,   # ★ 關鍵：針對晶圓瑕疵的強力增強

            mosaic=1.0,      # ★ 必加：馬賽克增強 (設為 1.0 代表 100% 使用)
            mixup=0.15,      # ★ 必加：混合增強 (讓兩張圖疊在一起，抗過擬合神技)
            degrees=5.0,     # 小幅度旋轉 (模擬晶圓擺放不正)
            fliplr=0.5,      # 左右翻轉 (晶圓通常對稱，這很有用)
            flipud=0.5,      # 上下翻轉
            # --- 移除 AdamW 與自訂 lr，使用預設 SGD ---
            # optimizer="AdamW",  <-- 移除
            # lr0=1e-4,           <-- 移除

            # Loss 權重
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

        # 3. 處理模型檔案
        weights_dir = os.path.join(PROJECT_NAME, f"fold_{fold_idx}", "weights")
        best_pt = os.path.join(weights_dir, "best.pt")
        last_pt = os.path.join(weights_dir, "last.pt")
        target_pt = os.path.join(weights_dir, f"training_{fold_idx}.pt")

        if os.path.exists(best_pt):
            os.rename(best_pt, target_pt)
            print(f"   --> 💾 模型已重新命名為: {target_pt}")

        if os.path.exists(last_pt):
            try:
                os.remove(last_pt)
            except:
                pass

        # 4. ★ 這裡執行畫圖 ★
        results_csv = os.path.join(PROJECT_NAME, f"fold_{fold_idx}", "results.csv")
        if os.path.exists(results_csv):
            plot_loss_with_values(
                results_csv, os.path.join(PROJECT_NAME, f"fold_{fold_idx}"), fold_idx
            )
        else:
            print(f"   --> ⚠️ 找不到 results.csv，無法畫圖 (路徑: {results_csv})")

        # 5. 驗證
        print(f"正在驗證 Fold {fold_idx}...")
        metrics = model.val(split="val")

        summary_results.append({
            "Fold": fold_idx,
            "Box_mAP50": metrics.box.map50,
            "Mask_mAP50": metrics.seg.map50,
            "Mask_mAP50-95": metrics.seg.map
        })

        # 6. 清理記憶體
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 輸出報告
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(PROJECT_NAME, "5_Fold_Final_Report.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n🎉 全部完成！總結報告已存於 {csv_path}")

if __name__ == "__main__":
    main()