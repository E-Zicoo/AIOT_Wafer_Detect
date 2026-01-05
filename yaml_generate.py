import pandas as pd
import os
import yaml

# --- 設定區 ---
CONFIG_DIR = "kfold_configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

CSV_FILES = [f'training_{i}.csv' for i in range(1, 6)]
BASE_YAML = "data.yaml"

def create_yaml(fold_idx, train_txt_rel_path, val_txt_rel_path, nc, names):
    """
    產生 YAML 檔
    注意：train 和 val 的路徑是相對於 path 的
    """
    yaml_content = {
        'path': '.',  # 【關鍵】設定為當前目錄，這樣相對路徑才有用
        'train': train_txt_rel_path, # 指向 kfold_configs/train_fold_X.txt
        'val': val_txt_rel_path,     # 指向 kfold_configs/val_fold_X.txt
        'nc': nc,
        'names': names
    }

    # 儲存 yaml
    yaml_path = os.path.join(CONFIG_DIR, f'wafer_fold_{fold_idx}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    return yaml_path

def main():
    # 1. 讀取 data.yaml 取得類別資訊
    if not os.path.exists(BASE_YAML):
        print("錯誤：找不到 data.yaml")
        return

    with open(BASE_YAML, 'r') as f:
        base_config = yaml.safe_load(f)
        nc = base_config.get('nc')
        names = base_config.get('names')
        print(f"類別設定: {names} (共 {nc} 類)")

    # 2. 製作 5-Fold
    for i in range(5):
        fold_idx = i + 1

        # 定義 CSV
        val_csv = CSV_FILES[i]
        train_csvs = [f for j, f in enumerate(CSV_FILES) if j != i]

        # --- 讀取路徑 (直接用 CSV 裡的字串，不加 os.getcwd) ---
        val_df = pd.read_csv(val_csv)
        val_imgs = val_df['sample'].tolist() # 直接拿相對路徑

        train_df = pd.concat([pd.read_csv(f) for f in train_csvs])
        train_imgs = train_df['sample'].tolist() # 直接拿相對路徑

        # --- 寫入 txt (存到 kfold_configs 資料夾) ---
        # 為了讓路徑整潔，我們在 txt 檔名也用相對路徑
        train_txt_name = f'train_fold_{fold_idx}.txt'
        val_txt_name = f'val_fold_{fold_idx}.txt'

        train_txt_full_path = os.path.join(CONFIG_DIR, train_txt_name)
        val_txt_full_path = os.path.join(CONFIG_DIR, val_txt_name)

        with open(train_txt_full_path, 'w') as f:
            f.write('\n'.join(train_imgs))

        with open(val_txt_full_path, 'w') as f:
            f.write('\n'.join(val_imgs))

        # --- 產生 yaml ---
        # 這裡告訴 YAML：txt 檔在哪裡 (相對於專案根目錄)
        train_rel_path_for_yaml = os.path.join(CONFIG_DIR, train_txt_name) # e.g. kfold_configs/train_fold_1.txt
        val_rel_path_for_yaml = os.path.join(CONFIG_DIR, val_txt_name)

        create_yaml(fold_idx, train_rel_path_for_yaml, val_rel_path_for_yaml, nc, names)
        print(f"Fold {fold_idx} 設定完成 -> {train_rel_path_for_yaml}")

if __name__ == "__main__":
    main()