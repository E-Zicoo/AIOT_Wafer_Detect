# 建立環境
conda create -n wafer_yolo python=3.9 -y

# 啟動環境 (以後每次要訓練都要跑這一行)
conda activate wafer_yolo
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# 安裝 YOLOv8
pip install ultralytics

# 安裝 Roboflow (方便你下載資料集)
pip install roboflow

# 安裝一些常用的輔助套件
pip install opencv-python-headless matplotlib pandas