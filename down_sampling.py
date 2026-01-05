
import cv2
import numpy as np
import csv
import os

def sensor_noise(img, shot_scale=30, read_sigma=0.01):
    img = img.astype(np.float32) / 255.0

    # Shot noise
    shot = np.random.poisson(img * shot_scale) / shot_scale

    # Read noise
    read = np.random.normal(0, read_sigma, img.shape)

    noisy = shot + read
    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)


# ===============================
# 低畫質相機模擬
# ===============================
def low_quality_camera_sim(img):
    h, w = img.shape[:2]

    # 1. Downsample
    scale = np.random.uniform(0.25, 0.5)
    img = cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR
    )

    # 2. Blur
    k = np.random.choice([3, 5])
    img = cv2.GaussianBlur(img, (k, k), 0)

    # 3. Sensor noise
    img = sensor_noise(
        img,
        shot_scale=np.random.uniform(20, 50),
        read_sigma=np.random.uniform(0.005, 0.02)
    )

    # 4. JPEG compression
    q = np.random.randint(10, 40)
    _, enc = cv2.imencode(
        ".jpg", img,
        [cv2.IMWRITE_JPEG_QUALITY, q]
    )
    img = cv2.imdecode(enc, 1)

    # 5. Upsample back
    img = cv2.resize(
        img,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    return img

def down_sample_image(input_image_path, output_image_path='/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/LQ'):
    # 讀取原始圖像
    img = cv2.imread(input_image_path)

    # 應用低畫質相機模擬
    low_quality_img = low_quality_camera_sim(img)
    basename = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_image_path, basename)
    # 儲存處理後的圖像
    cv2.imwrite(output_image_path, low_quality_img)
def process(csv_path):

    with open(csv_path, 'r',) as f:
        reader = csv.reader(f)
        header = next(reader) # 跳過標題
        for row in reader:
            if row:
                image_path=row[0]
                down_sample_image(image_path)

if __name__ == "__main__":
    csv_path=[
                '/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/runs/test.csv'
    ]
    for p in csv_path:
        process(p)
