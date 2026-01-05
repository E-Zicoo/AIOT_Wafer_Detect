from turtle import shape, width
import cv2
import csv
import cv2
import os

def read_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    #print("Original Image Shape:", image.shape)
    return image.shape
def resize_saving(image_path, size=(1024, 1024)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    resized_image = cv2.resize(image, size)
    cv2.imwrite(image_path, resized_image)
    #print(f"Resized image saved to: {save_path}")

helight=0
width=0
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/training_1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
        #resize_saving(image_path, size=(1024, 1024))
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/training_2.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
        #resize_saving(image_path, size=(1024, 1024))
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/training_3.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
        #resize_saving(image_path, size=(1024, 1024))
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/training_4.csv','r')as f:
    reader=csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
        #resize_saving(image_path, size=(1024, 1024))
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/training_5.csv','r')as f:
    reader=csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
        #resize_saving(image_path, size=(1024, 1024))
with open('/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/test.csv','r')as f:
    reader=csv.reader(f)
    next(reader)  # Skip header row if there is one
    for row in reader:
        image_path = row[0]  # Assuming the image path is in the first column
        shapeimg = read_image(image_path)
        if shapeimg:
            if helight < shapeimg[0]:
                helight = shapeimg[0]
            if width < shapeimg[1]:
                width = shapeimg[1]
print(f'Maximum Height: {helight}, Maximum Width: {width}')