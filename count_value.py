import os
from re import L
import matplotlib.pyplot as plt


def transpose(category):
    transposed_category={
        0:'BLOCK ETCH',
        1:'COATTING BAD',
        2:'PARTICLE',
        3:'PIQ PARTICLE',
        4:'PO CONTAMINATION',
        5:'SCRATCH',
        6:'SEZ BURNT'
    }


def parse_filename(filename):

    with open(filename, 'r') as file:
        lines = file.readline().strip()
        label=lines.split(' ')[0]
        #print(label)
    return label
if __name__ == '__main__':

    train='train/labels'
    valid='valid/labels'
    test='test/labels'

    tr_category={
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
    }
    va_category={
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
    }
    te_category={
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
    }
    tr=os.listdir(train)
    for label in tr:
        label_path=os.path.join(train,label)
        category_id=parse_filename(label_path)
        tr_category[int(category_id)].append(label_path)
    va=os.listdir(valid)
    for label in va:
        label_path=os.path.join(valid,label)
        category_id=parse_filename(label_path)
        va_category[int(category_id)].append(label_path)
    te=os.listdir(test)
    for label in te:
        label_path=os.path.join(test,label)
        category_id=parse_filename(label_path)
        te_category[int(category_id)].append(label_path)

    print('Train set:')
    for key in tr_category.keys():
        print(f'Category {key}: {len(tr_category[key])} samples')
    print('Validation set:')
    for key in va_category.keys():
        print(f'Category {key}: {len(va_category[key])} samples')
    print('Test set:')
    for key in te_category.keys():
        print(f'Category {key}: {len(te_category[key])} samples')

    # Visualize the distribution of samples in each category
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.bar(tr_category.keys(), [len(tr_category[key]) for key in tr_category.keys()])
    plt.title('Train Set')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')

    plt.subplot(1, 3, 2)
    plt.bar(va_category.keys(), [len(va_category[key]) for key in va_category.keys()])
    plt.title('Validation Set')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')

    plt.subplot(1, 3, 3)
    plt.bar(te_category.keys(), [len(te_category[key]) for key in te_category.keys()])
    plt.title('Test Set')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')

    plt.tight_layout()
    plt.savefig('category_distribution.png')
    #sample,yolo_label,labels