import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv


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
def sampling_kfold(category_dict, k=5, random_state=1230):
    """
    Perform k-fold sampling on the given category dictionary.

    Parameters:
    category_dict (dict): A dictionary where keys are category IDs and values are lists of data points.
    k (int): Number of folds.
    random_state (int): Seed for random number generator.

    Returns:
    list: A list containing k dictionaries, each representing a fold.
    """
    from sklearn.model_selection import KFold
    import numpy as np

    folds = [dict() for _ in range(k)]
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    for category_id, data_points in category_dict.items():
        data_points = np.array(data_points)
        for fold_index, (_, test_indices) in enumerate(kf.split(data_points)):
            folds[fold_index].setdefault(category_id, []).extend(data_points[test_indices].tolist())

    return folds

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
        tr_category[int(category_id)].append(label_path)
    te=os.listdir(test)
    for label in te:
        label_path=os.path.join(test,label)
        category_id=parse_filename(label_path)
        te_category[int(category_id)].append(label_path)

    folds=sampling_kfold(tr_category,k=5,random_state=1230)

    for i, fold in enumerate(folds):
        print(f"Fold {i+1}:")
        for category_id, data_points in fold.items():
            print(f" Category {category_id}: {len(data_points)} samples")
            with open(f'training_{i+1}.csv', 'a') as f:

                f=csv.writer(f)
                for data_point in data_points:
                    yolo_png='.'.join(data_point.split('.')[:-1])+'.jpg'
                    yolo_png=yolo_png.replace('labels','images')
                    f.writerow([yolo_png,data_point,category_id])
    print('Test set:')
    for key in te_category.keys():
        print(f'Category {key}: {len(te_category[key])} samples')
    with open('test.csv', 'a') as f:
        f=csv.writer(f)
        for key in te_category.keys():
            data_points=te_category[key]
            for data_point in data_points:
                yolo_png='.'.join(data_point.split('.')[:-1])+'.jpg'
                yolo_png=yolo_png.replace('labels','images')
                f.writerow([yolo_png,data_point,key])


    for i in range(5):
        df = pd.read_csv(f'training_{i+1}.csv')

        # 打亂列順序
        df = df.sample(frac=1, random_state=1230).reset_index(drop=True)

        # 存回原檔（或換檔名）
        df.to_csv(f'training_{i+1}.csv', index=False)
    df= pd.read_csv('test.csv')
    df = df.sample(frac=1, random_state=1230).reset_index(drop=True)
    df.to_csv('test.csv', index=False)


    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    category_names = {
        0: 'BLOCK ETCH',
        1: 'COATTING BAD',
        2: 'PARTICLE',
        3: 'PIQ PARTICLE',
        4: 'PO CONTAMINATION',
        5: 'SCRATCH',
        6: 'SEZ BURNT'
    }

    categories = list(category_names.keys())

    Y_MIN, Y_MAX = 0, 400

    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1.1])

    # ---------- Training folds ----------
    for i, fold in enumerate(folds):
        counts = [len(fold.get(c, [])) for c in categories]

        ax = fig.add_subplot(gs[0, i])
        bars = ax.bar(categories, counts)
        ax.set_title(f"Train Fold {i+1}")
        ax.set_ylim(Y_MIN, Y_MAX)

        ax.set_xticks(categories)
        ax.set_xticklabels(
            [category_names[c] for c in categories],
            rotation=45, ha='right'
        )

        if i == 0:
            ax.set_ylabel("Number of Samples")

        # 標數值
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count + 5,
                str(count),
                ha='center',
                va='bottom',
                fontsize=9
            )

    # ---------- Test set ----------
    test_counts = [len(te_category[c]) for c in categories]

    ax_test = fig.add_subplot(gs[1, :])
    bars = ax_test.bar(categories, test_counts)
    ax_test.set_title("Test Set")
    ax_test.set_ylim(Y_MIN, Y_MAX)
    ax_test.set_ylabel("Number of Samples")

    ax_test.set_xticks(categories)
    ax_test.set_xticklabels(
        [category_names[c] for c in categories],
        rotation=45, ha='right'
    )

    # 標數值
    for bar, count in zip(bars, test_counts):
        ax_test.text(
            bar.get_x() + bar.get_width() / 2,
            count + 5,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10
        )



    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("kfold_train_test_class_hist_fixed_ylim.png", dpi=300)
    plt.show()