import csv


INPUT_CSV_PATH='/LDAP_home/gyzhou-c/local/MIN_MAX/Final_Project/test.csv'
with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader) # 跳過標題
    samples=[]
    for row in reader:
        if row:
            real=row[0]
            sample=row[0].replace('train','LQ').replace('valid','LQ').replace('test','LQ')
            label=row[-1]
            samples.append((sample, real, label))
with open('d_test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sample', 'real', 'label'])
    for sample, real, label in samples:
        writer.writerow([sample, real, label])


