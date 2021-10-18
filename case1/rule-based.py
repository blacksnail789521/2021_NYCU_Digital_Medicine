import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

keyword_list = ['obese',  'obesity','asthma', 'atherosclerotic', 'cardiovascular','CAD', 'congestive', 'CHF', 'depression', 'diabetes''mellitus', 'DM', 'gallstones', 'cholecystectomy', 'gastroesophageal', 'reflux', 'GERD', 'gout', 'hypercholesterolemia', 'hypertension', 'HTN', 'hypertriglyceridemia', 'obstructive', 'apnea', 'OSA', 'osteoarthritis','OA', 'peripheral','vascular', 'PVD', 'venous','mg', 'MG']
section_categories = ['Diagnosis', 'Past or Present History of Illness', 'Social/Family History', 'Physical or Laboratory Examination', 'Medication.Disposition', 'Other']
train_dir = './Data/Train_Textual'
valid_dir = './Data/Test_Intuitive'
test_dir = './Data/Validation'

def get_dataset(train_dir, valid_dir):
    train_df = pd.DataFrame(columns=('text', 'label'))
    valid_df = pd.DataFrame(columns=('text', 'label'))

    cnt = 0
    for path in os.listdir(train_dir):
        with open(os.path.join(train_dir, path), 'r') as f:
            if path.startswith('U'):
                train_df.loc[cnt] = [f.read(), 0]
            else:
                train_df.loc[cnt] = [f.read(), 1]
            cnt += 1
    cnt = 0
    for path in os.listdir(valid_dir):
        with open(os.path.join(valid_dir, path), 'r') as f:
            if path.startswith('N'):
                valid_df.loc[cnt] = [f.read(), 0]
            else:
                valid_df.loc[cnt] = [f.read(), 1]
            cnt += 1
    
    return train_df, valid_df

def predict_testset_rule_based():
    test_data_df = pd.DataFrame(columns=('Filename', 'text'))
    output_csv = pd.DataFrame(columns=('Filename', 'Obesity'))

    cnt = 0
    for path in os.listdir(test_dir):
        with open(os.path.join(test_dir, path), 'r') as f:
            test_data_df.loc[cnt] = [path, f.read()]
            cnt += 1
    
    pred = []
    cnt = 0
    for i in range(len(test_data_df)):
        for j in range(len(keyword_list)):
            if keyword_list[j] in test_data_df.iloc[i,1]:
                pred.append(1)
                cnt+=1
                break
            elif j == len(keyword_list) - 1:
                pred.append(0)
                cnt+=1

    output_csv['Filename'] = test_data_df['Filename']
    output_csv['Obesity'] = pred
    output_csv = output_csv.sort_values(by=['Filename'])
    output_csv.to_csv('./output.csv', index=0)

def getKeywordCount(df):
    corpus = ''
    for i in range(len(df.iloc[:,0])):
        corpus+=df.iloc[i,0]

    counts = dict()
    words = corpus.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    sorted(counts.items(), key=lambda x:x[1])
    
    for keyword in keyword_list:
        print(keyword, ': ', counts.get(keyword))

if __name__ == '__main__':
    train_df, valid_df = get_dataset(train_dir, valid_dir)

    getKeywordCount(train_df)
    getKeywordCount(valid_df)

    # df = pd.concat([train_df, valid_df])
    # x_train, x_valid, y_train, y_valid = train_test_split(df['text'], df['label'], test_size=0.33)

    pred = []
    cnt = 0
    for i in range(len(train_df)):
        for j in range(len(keyword_list)):
            if keyword_list[j] in train_df.iloc[i,0]:
                pred.append(1)
                cnt+=1
                break
            elif j == len(keyword_list) - 1:
                pred.append(0)
                cnt+=1 
    print(f"train accuracy score: {accuracy_score(train_df.iloc[:,1].tolist(), pred)}")
    print(f"train report:\n{classification_report(train_df.iloc[:,1].tolist(), pred)}")

    pred = []
    cnt = 0
    for i in range(len(valid_df)):
        for j in range(len(keyword_list)):
            if keyword_list[j] in valid_df.iloc[i,0]:
                pred.append(1)
                cnt+=1
                break
            elif j == len(keyword_list) - 1:
                pred.append(0)
                cnt+=1
    print(f"valid accuracy score: {accuracy_score(valid_df.iloc[:,1].tolist(), pred)}")
    print(f"valid report:\n{classification_report(valid_df.iloc[:,1].tolist(), pred)}")    
    
    predict_testset_rule_based()