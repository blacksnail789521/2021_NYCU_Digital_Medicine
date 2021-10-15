import pandas as pd
import numpy as np
import os

output_dir = './Data/Output'
test_dir = './Data/Validation'
    
def voting():
    df = pd.DataFrame(columns=('logistic', 'adaboost',  'bert_l2', 'mlpclassifier', 'bert_30'))
    df['logistic'] = pd.read_csv(os.path.join(output_dir, 'logistic.csv'))['Obesity']
    df['adaboost'] = pd.read_csv(os.path.join(output_dir, 'adaboost.csv'))['Obesity']
    df['bert_l2'] = pd.read_csv(os.path.join(output_dir, 'bert_l2.csv'))['Obesity']
    df['mlpclassifier'] = pd.read_csv(os.path.join(output_dir, 'mlpclassifier.csv'))['Obesity']
    df['bert_30'] = pd.read_csv(os.path.join(output_dir, 'bert_30.csv'))['Obesity']

    lst = []
    for i in range(df.shape[0]):
        sum = 0
        for j in range(df.shape[1]-1):
            sum += df.iloc[i, j]
        if sum > (df.shape[1]/2):
            lst.append(1)
        else:
            lst.append(0)
    return lst

def predict_testset(result):
    test_data_df = pd.DataFrame(columns=('Filename', 'text'))
    output_csv = pd.DataFrame(columns=('Filename', 'Obesity'))

    cnt = 0
    for path in os.listdir(test_dir):
        with open(os.path.join(test_dir, path), 'r') as f:
            test_data_df.loc[cnt] = [path, f.read()]
            cnt += 1

    output_csv['Filename'] = test_data_df['Filename']
    output_csv['Obesity'] = result
    output_csv = output_csv.sort_values(by=['Filename'])
    output_csv.to_csv('./output.csv', index=0)

if __name__ == '__main__':
    result = voting()
    predict_testset(result)