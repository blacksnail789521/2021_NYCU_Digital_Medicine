import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

def predict_testset(vectorizer, model):
    test_data_df = pd.DataFrame(columns=('Filename', 'text'))
    output_csv = pd.DataFrame(columns=('Filename', 'Obesity'))

    cnt = 0
    for path in os.listdir(test_dir):
        with open(os.path.join(test_dir, path), 'r') as f:
            test_data_df.loc[cnt] = [path, f.read()]
            cnt += 1
    
    x_test_tfidf = vectorizer.transform(test_data_df['text'])
    pred = model.predict(x_test_tfidf)

    output_csv['Filename'] = test_data_df['Filename']
    output_csv['Obesity'] = pred
    output_csv = output_csv.sort_values(by=['Filename'])
    output_csv.to_csv('./output.csv', index=0)

if __name__ == '__main__':
    train_df, valid_df = get_dataset(train_dir, valid_dir)
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    model = LogisticRegression(solver='liblinear', C=10, penalty='l2')

    df = pd.concat([train_df, valid_df])
    x_train, x_valid, y_train, y_valid = train_test_split(df['text'], df['label'], test_size=0.33)

    x_train_tfidf = vectorizer.fit_transform(x_train)
    model.fit(x_train_tfidf, np.array(y_train, dtype=int))

    x_valid_tfidf = vectorizer.transform(x_valid)
    pred = model.predict(x_valid_tfidf)

    print(f"accuracy score: {accuracy_score(y_valid.tolist(), pred)}")
    print(f"report:\n{classification_report(y_valid.tolist(), pred)}")

    predict_testset(vectorizer, model)
