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

if __name__ == '__main__':
    train_df, valid_df = get_dataset(train_dir, valid_dir)
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    model = LogisticRegression(solver='liblinear', C=10, penalty='l2')

    df = pd.concat([train_df, valid_df])
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.33)

    x_train_tfidf = vectorizer.fit_transform(x_train)
    model.fit(x_train_tfidf, np.array(y_train, dtype=int))

    x_test_tfidf = vectorizer.transform(x_test)
    pred = model.predict(x_test_tfidf)

    print(f"accuracy score: {accuracy_score(y_test.tolist(), pred)}")
    print(f"report:\n{classification_report(y_test.tolist(), pred)}")
