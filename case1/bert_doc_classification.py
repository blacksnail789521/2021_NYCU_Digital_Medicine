from bert_document_classification.models import ObesityPhenotypingBert
import pandas as pd
import os

test_dir = './Data/Validation'
model = ObesityPhenotypingBert(device='cuda', batch_size=10)

def predict_testset(model):
    test_data_df = pd.DataFrame(columns=('Filename', 'text'))
    output_csv = pd.DataFrame(columns=('Filename', 'Obesity'))

    cnt = 0
    data = []
    for path in os.listdir(test_dir):
        with open(os.path.join(test_dir, path), 'r') as f:
            text = f.read()
            data.append(text)
            test_data_df.loc[cnt] = [path, text]
            cnt += 1

    predictions = model.predict(data).transpose(0, 1)
    pred_obesity = []
    for pred in predictions:
        pred_obesity.append(int(pred[4].item()))

    output_csv['Filename'] = test_data_df['Filename']
    output_csv['Obesity'] = pred_obesity
    output_csv = output_csv.sort_values(by=['Filename'])
    output_csv.to_csv('./output.csv', index=0)

if __name__ == "__main__":
    predict_testset(model)