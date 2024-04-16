import pickle
import pandas as pd

def load_data(file_path='data/test_set.csv'):
    return pd.read_csv(file_path, index_col='mk_CurrentCustomer')


def load_model(model_path='models/model.pkl'):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def predict(df, model, output_file_path='predictions/test_predictions.csv'):
    predictions = model.predict(df)
    if output_file_path:
        df_predictions = pd.DataFrame({
            'mk_CurrentCustomer': df.index,
            'Prediction': predictions
            }
        )
        df_predictions.to_csv(output_file_path, index=False)
    return predictions


if __name__ == '__main__':
    data = load_data()
    model = load_model()
    predict(data, model)
