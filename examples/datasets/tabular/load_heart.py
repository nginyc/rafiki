from examples.datasets.tabular.csv_file import load

# Loads the "Heart Disease UCI" CSV dataset from kaggle for the `TABULAR_REGRESSION` task
# Install Kaggle API. 
# Then from Rafiki root folder, run: `kaggle datasets download ronitf/heart-disease-uci -p data --unzip` to download the `heart.csv` file to `rafiki/data` folder.

def load_heart():
    load(
        dataset_url='data/heart.csv',
        out_train_dataset_path='data/heart_train.csv',
        out_val_dataset_path='data/heart_val.csv'
    )


if __name__ == '__main__':
    load_heart()    
    