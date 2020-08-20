
from examples.datasets.tabular.csv_file import load

# Loads the "Pima Indian Diabetes" CSV dataset from `https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv` for the `TABULAR_CLASSIFICATION` task

def load_diabetes():
    load(
        dataset_url='https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
        out_train_dataset_path='data/diabetes_train.csv',
        out_val_dataset_path='data/diabetes_val.csv'
    )


if __name__ == '__main__':
    load_diabetes()    
    