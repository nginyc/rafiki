from examples.datasets.tabular.csv_file import load

# Loads the "Titantic" CSV dataset from `https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html` for the `TABULAR_REGRESSION` task
def load_titanic():
    load(
        dataset_url='https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
        out_train_dataset_path='data/titanic_train.csv',
        out_val_dataset_path='data/titanic_val.csv'
    )


if __name__ == '__main__':
    load_titanic()    
    