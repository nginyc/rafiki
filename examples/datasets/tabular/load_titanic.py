from examples.datasets.tabular.csv_file import load

# Loads the "Titantic" CSV dataset from `https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html` for the `TABULAR_REGRESSION` task
def load_titanic():
    load(
        dataset_url='https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
        out_train_dataset_path='data/titanic_train.zip',
        out_test_dataset_path='data/titanic_test.zip',
        out_meta_txt_path='data/titanic_meta.txt',
        features=['Pclass','Sex','Age'],
        target='Survived'
    )


if __name__ == '__main__':
    load_titanic()    
    