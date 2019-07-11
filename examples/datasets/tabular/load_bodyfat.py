from examples.datasets.tabular.csv_file import load

# Loads the "Body Fat" CSV dataset from `http://course1.winona.edu/bdeppa/Stat%20425/Datasets.html` for the `TABULAR_CLASSIFICATION` task
def load_body_fat():
    load(
        dataset_url='https://course1.winona.edu/bdeppa/Stat%20425/Data/bodyfat.csv',
        out_train_dataset_path='data/bodyfat_train.csv',
        out_val_dataset_path='data/bodyfat_val.csv'
    )
    
if __name__ == '__main__':
    load_body_fat()    
    