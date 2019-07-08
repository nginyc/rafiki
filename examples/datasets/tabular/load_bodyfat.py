from examples.datasets.tabular.csv_file import load

# Loads the "Body Fat" CSV dataset from `http://course1.winona.edu/bdeppa/Stat%20425/Datasets.html` for the `TABULAR_CLASSIFICATION` task
def load_body_fat():
    load(
        dataset_url='https://course1.winona.edu/bdeppa/Stat%20425/Data/bodyfat.csv',
        out_train_dataset_path='data/bodyfat_train.zip',
        out_test_dataset_path='data/bodyfat_test.zip',
        out_meta_txt_path='data/bodyfat_meta.txt',
        features=['density',
                  'age',
                  'weight',
                  'height',
                  'neck',
                  'chest',
                  'abdomen',
                  'hip',
                  'thigh',
                  'knee',
                  'ankle',
                  'biceps',
                  'forearm',
                  'wrist'],
        target='bodyfat'
    )
    
if __name__ == '__main__':
    load_body_fat()    
    