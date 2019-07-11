from examples.datasets.corpus.ptb import load

# Loads the Penn Treebank sample dataset for the `POS_TAGGING` task
def load_sample_ptb(out_train_dataset_path='data/ptb_train.zip',
                    out_val_dataset_path='data/ptb_val.zip',
                    out_meta_tsv_path='data/ptb_meta.tsv',
                    validation_split=0.05):
    load(
        dataset_url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip',
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_meta_tsv_path=out_meta_tsv_path,
        validation_split=validation_split
    )

if __name__ == '__main__':
    load_sample_ptb()    
    