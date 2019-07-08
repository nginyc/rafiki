from examples.datasets.corpus.ptb import load

# Loads the Penn Treebank sample dataset for the `POS_TAGGING` task
def load_sample_ptb(out_train_dataset_path='data/ptb_for_pos_tagging_train.zip',
                    out_val_dataset_path='data/ptb_for_pos_tagging_val.zip',
                    out_meta_tsv_path='data/ptb_for_pos_tagging_meta.tsv'):
    load(
        dataset_url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip',
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_meta_tsv_path=out_meta_tsv_path
    )

if __name__ == '__main__':
    load_sample_ptb()    
    