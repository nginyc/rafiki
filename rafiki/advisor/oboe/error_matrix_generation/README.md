This is a quick example to show how to generate the error matrix from preprocessed datasets. It works on Unix and Linux but not on OS X for now.

# Dataset format

The datasets should be `csv` files. All the columns except the last are features; the last column is the class label.

# How to generate error and runtime matrices 
Run
```bash
bash start_matrix_generation.sh
```
It will create a `results` directory, with a subdirectory named by you (`matrix` by default) and containing results on individual datasets. We call this subdirectory the "csv directory".

Then an `error_matrix.csv` and a `runtime_matrix.csv` will be generated in the "csv directory", and move the csv files already merged into these matrices into `merged_csv_files`.

Finally the `error_matrix.csv` and a `runtime_matrix.csv` will be copied to `oboe/automl/defaults`.