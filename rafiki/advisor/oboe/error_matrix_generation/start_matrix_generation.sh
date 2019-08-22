# generate the error matrix and runtime matrix for each dataset
bash ../automl/generate_matrix.sh -p classification -m generate -s results -d dataset -r matrix -j ../automl/defaults/classification.json -n 5

# merge them into the combined error matrix and runtime matrix
bash ../automl/generate_matrix.sh -m merge -s results/matrix

echo "Move error_matrx to defaults..."
rm -r "../automl/defaults/error_matrix.csv"
cp "results/matrix/error_matrix.csv" "../automl/defaults/error_matrix.csv"

echo "Move runtime_matrx to defaults..."
rm -r "../automl/defaults/runtime_matrix.csv"
cp "results/matrix/runtime_matrix.csv" "../automl/defaults/runtime_matrix.csv"

echo "Matrix generation finished!"