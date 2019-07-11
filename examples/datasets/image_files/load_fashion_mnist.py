from examples.datasets.image_files.mnist import load

# Loads the official Fashion MNIST dataset for the `IMAGE_CLASSIFICATION` task
def load_fashion_mnist(out_train_dataset_path='data/fashion_mnist_for_image_classification_train.zip',
                        out_val_dataset_path='data/fashion_mnist_for_image_classification_val.zip',
                        out_meta_csv_path='data/fashion_mnist_for_image_classification_meta.csv'):
    
    load(
        train_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        train_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        test_images_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        test_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        label_to_name={
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        },
        out_train_dataset_path=out_train_dataset_path,
        out_val_dataset_path=out_val_dataset_path,
        out_meta_csv_path=out_meta_csv_path
    )

if __name__ == '__main__':
    load_fashion_mnist()    
    