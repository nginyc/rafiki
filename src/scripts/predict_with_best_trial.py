import pprint
from tensorflow import keras
import matplotlib.pyplot as plt

from admin import Admin

TEST_START_INDEX = 0
TEST_END_INDEX = 8

def get_predict_data():
    _, (test_images, test_labels) = \
        keras.datasets.fashion_mnist.load_data()

    return (
        test_images[TEST_START_INDEX:TEST_END_INDEX],
        test_labels[TEST_START_INDEX:TEST_END_INDEX]
    )

def show_first_image(images):
    plt.figure()
    plt.imshow(images[0])
    plt.colorbar()
    plt.gca().grid(False)
    print('Showing first image...')
    plt.show()

admin = Admin()

app_status = admin.get_app_status(
    name='fashion_mnist_app'
)

best_trials = app_status.get('best_trials')

if len(best_trials) == 0:
    raise Exception('No trials yet!')

best_trial = best_trials[0]

(predict_images, predict_labels) = get_predict_data()
show_first_image(predict_images)
predictions = admin.predict(best_trial.get('id'), predict_images)

print()
print('Predictions:')
pprint.pprint(predictions)
print()
print('Labels:')
pprint.pprint(predict_labels)