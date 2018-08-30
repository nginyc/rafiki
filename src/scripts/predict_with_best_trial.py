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
    image = images[0]
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.gca().grid(False)
    print('Showing first image...')
    pprint.pprint(image)
    plt.show()

admin = Admin()

trials = admin.get_best_trials_by_app(
    app_name='fashion_mnist_app',
    max_count=1
)
if len(trials) == 0:
    raise Exception('No trials yet!')

best_trial = trials[0]

(predict_images, predict_labels) = get_predict_data()
show_first_image(predict_images)

predictions = admin.predict_with_trial(
    app_name='fashion_mnist_app',
    trial_id=best_trial.get('id'), 
    queries=predict_images
)

print()
print('Predictions:')
pprint.pprint(predictions)
print()
print('Labels:')
pprint.pprint(predict_labels)