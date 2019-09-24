import numpy as np
import tensorflow as tf

from tensorflow import keras
from compress_model import get_compressed_model

np.random.seed(42)


def test_cp3(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=4, kernel_size=5, padding='same', activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')

        ]
    )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_images = np.expand_dims(train_images, -1)
    train_labels = train_labels

    test_images = np.expand_dims(test_images, -1)
    test_labels = test_labels

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_model(model, {
        'conv2d': ('cp3', 4),
    })

    compressed_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    print('Evaluate compressed model')
    test_loss, test_acc = compressed_model.evaluate(test_images,
                                                    test_labels,
                                                    verbose=0)

    compressed_model.summary()
    print('Test accuracy:', test_acc)

    for layer in compressed_model.layers:
        print(layer.name)


def test_cp3_optimize_rank(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=4, kernel_size=5, padding='same', activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')

        ]
    )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_images = np.expand_dims(train_images, -1)
    train_labels = train_labels

    test_images = np.expand_dims(test_images, -1)
    test_labels = test_labels

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_model(model, {
        'conv2d': ('cp3', 5),
    }, optimize_rank=True)

    compressed_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])

    print('Evaluate compressed model')
    test_loss, test_acc = compressed_model.evaluate(test_images,
                                                    test_labels,
                                                    verbose=0)

    compressed_model.summary()
    print('Test accuracy:', test_acc)

    for layer in compressed_model.layers:
        print(layer.name)

def test_cp3_sequential(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')

        ]
    )

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_images = np.expand_dims(train_images, -1)
    train_labels = train_labels

    test_images = np.expand_dims(test_images, -1)
    test_labels = test_labels

    model.fit(train_images,
              train_labels,
              epochs=1)

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = model
    ranks = [60, 3]
    for idx in range(2):
        compressed_model = get_compressed_model(compressed_model, {
            'conv2d': ('cp3', ranks[idx]),
        })

        compressed_model.compile(optimizer='adam',
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
        compressed_model.summary()
        print('Evaluate compressed model')
        test_loss, test_acc = compressed_model.evaluate(test_images,
                                                        test_labels,
                                                        verbose=0)
        print('Test accuracy:', test_acc)

    for layer in compressed_model.layers:
        print(layer.name)


#TODO: write regular tests
if __name__ == "__main__":
    test_cp3(10000)
    # test_cp3_sequential(10000)
    # test_cp3_optimize_rank(1000)
