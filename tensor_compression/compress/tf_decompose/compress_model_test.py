import numpy as np
import tensorflow as tf

from tensorflow import keras
from compress_model import get_compressed_sequential


def test_svd(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images[:100, ...], train_labels[:100, ...], epochs=1)

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_sequential(model, {
                'dense': ('svd', 750),
    })

    compressed_model.summary()

    for layer in compressed_model.layers:
        if layer.name == 'svd_layer':
            print("?????", layer.u.get_shape())
            print("?????", layer.s.get_shape())
            print("?????", layer.v.get_shape())

    compressed_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('Evaluate compressed model')
    test_loss, test_acc = compressed_model.evaluate(test_images, test_labels, verbose=0)
    print('Test accuracy:', test_acc)

    for layer in compressed_model.layers:
        print(layer.name)

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
            tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
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

    compressed_model = get_compressed_sequential(model, {
        'conv2d': ('cp3', 50),
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
    test_svd(1000)
