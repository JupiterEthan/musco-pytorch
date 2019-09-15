import tensorflow as tf

from tensorflow import keras
from compress_model import get_compressed_model


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

    compressed_model = get_compressed_model(model, {
                        'dense': ('svd', 300),
    })

    compressed_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('Evaluate compressed model')
    test_loss, test_acc = compressed_model.evaluate(test_images, test_labels, verbose=0)
    print('Test accuracy:', test_acc)

    for layer in compressed_model.layers:
        print(layer.name)


def test_svd_iterative(take_first=None):
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

    ranks = [750, 350]
    compressed_model = model
    for idx in range(2):
        compressed_model = get_compressed_model(compressed_model, {
                            'dense': ('svd', ranks[idx]),
        })

        compressed_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print('Evaluate compressed model')
        test_loss, test_acc = compressed_model.evaluate(test_images, test_labels, verbose=0)
        print('Test accuracy:', test_acc)

        for layer in compressed_model.layers:
            print(layer.name)


#TODO: write regular tests
if __name__ == "__main__":
    # test_svd(10000)
    test_svd_iterative(10000)
