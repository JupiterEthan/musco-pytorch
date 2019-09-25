import numpy as np
import tensorflow as tf

from tensorflow import keras
from compress_model import get_compressed_sequential, get_compressed_model

np.random.seed(42)


def test_tucker2(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               name='test'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_sequential(model, {
        'test': ('tucker2', (50, 50)),
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


def test_tucker2_seq(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               name='test'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = model
    ranks = [(64, 64), (40, 40), (20, 20)]
    for idx in range(len(ranks)):
        compressed_model = get_compressed_sequential(compressed_model, {
            'test': ('tucker2', ranks[idx]),
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


def test_tucker2_optimize_rank(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               name='test'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_sequential(model, {
        'test': ('tucker2', (50, 50)),
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


def test_tucker2_optimize_rank(take_first=None):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               activation='relu',
                               name='test'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,
              train_labels,
              epochs=1)

    model.summary()

    print('Evaluate source model')
    test_loss, test_acc = model.evaluate(test_images,
                                         test_labels,
                                         verbose=0)
    print('Test accuracy:', test_acc)

    compressed_model = get_compressed_sequential(model, {
        'test': ('tucker2', (50, 50)),
    }, optimize_rank=True, vbmf=True, vbmf_weaken_factor=0.8)

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


def test_tucker2_model_compress(take_first=1000):
    def createModel():
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   input_shape=(28, 28, 1),
                                   name='conv_1')(inputs)
        x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu',
                                   name='conv_2')(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='dense_1')(x)
        return tf.keras.Model(inputs, outputs)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if take_first is not None:
        train_images = train_images[:take_first, ...]
        train_labels = train_labels[:take_first, ...]

        test_images = test_images[:take_first, ...]
        test_labels = test_labels[:take_first, ...]

    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    model = createModel()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

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
        'conv_2': ('tucker2', (50, 50)),
    }, optimize_rank=True, vbmf=True, vbmf_weaken_factor=0.8)
    # compressed_model = model

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


#TODO: write regular tests
if __name__ == "__main__":
    # test_tucker2(1000)
    # test_tucker2_seq(1000)
    # test_tucker2_optimize_rank(1000)
    test_tucker2_model_compress(1000)
