import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import callbacks, layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image

from capsulelayers import CapsuleLayer, Length, Mask, PrimaryCap
from utils import combine_images
from DatasetConfig import KTH as Dataset

K.set_image_data_format('channels_last')
np.set_printoptions(threshold=np.nan)


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256,
                          kernel_size=9,
                          strides=1,
                          padding='valid',
                          activation='relu',
                          name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1,
                             dim_capsule=8,
                             n_channels=32,
                             kernel_size=9,
                             strides=2,
                             padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class,
                             dim_capsule=16,
                             routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training, evaluation (prediction) amd analyzing
    train_model = models.Model(x, out_caps)
    eval_model = models.Model(x, out_caps)
    primaryCap_model = models.Model(x, primarycaps)
    digitCap_model = models.Model(x, digitcaps)

    return train_model, eval_model, primaryCap_model, digitCap_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, target_shape, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_gen(args):
        train_datagen = ImageDataGenerator(width_shift_range=args.shift_fraction,
                                           height_shift_range=args.shift_fraction,
                                           rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           data_format='channels_last')
        train_generator = train_datagen.flow_from_directory(
            directory=Dataset['train_path'],
            target_size=target_shape,
            batch_size=args.batch_size,
            color_mode='grayscale',
            class_mode='categorical')
        while 1:
            x_batch, y_batch = train_generator.next()
            yield [x_batch, y_batch]

    def val_gen(args):
        validation_datagen = ImageDataGenerator(
            rescale=1./255, data_format='channels_last')
        validation_generator = validation_datagen.flow_from_directory(
            directory=Dataset['validation_path'],
            target_size=target_shape,
            batch_size=args.batch_size,
            color_mode='grayscale',
            class_mode='categorical')
        while 1:
            x_batch, y_batch = validation_generator.next()
            yield [x_batch, y_batch]

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_gen(args),
                        steps_per_epoch=int(
                            Dataset['nb_train_sample'] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=val_gen(args),
                        validation_steps=int(
                            Dataset['nb_validation_sample']/args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, target_shape, args):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory=Dataset['nb_test_path'],
        target_size=target_shape,
        shuffle=False,
        batch_size=args.batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )
    y_pred = model.predict_generator(generator=test_generator)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1)
                              == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(
        args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def leak(model, data, s_dir, lw, args):
    x_test, y_test = data
    file = open(s_dir, 'w')
    file.write(np.array2string(model.predict(x_test)))
    file.close()


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(
        description="Capsule Network on Action Recognition Datasets.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--train', action='store_true',
                        help="train the model")
    parser.add_argument('--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-l', '--leak', default=None,
                        help="outputs activation of intermediate layer. available layers: pc=primarycaps, dc=digitcaps")
    parser.add_argument('--target_shape', nargs='+',
                        default=Dataset['target_shape'], type=int)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    target_shape = (args.target_shape[0], args.target_shape[1])
    model, eval_model, primaryCap_model, digitCap_model = CapsNet(input_shape=(target_shape[0], target_shape[1], Dataset['shape'][2]),
                                                                  n_class=Dataset['nb_classes'],
                                                                  routings=args.routings)

    model.summary()

    # train, test or analyze

    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
        print('weight loaded')

    if args.train is True:  # Train the model
        train(model=model, target_shape=target_shape, args=args)

    if args.testing is True:  # Test the model
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, target_shape=target_shape,  args=args)

    if args.leak is not None:

        if args.leak == 'pc':  # shows output of primaryCap layer
            leak(model=primaryCap_model, data=sampleData,
                 s_dir='result/leakpc.txt', lw=8, args=args)
        elif args.leak == 'dc':
            leak(model=digitCap_model, data=sampleData,
                 s_dir='result/leakdc.txt', lw=16, args=args)
        elif args.leak == 'mask':
            leak(model=masked_model, data=sampleData,
                 s_dir='result/leakMask.txt', lw=160, args=args)
        elif args.leak == 'decoder':
            leak(model=eval_model, data=sampleData,
                 s_dir='result/leakDecoder.txt', lw=28, args=args)
        elif args.leak == 'all':
            leak(model=primaryCap_model, data=sampleData,
                 s_dir='result/leakpc.txt', lw=8, args=args)
            leak(model=digitCap_model, data=sampleData,
                 s_dir='result/leakdc.txt', lw=16, args=args)
            leak(model=masked_model, data=sampleData,
                 s_dir='result/leakMask.txt', lw=160, args=args)
            leak(model=decoder_model, data=sampleData,
                 s_dir='result/leakDecoder.txt', lw=28, args=args)
