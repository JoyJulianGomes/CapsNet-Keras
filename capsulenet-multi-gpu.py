import argparse
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from capsulenet import (CapsNet, margin_loss, test)
from DatasetConfig import KTH as Dataset

K.set_image_data_format('channels_last')
np.set_printoptions(threshold=np.nan)


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
                               batch_size=args.batch_size, histogram_freq=args.debug)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (0.9 ** epoch))

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
                        callbacks=[log, tb, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(
        description="Capsule Network on Action Recognition Datasets.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--train', action='store_true',
                        help="train the model")
    parser.add_argument('--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--target_shape', nargs='+',
                        default=Dataset['target_shape'], type=int)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    target_shape = (args.target_shape[0], args.target_shape[1])
    with tf.device('/cpu:0'):
        model, eval_model, primaryCap_model, digitCap_model = CapsNet(input_shape=(target_shape[0], target_shape[1], Dataset['shape'][2]),
                                                                      n_class=Dataset['nb_classes'],
                                                                      routings=args.routings)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.train is True:
        # define muti-gpu model
        multi_model = multi_gpu_model(model, gpus=args.gpus)
        train(model=multi_model, target_shape=target_shape, args=args)
        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
        test(model=eval_model, target_shape=target_shape, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, target_shape=target_shape, args=args)
