"""
CapsuleNet implementation for Bangla Digits
"""
import numpy as np
from keras import models
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_BengaliHandwrittenDigit():
    # load data
    train_data_dir = 'datasets/BengaliDigits/train'
    validation_data_dir = 'datasets/BengaliDigits/validation'
    test_data_dir = 'datasets/BengaliDigits/test'

    nb_train_samples = 300
    nb_validation_samples = 100
    nb_test_samples = 70

    img_width, img_height = 28, 28
    input_shape = (img_width, img_height)

    # image conversion from int to float and data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2)

    # no data augmentation for validation
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    # no data augmentation for test
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_train_samples,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_validation_samples,
        color_mode="grayscale",
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_test_samples,
        color_mode="grayscale",
        class_mode='categorical')

    (x_train, y_train) = train_generator.next()
    (x_validation, y_validation) = validation_generator.next()
    (x_test, y_test) = test_generator.next()
    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def load_BengalOCR():
    target_shape = (28, 28)

    train_sample = 1740*10
    validation_sample = 145*10
    test_sample = 97*10

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       data_format='channels_last')
    train_generator = train_datagen.flow_from_directory(
        directory='E:/Work/Thesis/CapsNet-Keras/datasets/Bangla OCR Dataset/numerical/train',
        target_size=target_shape,
        batch_size=train_sample,
        color_mode='grayscale',
        class_mode='categorical'
    )
    (x_train, y_train) = train_generator.next()

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        directory='E:/Work/Thesis/CapsNet-Keras/datasets/Bangla OCR Dataset/numerical/validation',
        target_size=target_shape,
        batch_size=validation_sample,
        color_mode="grayscale",
        class_mode='categorical'
    )
    (x_validation, y_validation) = validation_generator.next()

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory='E:/Work/Thesis/CapsNet-Keras/datasets/Bangla OCR Dataset/numerical/test',
        target_size=target_shape,
        batch_size=test_sample,
        color_mode="grayscale",
        class_mode='categorical'
    )
    (x_test, y_test) = test_generator.next()

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def load_Agressive():
    target_shape = (28, 28)

    train_sample = 7152+4768
    validation_sample = 3540+2360
    test_sample = 1200+800

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       data_format='channels_last')
    train_generator = train_datagen.flow_from_directory(
        directory='E:/Work/Thesis/Agressive-Binary/dataset/Agressive/train',
        target_size=target_shape,
        batch_size=train_sample,
        color_mode='grayscale',
        class_mode='categorical'
    )
    (x_train, y_train) = train_generator.next()

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        directory='E:/Work/Thesis/Agressive-Binary/dataset/Agressive/validation',
        target_size=target_shape,
        batch_size=validation_sample,
        color_mode="grayscale",
        class_mode='categorical'
    )
    (x_validation, y_validation) = validation_generator.next()

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory='E:/Work/Thesis/Agressive-Binary/dataset/Agressive/test',
        target_size=target_shape,
        batch_size=test_sample,
        color_mode="grayscale",
        class_mode='categorical'
    )
    (x_test, y_test) = test_generator.next()

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def customTest():
    # load data
    test_data_dir = 'datasets/customTest'

    nb_test_samples = 2

    img_width, img_height = 28, 28
    input_shape = (img_width, img_height)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_test_samples,
        color_mode="grayscale",
        class_mode='categorical')

    (x_test, y_test) = test_generator.next()
    # print(x_test)
    # print(y_test)
    # plt.imshow(x_test[0])
    # plt.show()
    # plt.imshow(x_test[1])
    # plt.show()
    return (x_test, y_test), (x_test, y_test)
