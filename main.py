# from __future__ import print_function
import os
import argparse
import tensorflow as tf

from deep_emotion import Deep_Emotion
from generate_data import Generate_data

tf.random.set_seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                               help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type= int, help= 'number of epochs')
    parser.add_argument('-lr', '--learning_rate', type= float, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, help= 'training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup :
        generate_dataset = Generate_data(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images('train')
        # generate_dataset.save_images('test')
        generate_dataset.save_images('val')

    if args.hyperparams:
        epochs = args.epochs
        lr = args.learning_rate
        batchsize = args.batch_size
    else : # setting hyperparameters as mentioned in paper
        epochs = 500
        lr = 0.005
        batchsize = 32

    if args.train:
        net = Deep_Emotion()
        # net.to(device)
        print("Model architecture: ", net)
        train_img_dir = os.path.join(args.data, 'train')
        validation_img_dir = os.path.join(args.data, 'val')

        # transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=train_img_dir,
            color_mode='grayscale',
            batch_size=batchsize,
            image_size=(48,48)
        )
        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=validation_img_dir,
            color_mode='grayscale',
            batch_size=batchsize,
            image_size=(48,48)
        )

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        net.compile(
            optimizer=optimizer,
            loss=loss_object,
            metrics=['accuracy']
        )

        net.fit(
            x=train_dataset,
            epochs=epochs,
            validation_data=validation_dataset
        )
