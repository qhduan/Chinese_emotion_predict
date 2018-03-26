#!/usr/bin/env python3

import pickle
import numpy as np
from tqdm import tqdm
from keras import layers, models
from keras import callbacks, optimizers
from keras import regularizers

from utils import MAX_LENGTH, LABEL_DICT


def build_model(shape, vocab_size):
    """构建模型
    """
    input_layer = layers.Input(shape=shape)

    m = input_layer
    m = layers.Embedding(vocab_size, 64)(m)
    m = layers.Dropout(0.1)(m)

    m = layers.GRU(
        32,
        return_sequences=True,
        # recurrent_dropout=0.2,
        kernel_regularizer=regularizers.l2(0.001)
    )(m)

    m = layers.GRU(
        32,
        return_sequences=True,
        # recurrent_dropout=0.2,
        kernel_regularizer=regularizers.l2(0.001)
    )(m)

    atten = m
    atten = layers.Flatten()(atten)
    atten = layers.Dense(shape[0], activation='softmax')(atten)
    atten = layers.RepeatVector(32)(atten)
    atten = layers.Permute((2, 1))(atten)

    m = layers.Multiply()([m, atten])

    # m = layers.Add()([m, emb])

    m = layers.Flatten()(m)
    m = layers.GaussianNoise(0.01)(m)

    m = layers.Dense(
        300, activation='linear',
        kernel_regularizer=regularizers.l2(0.01)
    )(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('tanh')(m)
    m = layers.Dropout(0.4)(m)
    m = layers.Dense(
        300, activation='linear',
        kernel_regularizer=regularizers.l2(0.01)
    )(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('tanh')(m)
    m = layers.Dropout(0.4)(m)
    m = layers.Dense(len(LABEL_DICT), activation='softmax')(m)

    atten_model = models.Model(
        inputs=[input_layer],
        outputs=atten
    )

    model = models.Model(
        inputs=[input_layer],
        outputs=m
    )

    optimizer = optimizers.Adam(lr=0.001, clipnorm=5.)

    model.compile(
        optimizer,
        'categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model, atten_model


def main():
    """训练
    """

    x_train, x_test, y_train, y_test, class_weight = pickle.load(open('train_data.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    input_shape = x_train[0].shape
    model, atten_model = build_model(input_shape, len(ws))

    nb_epoch = 20
    batch_size = 64 * 8
    steps_per_epoch = int(len(x_train) / batch_size) + 1
    validation_steps = int(len(x_test) / batch_size) + 1

    model.save_weights('./model_gru_last_weights.hdf5')
    atten_model.save_weights('./model_gru_last_weights.hdf5')

    model.fit_generator(
        generator=batch_flow(
            x_train, y_train, batch_size=batch_size
        ),
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        validation_data=batch_flow(
            x_test, y_test, batch_size=batch_size
        ),
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=[
            callbacks.ReduceLROnPlateau(min_lr=1e-6),
            callbacks.ModelCheckpoint(
                monitor='val_acc',
                filepath='./model_gru_weights.hdf5',
                verbose=1,
                save_best_only=True)
        ]
    )

    model.save_weights('./model_gru_last_weights.hdf5')
    atten_model.save_weights('./model_gru_last_weights.hdf5')


def batch_flow(inputs, targets, batch_size=32):
    """流动数据流
    """

    x_batch, y_batch = [], []

    while True:

        if len(x_batch) == batch_size:
            x_batch, y_batch = np.array(x_batch), np.array(y_batch)
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        ind = np.random.randint(0, len(inputs))
        x = inputs[ind]
        y = targets[ind]
        x_batch.append(x)
        y_batch.append(y)


if __name__ == '__main__':
    main()