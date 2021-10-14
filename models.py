#!/usr/bin/env python3

from keras.layers import Dense, LSTM, Bidirectional, Concatenate, Dropout, Input, RepeatVector
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_LSTM(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    model = Sequential(name='LSTM')
    model.add(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), 
                   recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),
                   input_shape=(1,DIMENSION))) # 5*33 + 2
    model.add(Dense(no_activities, activation='softmax'))
    return model

def get_LSTM_H(input_dim, output_dim, max_lenght, no_activities, DIMENSION):

    inputs = keras.Input(shape=(DIMENSION,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIMENSION-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(30)(input2)]
    
    LSTM = layers.LSTM(output_dim, activation='relu', input_shape = (5, DIMENSION-2), name='LSTM')(list_Frames_3D)
    
    Dropout_LSTM = layers.Dropout(0.1, name='Dropout_LSTM')(LSTM)

    Concatenate = layers.concatenate([input1,Dropout_LSTM], axis=1)
    
    Dense = layers.Dense(220, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    
    Dropout_Dense = layers.Dropout(0.3, name='Dropout_Dense')(Dense)
    
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Dense)

    model = keras.Model(inputs=inputs, outputs=Output, name="modelDENSE")
    
    return model

def get_Transformer(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    
    window = int((DIMENSION - 2)/(33))
    
    inputs = keras.Input(shape=(DIMENSION,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIMENSION-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(window)(input2)]
    
    Transformer = TransformerBlock(DIMENSION-2, 16, 64)(list_Frames_3D)
    
    BatchNormalization = layers.BatchNormalization()(Transformer)

    Concatenate = layers.concatenate([input1,BatchNormalization], axis=1)
    
    Dense = layers.Dense(4700, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    
    Dropout_Dense = layers.Dropout(0.2, name='Dropout_Dense')(Dense)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Dense)

    model = keras.Model(inputs=inputs, outputs=Output, name="modelDENSE")
    
    return model
    
def get_biLSTM_H(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    
    window = int((DIMENSION - 2)/(33))
    
    inputs = keras.Input(shape=(DIMENSION,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIMENSION-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(window)(input2)]

    BI_LSTM = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (window, DIMENSION-2),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM'))(list_Frames_3D)
          
    Dropout_LSTM = layers.Dropout(0.2, name='Dropout_LSTM')(BI_LSTM)

    BatchNormalization = layers.BatchNormalization()(Dropout_LSTM)

    Concatenate = layers.concatenate([input1,BatchNormalization], axis=1)
    
    Dense1 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)

    Dense2 = layers.Dense(2200, activation='relu', use_bias=False,name='denseLayer2')(Dense1)
    
    Dropout_Final = layers.Dropout(0.4, name='Dropout_Final')(Dense2)

    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modelDENSE")
    
    return model


def get_biLSTM(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim)))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_Ensemble2LSTM(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    model1 = Sequential()
    model1.add(Bidirectional(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),input_shape=(1,DIMENSION)))) # 5*33 + 2

    model2 = Sequential()
    model2.add(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),input_shape=(1,DIMENSION))) # 5*33 + 2

    model = Sequential(name='Ensemble2LSTM')
    model.add(Concatenate([model1, model2], mode='concat'))
    model.add(Concatenate())
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeEnsembleLSTM(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    model1 = Sequential()
    model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model1.add(Bidirectional(LSTM(output_dim, return_sequences=True)))

    model2 = Sequential()
    model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model2.add(LSTM(output_dim, return_sequences=True))

    model = Sequential(name='CascadeEnsembleLSTM')
    model.add(Concatenate([model1, model2], mode='concat'))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeLSTM(input_dim, output_dim, max_lenght, no_activities, DIMENSION):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def compileModel(model):
    optimizer = Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
