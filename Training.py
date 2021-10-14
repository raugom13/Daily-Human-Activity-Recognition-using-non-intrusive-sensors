#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:41:44 2021

@author: raugom13
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import csv
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import compute_class_weight

import random
from random import randint

from matplotlib import pyplot

import PrepareData as Prepare
import models

import tensorflow as tf

# It takes the memory of the GPU in a progressive way
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Important parameters
seed = 7
units = 64
batch_size = 256
epochs = 50
DIMENSION = 5*33 + 2 # Data dimension (60 min * 33 sensors + 2 hours)

if __name__ == '__main__':

    """The entry point"""
    # set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--v', dest='model', action='store', default='', help='deep model')
    args = p.parse_args()

    print(Prepare.datasetsNames)
    for dataset in Prepare.datasetsNames:
        
        Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation, dictAct = Prepare.getData(dataset)
        
        target_names = sorted(dictAct, key=dictAct.get)
              
        cvaccuracy = []
        cvscores = []
        modelname = ''

        args_model = str(args.model)
                
        if 'Ensemble' in args_model:
            input_dim = len([Xtrain, Xtrain])
            X_train_input = [Xtrain, Xtrain]
            X_test_input = [Xtest, Xtest]
            X_validation_input = [Xvalidation, Xvalidation]
            
        elif 'LSTM_H' in args_model:
            input_dim = len(Xtrain)
            X_train_input = np.array(Xtrain, dtype=object).astype(float)
            X_test_input = np.array(Xtest, dtype=object).astype(float)
            X_validation_input = np.array(Xvalidation, dtype=object).astype(float)
            
        else:
            input_dim = len(Xtrain)
            X_train_input = np.array(Xtrain, dtype=object).astype(float)
            X_test_input = np.array(Xtest, dtype=object).astype(float)
            X_validation_input = np.array(Xvalidation, dtype=object).astype(float)          
         
        Y_train = np.array(Ytrain, dtype=object).astype(float)
        Y_test = np.array(Ytest, dtype=object).astype(float)
        Y_validation = np.array(Yvalidation, dtype=object).astype(float)
                             
        no_activities = len(dictAct)
                      
        if args_model == 'LSTM':
            model = models.get_LSTM(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'LSTM_H':
            model = models.get_LSTM_H(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'biLSTM':
            model = models.get_biLSTM(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'biLSTM_H':
            model = models.get_biLSTM_H(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'Ensemble2LSTM':
            model = models.get_Ensemble2LSTM(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'CascadeEnsembleLSTM':
            model = models.get_CascadeEnsembleLSTM(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        elif args_model == 'CascadeLSTM':
            model = models.get_CascadeLSTM(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)
        else:
            model = models.get_biLSTM_H(input_dim, units, Prepare.max_lenght, no_activities, DIMENSION)

        model = models.compileModel(model)
        modelname = model.name

        currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        csv_logger = CSVLogger(
            model.name + '-' + dataset + '-' + str(currenttime) + '.csv')
        model_checkpoint = ModelCheckpoint(
            model.name + '-' + dataset + '-' + str(currenttime) + '.h5',
            monitor='acc',
            save_best_only=True)
        
        # train the model
        print('Begin training ...')                                            
               
        history = None    
        history = model.fit(X_train_input, Y_train, validation_data=(X_validation_input, Y_validation), epochs=epochs, batch_size=batch_size, verbose=1,
                  callbacks=[csv_logger, model_checkpoint])
        
        # evaluate the model
        print('Begin testing ...')
        scores = model.evaluate(X_test_input, Y_test, batch_size=batch_size, verbose=1)
        print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
        
        print(str(scores))
        print('Test Loss: %.3f' % scores[0])
        print('Test Accuracy: %.3f' % scores[1])

        print('Report:')
        target_names = sorted(dictAct, key=dictAct.get)
        
        ####  Training plot
        pyplot.title('Learning Curves (Accuracy)')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.plot(history.history['accuracy'], color='green', linewidth=2, label='Training accuracy')
        pyplot.plot(history.history['val_accuracy'], color='red', linewidth=2, label='Validation accuracy')
        pyplot.legend()
        PLOT_TRAINING = "./Results/training_curve_LSTM_accuracy.svg"
        pyplot.savefig(PLOT_TRAINING, format='svg', dpi=1200)
        
        ####  Loss plot
        pyplot.clf()
        pyplot.title('Learning Curves (Loss)')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.plot(history.history['loss'], color='green', linewidth=2, label='Training loss')
        pyplot.plot(history.history['val_loss'], color='red', linewidth=2, label='Validation loss')
        pyplot.legend()
        PLOT_LOSS = "./Results/training_curve_LSTM_loss.svg"
        pyplot.savefig(PLOT_LOSS, format='svg', dpi=1200)
        
        classes = model.predict(X_test_input, batch_size=batch_size)
        classes = np.argmax(classes, axis=1)
        print(classification_report(list(Y_test), classes, target_names=target_names))
        print('Confusion matrix:')
        labels = list(dictAct.values())
        print(confusion_matrix(list(Y_test), classes, labels))

        cvaccuracy.append(scores[1] * 100)
        cvscores.append(scores)

        print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))
        
        # Save the model
        model.save(os.path.join('./Results/', 'modelLSTM-final.h5'))

        currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        csvfile = './Results/cv-scores-' + modelname + '-' + dataset + '-' + str(currenttime) + '.csv'

        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in cvscores:
                writer.writerow([",".join(str(el) for el in val)])