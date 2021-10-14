#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:34:48 2021

@author: raugom13
"""

import tensorflow as tf
import numpy as np
import random
from time import time
import csv
import math
import os
from datetime import datetime as dt, timedelta
from collections import Counter

# It takes the memory of the GPU in a progressive way
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Start time
start_time = time()

sensors = dict()
sensors = {
    "M001" : 0, "M002" : 0, "M003" : 0, "M004" : 0, "M005" : 0, "M006" : 0, "M007" : 0, "M008" : 0, "M009" : 0,
    "M010" : 0, "M011" : 0, "M012" : 0, "M013" : 0, "M014" : 0, "M015" : 0, "M016" : 0, "M017" : 0, "M018" : 0,
    "M019" : 0, "M020" : 0, "M021" : 0, "M022" : 0, "M023" : 0, "M024" : 0, "M025" : 0, "M026" : 0, "M027" : 0,
    "M028" : 0, "D001" : 0, "D002" : 0, "D003" : 0, "T001" : 20, "T002" : 20
}

activities = dict()
activities = {
    "Bed_to_Toilet" : 0, "Chores" : 1, "Desk_Activity" : 2, "Dining_Rm_Activity" : 3, "Eve_Meds" : 4,
    "Guest_Bathroom" : 5, "Kitchen_Activity" : 6, "Leave_Home" : 7, "Master_Bathroom" : 8, "Meditate" : 9,
    "Watch_TV" : 10, "Sleep" : 11, "Read": 12, "Morning_Meds" : 13, "Master_Bedroom_Activity" : 14, "Other" : 15     
}

group = dict()
group = {
    "Bed_to_Toilet" : "Bed_to_Toilet", 
    "Sleep" : "Sleep", 
    "Chores" : "Chores", 
    "Master_Bedroom_Activity" : "Master_Bedroom_Activity",    
    "Eve_Meds" : "Eve_Meds",
    "Morning_Meds" : "Morning_Meds",
    "Kitchen_Activity" : "Kitchen_Activity",
    "Desk_Activity" : "Desk_Activity",
    "Dining_Rm_Activity" : "Dining_Rm_Activity",
    "Guest_Bathroom" : "Guest_Bathroom",    
    "Leave_Home" : "Leave_Home",
    "Master_Bathroom" : "Master_Bathroom",
    "Meditate" : "Meditate",
    "Watch_TV" : "Watch_TV",
    "Read": "Read",        
    "Other" : "Other"    
}

filters = dict()
filters = {
    "Other" : 0.05,
    "Sleep" : 0.05
}

# Training percentages
validation_percent = 0.1
test_percent = 0.2

transMatrix = np.zeros((15, 15)).astype(int)

datasets = ["./dataset/milan_edited"]
datasetsNames = [i.split('/')[-1] for i in datasets]
secondsDIV = 60
max_lenght = 33
dimWINDOW = 5
stack_size = 2

# Temperature values to normalization
tempMIN = 13
tempMAX = 30

def date_generator(row):

    day = row[0]
    hour = row[1].partition('.')
    
    date = dt.strptime((day + ' ' + hour[0]),"%Y-%m-%d %H:%M:%S")
    
    return date

def row_generator(sensors, group, activities, ACT, hourX, hourY):
    
    sensors["T001"] = (float(sensors["T001"]) - 13)/(30-13)
    sensors["T002"] = (float(sensors["T002"]) - 13)/(30-13)
    
    row = [hourX, hourY, sensors["M001"], sensors["M002"], sensors["M003"], sensors["M004"], sensors["M005"], 
           sensors["M006"], sensors["M007"], sensors["M008"], sensors["M009"], sensors["M010"], sensors["M011"],
           sensors["M012"], sensors["M013"], sensors["M014"], sensors["M015"], sensors["M016"], sensors["M017"], 
           sensors["M018"], sensors["M019"], sensors["M020"], sensors["M021"], sensors["M022"], sensors["M023"],
           sensors["M024"], sensors["M025"], sensors["M026"], sensors["M027"], sensors["M028"], sensors["D001"],
           sensors["D002"], sensors["D003"], sensors["T001"], sensors["T002"], activities[group[ACT]]]
    
    return row

def match_dictionaries(dict_base, dict_final):
    for key in dict_base.keys():
        dict_final[key] = dict_base[key]
    return dict_final

def load_dataset(filename):

    file_rows = []
    processed_rows = []
    elements = 0
    counter = 0
    num_rows = 0
    
    dicSensorsAcum = dict()
    dicSensorsAcum = {
        "M001" : 0, "M002" : 0, "M003" : 0, "M004" : 0, "M005" : 0, "M006" : 0, "M007" : 0, "M008" : 0, "M009" : 0,
        "M010" : 0, "M011" : 0, "M012" : 0, "M013" : 0, "M014" : 0, "M015" : 0, "M016" : 0, "M017" : 0, "M018" : 0,
        "M019" : 0, "M020" : 0, "M021" : 0, "M022" : 0, "M023" : 0, "M024" : 0, "M025" : 0, "M026" : 0, "M027" : 0,
        "M028" : 0, "D001" : 0, "D002" : 0, "D003" : 0, "T001" : 20, "T002" : 20
    }
    
    #Activity (Activity 16 -> OTHER)
    ACT = "Other";
    ACT_NEW = "Sleep"

    with open(filename, 'rb') as features:
        database = features.readlines()
        for i, line in enumerate(database):  # each line
            f_info = line.decode().split()  # find fields
            file_rows.append(f_info)
                    
    # I get the start date from the data file
    print(file_rows[0][0])
    initial_date = date_generator(file_rows[0])
    print(initial_date)
    
    # I get the final date from the data file
    final_date = date_generator(file_rows[elements-1])
    print(final_date)
  
    date = initial_date - timedelta(seconds=initial_date.second)
    while(date < final_date):
        
        # If an event occurs in the data file
        while(date > date_generator(file_rows[counter])):
            
            # Sensor
            ID_sensor = file_rows[counter][2]
            
            # Status
            ID_state = file_rows[counter][3]
            
            if(ID_state == "ON" or ID_state == "OPEN"): 
                ID_state = 1
                dicSensorsAcum[ID_sensor] = ID_state
                
            elif(ID_state == "OFF" or ID_state == "CLOSE"): 
                ID_state = 0
                
            else:
                ID_state = float(ID_state)
                dicSensorsAcum[ID_sensor] = ID_state
                
            # Dictionary refresh
            sensors[ID_sensor] = ID_state # Real value
            
            # If a change in activity occurs
            if(len(file_rows[counter]) > 4):
                
                ACT_PREV = ACT_NEW
                
                if(file_rows[counter][5] == 'begin'):                  
                    ACT = file_rows[counter][4]
                    ACT_NEW = ACT
                else:
                    ACT = "Other"
                    
                transMatrix[activities[ACT_PREV]][activities[ACT_NEW]] += 1
            
            counter += 1
            
        date_cat = dt.strftime(date,"%Y-%m-%d %H:%M:%S")[11:19]
        hourX, hourY = generate_sin_cos_hour(date_cat)
        processed_rows.append(row_generator(dicSensorsAcum, group, activities, ACT, hourX, hourY))
        dicSensorsAcum = match_dictionaries(sensors,dicSensorsAcum)
        num_rows += 1
        date = date + timedelta(seconds=secondsDIV)
        
        
    # Save the transition matrix
    path_TransitionMatrix = './Results/' + datasetName + '_TransitionMatrix.csv'
    if os.path.exists(path_TransitionMatrix):
        os.remove(path_TransitionMatrix)   
    with open(path_TransitionMatrix, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(transMatrix)
    
    return processed_rows

def generate_sin_cos_hour(hour_min):
    
    hour = int(hour_min[0:2])
    min = int(hour_min[3:5])
    sec = int(hour_min[6:8])
    hourX = math.cos((2*math.pi*(hour + (min/60) + (sec/3600)))/(24))
    hourY = math.sin((2*math.pi*(hour + (min/60) + (sec/3600)))/(24))
    
    #                                              (x) - minimum
    # I NORMALIZE THE VALUES BETWEEN 0 AND 1 ->  -----------------
    #                                            maximum - minimum
    
    hourX = (hourX + 1)/(2)
    hourY = (hourY + 1)/(2)
    
    return hourX, hourY

def generate_categorical_dictionarie():
    
    dictTimes = {}
    date = "00:00:00"
    date = dt.strptime(date,"%H:%M:%S")
    for i in range(0,int((24*60*60)/secondsDIV)):
        date_s = dt.strftime(date,"%Y-%m-%d %H:%M:%S")[11:19]
        dictTimes[date_s] = i
        date = date + timedelta(seconds=secondsDIV)
    
    return dictTimes

def convert_proccesed_rows(processed_rows):
    X = []
    Y = []
    T = []
    
    for i, row in enumerate(processed_rows):
        T.append(processed_rows[i][0:2])
        X.append(processed_rows[i][2:35])
        Y.append(processed_rows[i][35])
    
    return X, Y, T

def sliding_window(X, Y, T):
    Xwindow = []
    Ywindow = []
    
    Xacum = []
    
    for i in range(0,len(X) - dimWINDOW + 1):
        Xacum = []
        Xacum.append(T[i+dimWINDOW-1][0])
        Xacum.append(T[i+dimWINDOW-1][1])
        
        for j in range(0, dimWINDOW):
            for k in range(0, 33):
                Xacum.append(X[i+j][k])
        Xwindow.append(Xacum)
        Ywindow.append(Y[i+dimWINDOW-1])        

    return Xwindow, Ywindow

def X_Y_time(X, Y, T):
    Xwindow = []
    Ywindow = []
    
    Xacum = []
    
    for i in range(0, len(X)):
        Xacum = []
        Xacum.append(T[i][0])
        Xacum.append(T[i][1])
        
        for j in range(0,33):
            Xacum.append(X[i][j])
        Xwindow.append(Xacum)
        Ywindow.append(Y[i])
            
    return Xwindow, Ywindow

def delete_rows(X, Y, dicUnits):
        
    for key in dicUnits.keys():
        
        filter_array = []
        buffer = []    
        Xprocessed = []
        Yprocessed = []
        
        for i in range(0, len(Y)):
            if(Y[i] == activities[key]):                           
                buffer.append(i)
            elif(Y[i] != activities[key] and len(buffer)>0):
                filter_array.append(buffer)
                buffer = []
        if(len(buffer)>0):
            filter_array.append(buffer)
            buffer = []
            
        for i in range(0, len(filter_array)):
            
            if(len(filter_array[i])>(1/dicUnits[key])):
                rows = round((len(filter_array[i]))*(dicUnits[key]/2),0)                        
                for j in range(0, len(filter_array[i])):
                    if(j < rows or j >= len(filter_array[i]) - rows):
                        buffer.append(filter_array[i][j])                           
            else:
                for j in range(0, len(filter_array[i])):
                    buffer.append(filter_array[i][j])
            
        counter = 0
        
        for i in range(0, len(Y)):
            if(Y[i] == activities[key]):
                if(i == buffer[counter]):
                    Xprocessed.append(X[i])
                    Yprocessed.append(Y[i])
                    counter += 1
                
            else:
                Xprocessed.append(X[i])
                Yprocessed.append(Y[i])
                
        X = []
        X = Xprocessed
        Y = []
        Y = Yprocessed

    return X, Y
            
def stack_group(X, Y):
       
    counter = 0
    Xstack = []
    Ystack = []
    stackX = []
    stackY = []
    sensor = []
    
    for i in range(0, len(X)):
        if(counter >= stack_size):
            Xstack.append(stackX)
            Ystack.append(stackY)
            stackX = []
            stackY = []
            counter = 0
        for j in range(0, len(X[i])):
            sensor.append(X[i][j])
        stackX.append(sensor)
        stackY.append(Y[i])
        counter += 1
        sensor = []
    Xstack.append(stackX)
    Ystack.append(stackY)
    
    return Xstack, Ystack

def generate_train_test_validation(days, test_percent, validation_percent):
    
    Ntest = round(days*test_percent,)
    Nvalidation = round(days*validation_percent,)
    Ntrain = days - Ntest - Nvalidation
    
    buffer = []
    for i in range(0, days):
        buffer.append(i)
     
    test = np.random.choice(buffer, size = Ntest, replace = False).tolist()
    
    for i in range(0, len(test)):
        buffer.remove(test[i])
        
    validation = np.random.choice(buffer, size = Nvalidation, replace = False).tolist()

    for i in range(0, len(validation)):
        buffer.remove(validation[i])

    train = np.random.choice(buffer, size = Ntrain, replace = False).tolist()
    
    return train, test, validation

    
def day_split(X, Y, split):
    
    Xproccesed = []
    Yproccesed = []   
    
    for i in range(0, int(len(split))):
        for j in range(0, int(len(X[split[i]]))):            
            Xproccesed.append(X[split[i]][j])
            Yproccesed.append(Y[split[i]][j])

    return Xproccesed, Yproccesed
            
if __name__ == '__main__':
    
    for filename in datasets:
        datasetName = filename.split("/")[-1]
        print('Loading ' + datasetName + ' dataset ...')
        
        processed_rows = load_dataset(filename)
        X, Y, T = convert_proccesed_rows(processed_rows)
        Xwindow, Ywindow = sliding_window(X,Y,T)      
        Xwindow, Ywindow = delete_rows(Xwindow, Ywindow, filters)
        
        Xwindow, Ywindow = stack_group(Xwindow, Ywindow)
        
        train, test, validation = generate_train_test_validation(len(Xwindow), test_percent, validation_percent)

        split = dict()
        split = {"train":train, "test": test, "validation": validation}
        
        for group in split:
            
            X, Y = day_split(Xwindow, Ywindow, split[group])
        
            X = np.array(X, dtype=object).astype(float)
            Y = np.array(Y, dtype=object).astype(float)
               
            if not os.path.exists('npy/splitted'):
                os.makedirs('npy/splitted')
    
            # TRAIN
            np.save('./npy/splitted/' + datasetName + '-x' + group + '.npy', X)
            np.save('./npy/splitted/' + datasetName + '-y' + group + '.npy', Y)
            
        # ACTIVITIES
        np.save('./npy/splitted/' + datasetName + '-act.npy', activities)
        
def getData(datasetName):
    
    # TRAIN
    Xtrain = np.load('./npy/splitted/' + datasetName + '-xtrain.npy', allow_pickle=True)
    Ytrain = np.load('./npy/splitted/' + datasetName + '-ytrain.npy', allow_pickle=True)
    
    # TEST
    Xtest = np.load('./npy/splitted/' + datasetName + '-xtest.npy', allow_pickle=True)
    Ytest = np.load('./npy/splitted/' + datasetName + '-ytest.npy', allow_pickle=True)
    
    # VALIDATION
    Xvalidation = np.load('./npy/splitted/' + datasetName + '-xvalidation.npy', allow_pickle=True)
    Yvalidation = np.load('./npy/splitted/' + datasetName + '-yvalidation.npy', allow_pickle=True)
    
    # ACTIVITIES
    dictAct = np.load('./npy/splitted/' + datasetName + '-act.npy', allow_pickle=True).item()
        
    return Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation, dictAct        
