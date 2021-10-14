![Requirements](https://img.shields.io/badge/Python-3.8.5-lightgrey)
![Commit](https://img.shields.io/github/last-commit/raugom13/Daily-Human-Activity-Recognition-using-non-intrusive-sensors) 

# Daily-Human-Activity-Recognition-using-non-intrusive-sensors

A prediction model has been developed based on recurrent neural networks, specifically on bidirectional LSTM networks, to obtain in real-time the activity being carried out by the individuals in their homes, based on the information provided by a set of different sensors installed at each person’s home. The prediction model provides a 95.42% accuracy rate, improving the results of similar models currently in use. In order to obtain a reliable model with a high accuracy rate, a series of processing and filtering processes have been carried out on the data, such as a method based on a sliding window or a stacking and re-ordering algorithm, that are subsequently used to train the neural network, obtained from the public database CASAS.

Link to the paper: https://www.mdpi.com/1424-8220/21/16/5270

# Setup and demo

- Clone this repo to your local machine using 
 ```
 git clone https://github.com/raugom13/Daily-Human-Activity-Recognition-using-non-intrusive-sensors.git
  ```
  
- Execute this line to convert the raw data into .npy format and split it to test, train and validation.
 ```
 python3 PrepareData.py
  ```
- Execute this line to perform the training of the neural network. In addition, you can select the architecture of the network (biLSTM_H) and store the relevant information of the training into a log file (attempt1.log).
 ```
 python3 Training.py --v biLSTM_H > attempt1.log
  ```
The model has been trained on an Intel(R) Core(TM) i9-10900K CPU@3.70 GHz / 128 Gb with two RTX3090 GPUs.

# Configuration of PrepareData.py
- It is possible to filter the number of events of the activities to be placed in this part of the code together with the filtering ratio (as a percentage of 1).
 ```
filters = {
    "Other" : 0.05,
    "Sleep" : 0.05
}
  ```
- By modifying this part of the code it is possible to vary the weight of the data distribution in the training, test and validation sets.
```
# Training percentages
validation_percent = 0.1
test_percent = 0.2
  ```
- In progress...
```
secondsDIV = 60
max_lenght = 33
dimWINDOW = 5
stack_size = 2
  ```
