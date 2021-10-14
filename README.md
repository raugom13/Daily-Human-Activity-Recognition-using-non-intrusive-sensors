# Daily-Human-Activity-Recognition-using-non-intrusive-sensors

python3 PrepareData.py
python3 Training.py --v biLSTM_H > attempt1.log

# Setup and demo

- Clone this repo to your local machine using 
 ```
 git clone https://github.com/raugom13/Daily-Human-Activity-Recognition-using-non-intrusive-sensors.git
  ```
  
- Execute this line to convert the raw data into .npy format and split it to test, train and validation.
 ```
 python3 PrepareData.py
 
  ```
- Execute this line to perform the training of the neural network. In addition, you can select the architecture of the network (biLSTM_H) and save the relevant information of the training into a log file (attempt1.log).
 ```
 python3 Training.py --v biLSTM_H > attempt1.log
  ```
