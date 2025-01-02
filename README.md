# MNIST Dataset Test with Amazon EC2

## Project Overview
This project leverages the MNIST dataset to test and evaluate the performance of a convolutional neural network (CNN) model on an Amazon EC2 instance. The focus is on implementing techniques to optimize training, reduce parameters, and increase accuracy.

## Features
- **Dataset**: MNIST dataset of handwritten digits.
- **Environment**: Deployed on Amazon EC2.
- **Model**: Convolutional Neural Network (CNN).
- **Optimization Techniques**: Group normalization, learning rate schedules, and parameter reduction strategies.

## Prerequisites
1. An Amazon EC2 instance set up with the required dependencies.
2. Python 3.8 or above installed.
3. Virtual environment tools like `venv` or `conda`.
4. GPU support (if applicable).

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Set up a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset:
   - The MNIST dataset is automatically downloaded using PyTorch's `torchvision.datasets.MNIST` module.

2. Train the model:
   ```bash
   python train.py
   ```
   
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
- **Training Accuracy**: Achieved 99.27% accuracy after 15 epochs.
- **Test Accuracy**: Achieved 99.4% accuracy on the test set.


## Model 1
Targets: 
Achieve good accuracy with no restriction in parameters.

Results: 
Parameters: 13808
Best Train Accuracy: 98.42
Best Test Accuracy: 99.40% (8th Epoch), 99.40% (12th Epoch)

Analysis:
Architecture is fine, need to reduce the parameters

## Model 2
Targets:
Needs to reduce the parameters while maintaining the accuracy

Results:
Parameters: 5636
Best Train Accuracy: 98.67
Best Test Accuracy: 99.34% (7th Epoch), 99.36% (10th Epoch)

Analysis:
Accuracy is not able to be reached above 99.36 and there are overfitting problem after 10th epochs. So needs to find different alternative

## Model 3
Targets:

Results:
Parameters: 13808
Best Train Accuracy: 98.42
Best Test Accuracy: 99.40% (8th Epoch), 99.40% (12th Epoch)

Analysis:

## Model 4
Targets:

Results:
Parameters: 7776
Best Train Accuracy: 98.70
Best Test Accuracy: 99.48% (12th Epoch), 99.46% (14th Epoch)

Analysis:
We can acheive 99.4 test accuracy within 15 epochs with lower than 8k parameters

### Sample Logs
```

On EC2 Instance
EPOCH: 0
Loss=0.08195843547582626 Batch_id=468 Accuracy=89.26: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 92.39it/s]

Test set: Average loss: 0.0886, Accuracy: 9760/10000 (98%)

EPOCH: 1
Loss=0.03760998323559761 Batch_id=468 Accuracy=97.79: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 93.50it/s]

Test set: Average loss: 0.0508, Accuracy: 9879/10000 (99%)

EPOCH: 2
Loss=0.0626540258526802 Batch_id=468 Accuracy=98.32: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 94.87it/s]

Test set: Average loss: 0.0383, Accuracy: 9898/10000 (99%)

EPOCH: 3
Loss=0.014102893881499767 Batch_id=468 Accuracy=98.53: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 93.88it/s]

Test set: Average loss: 0.0314, Accuracy: 9906/10000 (99%)

EPOCH: 4
Loss=0.04098793491721153 Batch_id=468 Accuracy=98.74: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 94.25it/s]

Test set: Average loss: 0.0292, Accuracy: 9914/10000 (99%)

EPOCH: 5
Loss=0.022838272154331207 Batch_id=468 Accuracy=98.73: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 93.63it/s]

Test set: Average loss: 0.0309, Accuracy: 9919/10000 (99%)

EPOCH: 6
Loss=0.049984320998191833 Batch_id=468 Accuracy=99.11: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 96.08it/s]

Test set: Average loss: 0.0240, Accuracy: 9933/10000 (99%)

EPOCH: 7
Loss=0.04015188291668892 Batch_id=468 Accuracy=99.14: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 94.24it/s]

Test set: Average loss: 0.0224, Accuracy: 9938/10000 (99%)

EPOCH: 8
Loss=0.012840651907026768 Batch_id=468 Accuracy=99.18: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 91.07it/s]

Test set: Average loss: 0.0225, Accuracy: 9941/10000 (99%)

EPOCH: 9
Loss=0.01096449326723814 Batch_id=468 Accuracy=99.21: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 92.11it/s]

Test set: Average loss: 0.0225, Accuracy: 9942/10000 (99%)

EPOCH: 10
Loss=0.00868392363190651 Batch_id=468 Accuracy=99.19: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 91.93it/s]

Test set: Average loss: 0.0226, Accuracy: 9939/10000 (99%)

EPOCH: 11
Loss=0.016592305153608322 Batch_id=468 Accuracy=99.25: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 93.38it/s]

Test set: Average loss: 0.0219, Accuracy: 9939/10000 (99%)

EPOCH: 12
Loss=0.00719126733019948 Batch_id=468 Accuracy=99.22: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 94.68it/s]

Test set: Average loss: 0.0222, Accuracy: 9939/10000 (99%)

EPOCH: 13
Loss=0.008978242985904217 Batch_id=468 Accuracy=99.22: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 94.13it/s]

Test set: Average loss: 0.0218, Accuracy: 9938/10000 (99%)

EPOCH: 14
Loss=0.019389057531952858 Batch_id=468 Accuracy=99.27: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 95.30it/s]

Test set: Average loss: 0.0218, Accuracy: 9936/10000 (99%)

On Local CPU


EPOCH: 0
Loss=0.07516708970069885 Batch_id=937 Accuracy=91.63: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:08<00:00, 13.73it/s]

Test set: Average loss: 0.0471, Accuracy: 9871/10000 (99%)

EPOCH: 1
Loss=0.07853762805461884 Batch_id=937 Accuracy=97.15: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:14<00:00, 12.59it/s] 

Test set: Average loss: 0.0471, Accuracy: 9845/10000 (98%)

EPOCH: 2
Loss=0.2707858681678772 Batch_id=937 Accuracy=97.74: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:11<00:00, 13.19it/s] 

Test set: Average loss: 0.0309, Accuracy: 9905/10000 (99%)

EPOCH: 3
Loss=0.04555179551243782 Batch_id=937 Accuracy=97.92: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:11<00:00, 13.17it/s] 

Test set: Average loss: 0.0274, Accuracy: 9923/10000 (99%)

EPOCH: 4
Loss=0.2997715473175049 Batch_id=937 Accuracy=97.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:10<00:00, 13.38it/s] 

Test set: Average loss: 0.0319, Accuracy: 9906/10000 (99%)

EPOCH: 5
Loss=0.044416915625333786 Batch_id=937 Accuracy=98.13: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:10<00:00, 13.34it/s] 

Test set: Average loss: 0.0249, Accuracy: 9927/10000 (99%)

EPOCH: 6
Loss=0.23659227788448334 Batch_id=937 Accuracy=98.58: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:10<00:00, 13.37it/s] 

Test set: Average loss: 0.0187, Accuracy: 9941/10000 (99%)

EPOCH: 7
Loss=0.005379190668463707 Batch_id=937 Accuracy=98.63: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:13<00:00, 12.85it/s] 

Test set: Average loss: 0.0185, Accuracy: 9947/10000 (99%)

EPOCH: 8
Loss=0.05623600631952286 Batch_id=937 Accuracy=98.67: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:23<00:00, 11.26it/s] 

Test set: Average loss: 0.0179, Accuracy: 9944/10000 (99%)

EPOCH: 9
Loss=0.02388313226401806 Batch_id=937 Accuracy=98.69: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:17<00:00, 12.12it/s] 

Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99%)

EPOCH: 10
Loss=0.010712970048189163 Batch_id=937 Accuracy=98.73: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:22<00:00, 11.31it/s] 

Test set: Average loss: 0.0172, Accuracy: 9944/10000 (99%)

EPOCH: 11
Loss=0.0042505040764808655 Batch_id=937 Accuracy=98.74: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:22<00:00, 11.35it/s] 

Test set: Average loss: 0.0180, Accuracy: 9944/10000 (99%)

EPOCH: 12
Loss=0.041792675852775574 Batch_id=937 Accuracy=98.80: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:14<00:00, 12.63it/s] 

Test set: Average loss: 0.0179, Accuracy: 9948/10000 (99%)

EPOCH: 13
Loss=0.008874033577740192 Batch_id=937 Accuracy=98.77: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:18<00:00, 11.97it/s] 

Test set: Average loss: 0.0172, Accuracy: 9947/10000 (99%)

EPOCH: 14
Loss=0.04638266935944557 Batch_id=937 Accuracy=98.70: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:23<00:00, 11.17it/s] 

Test set: Average loss: 0.0174, Accuracy: 9946/10000 (99%)
```

## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 12, 24, 24]             864
              ReLU-6           [-1, 12, 24, 24]               0
       BatchNorm2d-7           [-1, 12, 24, 24]              24
           Dropout-8           [-1, 12, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             120
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]           1,080
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
      BatchNorm2d-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           1,728
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 7,776
Trainable params: 7,776
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.53
Params size (MB): 0.03
Estimated Total Size (MB): 0.56
----------------------------------------------------------------
```
