_**Support Vector Machine**_

_Simple SVM from scratch using CVXOPT._

Based on: D. J. Sebald and J. A. Bucklew, "Support vector machine techniques for nonlinear equalization," in IEEE Transactions on Signal Processing, vol. 48, no. 11, pp. 3217-3226, Nov. 2000.

_Dependency:_
numpy,
cvxopt,
matplotlib,
sklearn (For convenience)


**_Usage:_** 

python3 main.py -h
usage: main.py [-h] [--test_type TEST_TYPE] [--test_number TEST_NUMBER]

Support Vector Machine from Scratch

optional arguments:
  -h, --help            show this help message and exit
  --test_type TEST_TYPE
                        Select test's type from: [linear, non_linear]
  --test_number TEST_NUMBER
                        Insert ID for non linear dataset: [1:RandomNonLinear,
                        2:XDataset, 3:MoonDataset, 4:CirclesDataset,
                        6:IrisDataset]


**_Example:_**

python3 main.py --test_type linear
python3 main.py --test_type non_linear --test_number 1