## 1. Introduction
This module contains workloads to evaluate the system performance using use cases related to Image Classification and Object Detection using Tensorflow. It has workloads “resnet50_v1” and “ssd_mobilenet” and can run Single , Multi-Batch and Multi-Instance scenarios.
Workloads are build and tested using Tensorflow (version 1.12) framework. For More information about Tensorflow please follow the link: https://www.tensorflow.org. Workloads run with fp32 precision by default.

## 2. System Requirements
 This Module can run on all the systems supported by tensorflow.
 
## 3. Run Benchmark

##### Steps to configure the machine 
####### Ubuntu
1. Clone the AIXPRT Github repository.

2. Install dependencies:
    ```
    sudo apt-get update
    sudo apt-get install python3 python3-numpy python3-pil
    sudo apt-get install python python-numpy python-pil
    sudo apt install python3-pip
    pip3 install opencv-python
    ```
3. Install Tensorflow
   * For instructions to install on Intel CPU and AMD CPU, follow [Tensorflow Website](https://www.tensorflow.org/install/)    NVIDIA GPU NVIDIA [Tensorflow GPU Website](https://www.tensorflow.org/install/gpu)

   * To install Tensorflow with AMD ROCm support follow the instructions [AMD ROCM Tensorflow](https://rocm.github.io/dl.html)
   NOTE : on AMD-GPU , Ubuntu 18.04 has the support for latest drivers and is recommend to use Ubuntu 18.04.
   
   * Below are some simple instruction to run the benchmark. However users are free to choose to install any different type of tensorflow according to the system they are running on.
   
    ```
    # CPU
    sudo apt-get install python3-pip
    pip3 install tensorflow
     ```

    ```
    # GPU
    sudo apt-get purge nvidia*                # remove any installed drivers
    sudo add-apt-repository ppa:graphics-drivers/ppa    # get the repository
    sudo apt update                        # update the apt request
    ubuntu-drivers devices                    # confirm desired driver is present
    sudo apt install nvidia-410                # install the desired driver
    reboot
    nvidia-smi                          # check that the desired driver version is installed as below (410.xx)
    sudo apt-get install python3-pip
    pip3 install tensorflow-gpu
     ```
####### Windows
* Install dependencies 
  1. [Python3 for windows](https://www.python.org/downloads/windows/) .Please make sure to install the right version of python3 that is supported by tensorflow.
  
  2. Navigate to python3 install directory and duplicate python.exe file. Rename the duplicate file to python3.exe
  
  3. Install dependencies 
     ```
     pip3 install pyreadline
     pip3 install Pillow
     pip3 install opencv-python
     pip3 install --upgrade wmi
     pip3 install --upgrade pypiwin32
     
     ```
   4. Install Tensorflow 
   
   
     For instructions to install on Intel CPU and AMD CPU, follow [Tensorflow Website](https://www.tensorflow.org/install/pip)
     NVIDIA GPU NVIDIA [Tensorflow GPU Website](https://www.tensorflow.org/install/gpu#windows_setup). 
     
       * Below are some simple instruction to run the benchmark. However users are free to choose to install any different type of 
         tensorflow according to the system they are running on.
   
         ```
         # CPU
         pip3 install tensorflow
          ```

         ```
         # GPU
          ```
          a. Install CUDA which will also update the drivers. Please pick the version as recommended on 
             [Tensorflow](https://www.tensorflow.org/install/gpu#software_requirements) gpu setup
          b. Install CUDNN as recomended on the page above 
          c. Install TensorRT
          d. pip3 install tensorflow-gpu
       
       
       * To check the installation run ` pip3 list ` , output should contain tensorflow as one of the item.
     


##### Steps to run benchmark
 1. Navigate to directory:
 
    ```
    cd AIXPRT/Harness
    ```
    
 2. Run the benchmark:
 
    ```
    python3 index.py
    
    ```
 3. If running on GPU target , please edit AIXPRT/Config/{filename.json} to set "hardware" to gpu .  
##### Results

Benchmark runs and finished pointing to the results location. 
All the results are located at AIXPRT/Results/ after the benchmark run. 


