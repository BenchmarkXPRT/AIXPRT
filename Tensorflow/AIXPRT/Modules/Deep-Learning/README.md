## 1. Introduction
This module contains workloads to evaluate the system performance using use cases related to Image Classification using Tensorflow. It has workloads “resnet50_v1” and “ssd_mobilenet” and can run Single and Multi-Batch size scenarios.
Workloads are build and tested using Tensorflow (version 1.11) framework. For More information about Tensorflow please follow the link: https://www.tensorflow.org. Workloads run with fp32 precision.

## 2. System Requirements
 This Module can run on all the systems supported by tensorflow.
 
## 3. Run Benchmark

##### Steps to configure the machine 
1. Checkout the AIXPRT Github repository or download as zip . If you download this module as zip , please make sure that the root folder name is AIXPRT.

2. Install dependencies:
    ```
    sudo apt-get install python3 python3-numpy python3-pil
    ```
3. Install python3 version of Tensorflow
   * For instructions to install on Intel CPU , AMD CPU and NVIDIA GPU , follow [Tensorflow Website](https://www.tensorflow.org/install/). 

   * To install Tensorflow with AMD ROCm follow the instructions [AMD ROCM Tensorflow](https://rocm.github.io/dl.html)
    
##### Steps to run benchmark
 1. Navigate to directory:
 
    ```
    cd AIXPRT/Harness
    ```
    
 2. Run the benchmark:
 
    ```
    python3 index.py
    ```
   
##### Results

Benchmark runs and finished pointing to the results location. 
All the results are located at AIXPRT/Results/ after the benchmark run. 


## 4. Known issues
 - Fp16 and int8 precisions are not supported. 



