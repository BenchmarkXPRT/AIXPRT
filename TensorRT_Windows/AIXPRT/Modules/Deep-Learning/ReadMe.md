### 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to image classification and object detection using TensorRT optimizations.
It has workloads “ResNet50_v1”, "SSD-MobileNet-v1" and can run single and multi-batchsize and muti-instance scenarios.

## 2. System Requirements
For NVIDIA Discrete Graphics Cards
* Operating System:
	Windows
* GPU:
	  [CUDA enabled NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)

## 3. Run Benchmark

#### Steps to configure the machine and workloads

1. Install dependencies:
   * These workloads will require CUDA, cuDNN and TensorRT . Please install the appropriate version of Cuda and cudnn associated with the TensorRT version. Benchmark is tested with TensorRT 6 and does not support older version.

    [Note] The instructions below assume the user is not connected to Internet via Proxy.
   * Install [Visual Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019). During the installation of Visual Studio please select the below option in the installation wizard and install. <br />
        a. "Workloads" tab select "Desktop development with C++" <br />
        b. "Individual components" tab select "MSBuild" <br />

   * Install CUDA [CUDA](https://developer.nvidia.com/cuda-downloads)

   * Restart the system after installing CUDA

   * Install [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)

   * Install [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-zip)

   * Install [python3](https://www.python.org/downloads/)

   * Download and run  [OpenCV version 2](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.13/). This will extract OpenCV-2. Add \opencvDownloadFolders\opencv\build\x64\vc14\bin to System Environment Variables path.

   * Run this command in Windows command line `pip install pillow numpy pywin32 wmi opencv-python`

2. Setup AIXPRT environment variable

   * On Windows search, type "Advanced system Settings" and open View Advanced system Settings .Click on  "Environment Variable" .
   * In "System variable" section select "New". Set variable name to AIXPRT_INCLUDE and add the paths below on your system for variable values separated by a semicolon. Save the environment variable after adding.


      ```

      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cudaVersion}\include
      {path_to_TensorRTDownload}\TensorRT-xx.Windows10.x86_64.cuda-yy.cudnn-zz\TensorRT-{version}\samples\common\windows
      {path_to_TensorRTDownload}\TensorRT-xx.Windows10.x86_64.cuda-yy.cudnn-zz\TensorRT-{version}\samples\common
      {path_to_TensorRTDownload}\TensorRT-xx.Windows10.x86_64.cuda-yy.cudnn-zz\TensorRT-{version}\include
      {path_to_}\python\AppData\Local\Programs\Python\Python3x\include
      {path_to_python}\AppData\Local\Programs\Python\Python3x\libs
      {path_to_opencvDownload}\opencv\sources\include
      {path_to_opencvDownload}\opencv\build\include
      {path_to_opencvDownload}\opencv\build\include\opencv2
      {path_to_opencvDownload}\opencv\build\include\opencv2\core

      ```

   * In "System variable" section select "New". Set variable name to AIXPRT_LINKER and the paths below on your system for variable values separated by a semicolon. Save the environment variable after adding.


      ```

      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cudaVersion}\lib\x64\*.lib
      {path_to_python}\AppData\Local\Programs\Python\Python3x\libs\*.lib
      {path_to_TensorRTDownload}\TensorRT-xx.Windows10.x86_64.cuda-yy.cudnn-zz\TensorRT-{version}\lib\*.lib
      {path_to_opencvDownload}\opencv\build\x64\vc14\lib\*.lib

      ```


3. Compile the sources

	* Navigate to  <AIXPRT_Directory>/Modules/Deep-Learning/workloads/commonsources/bin/src/MultiStream_ResNet and open the .sln (Microsoft Visual Studio Solution) file in Visual Studio.

  * On top-right side of the Visual Studio UI, change "Solution Configuration" to "Release". Click the green button which says "Local Windows Debugger" to compile the source. Once the code compiles a console window opens to confirm the success, press any button to close that console.

  * Navigate to  <AIXPRT_Directory>/Modules/Deep-Learning/workloads/commonsources/bin/src/MultiStream_ssd and open the .sln (Microsoft Visual Studio Solution) file in Visual Studio.

  * On top-right side of the Visual Studio UI, change "Solution Configuration" to "Release". Click the green button which says "Local Windows Debugger" to compile the source. Once the code compiles a console window opens to confirm the success, press any button to close that console.


#### Steps to run the benchmark
 1. Navigate to directory:<br />
 	`cd /{path_to}/AIXPRT/Harness/`

 2. Run the benchmark:<br />
	`python3 index.py`


   On Windows, the python3 command is not recognized by default. Please make a copy of your python.exe and rename as python3.exe

## 3. Results

When the test is complete, the benchmark saves the results to AIXPRT/Results in JSON format, and also generates CSV files with the name {ConfigName}_RESULTS_SUMMARY.csv
To submit results, please follow the instructions in AIXPRT/ResultSubmission.md or at [AIXPRT Results Submission](https://github.com/BenchmarkXPRT/Public-AIXPRT-Resources/blob/master/OtherDocuments/ResultSubmission.md)


##### Sample results summary file <br/>

   Each results summary file has three sections: SYSTEM INFORMATION, RESULTS SUMMARY and DETAILED RESULTS.<br/>
   1. SYSTEM INFORMATION <br/>
   This section provides basic information about the system under test. <br/>

   ![alt text](https://github.com/BenchmarkXPRT/Public-AIXPRT-Resources/blob/master/assets/tensorflow_systemInfo.png)

   2. RESULTS SUMMARY <br/>
   AIXPRT measures inference latency and throughput for image recognition (ResNet-50) and object detection (SSD-MobileNet) tasks. Batching tasks allows AI applications to achieve higher levels of throughput, but higher throughput may come at the expense of increased latency per task. In real-time or near real-time use cases like performing image recognition on individual photos being captured by a camera, lower latency is important to enable better user experience. In other cases, like performing image recognition on a large library of photos, higher throughput through batching images or concurrent instances may allow faster completion of the overall workload. The achieve optimal latency and/or throughput levels, AI applications often tune batch sizes and/or concurrent instances according to a system’s hardware capabilities, such as the number of available processor cores and threads.To represent a spectrum of common tunings, AIXPRT tests AI tasks in different batch sizes (1 - 32 is the default in this package) that are relevant to the target test system.
   AIXPRT then reports the maximum throughput and minimum latency for image recognition (ResNet-50) and object detection (SSD-MobileNet v1)usages.<br/>
   The AIXPRT results summary (example below) makes it easier to quickly identify relevant comparisons between systems. <br/>

   ![alt text](https://github.com/BenchmarkXPRT/Public-AIXPRT-Resources/blob/master/assets/results_summary.png)


   3. DETAILED RESULTS <br/>
   This section shows the throughput and latency results for each AI task configuration tested by the benchmark.
   AIXPRT runs each AI task (e.g. ResNet-50, Batch1, on CPU) multiple times and reports the average inference throughput and corresponding latency percentiles.

   ![alt text](https://github.com/BenchmarkXPRT/Public-AIXPRT-Resources/blob/master/assets/detailed_results.png)
