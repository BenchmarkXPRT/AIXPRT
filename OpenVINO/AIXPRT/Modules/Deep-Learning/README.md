## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object detection using OpenVINO.
It has workloads “resnet50_v1” and "ssd-mobilenet" and can run Single Batch, Multi-Batch and Multi Instance scenarios.

## 2. System Requirements

* **Operating Systems**: Ubuntu 18.04 LTS, Windows 10
* **CPU**: 
	  6th to 10th generation Intel Core and Intel Xeon processors 
    Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics 
* **GPU**:
	  6th to 11th generation Intel Core processor with Iris® Pro graphics and Intel HD Graphics 
    6th to 8th generation Intel Xeon processor with Iris Pro graphics and Intel HD Graphics (excluding the e5 product family, which does not have graphics) 
* **VPU**: 
    Intel Movidius Neural Compute Stick, HDDL-r  ( Only Batch size 1 is supported, and on **Ubuntu**)
     
## 3. Steps to Run Benchmark

### Installation And System Setup
#### 1. Unzip the AIXPRT installation package
       
#### 2. Install dependencies

##### (a) Linux
	
   * Install python3 packages
      ```
      sudo apt update
      sudo apt-get install python3-pip python3-numpy git python3-opencv
      pip3 install Pillow opencv-python
      ```
  
   * If running on GPU target , install latest GPU drivers by running the below commands. Reboot the machine after installation.
       
       ```
         sudo add-apt-repository ppa:intel-opencl/intel-opencl
         sudo apt-get update
         sudo apt-get install intel-opencl
       ```

   * If running on Myriad target, Install [OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) full package version , following the instructions for [Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).<br/>

    Give executable permissions to ```install_myriad_bootrules.sh``` scripts located at ~/AIXPRT/install. Make sure you have an active internet connection.

          ```
            cd ~/AIXPRT/install
            sudo ./compile_AIXPRT_sources.sh </path/to/AIXPRT/> </path/to/OpenVINO/>
            sudo ./install_myriad_bootrules.sh
          ``` 

   * If running on HDDL targets , Install [OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) full package version , following the instructions for [Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).<br/>
   
   Give executable permissions to ```install_HDDLR_dep.sh``` scripts located at ~/AIXPRT/install. Make sure you have an active internet connection.

         ```
            cd ~/AIXPRT/install
            sudo ./compile_AIXPRT_sources.sh </path/to/AIXPRT/> </path/to/OpenVINO/>
            sudo ./install_HDDLR_dep.sh
            
         ``` 

   * Run the benchmark
         ```
            cd ~/AIXPRT/Harness
            python3 index.py

         ``` 


##### (b) Windows
   * Install dependencies 
          ```
          cd AIXPRT/install
          setup_AIXPRT.bat 
    
          ```

   * Run the benchmark
         ```
            cd AIXPRT/Harness
            python3 index.py
            
         ``` 
   * During the installation process, please review any prompts and allow the installation of necessary dependencies.

***Note***: Above step will run the benchmark with configuration with CPU as target, int8 precision on batches 1,2,4,8,16 and 32 for resnet-50 and ssd-mobilenet.
   If you prefer to change the configuration , please edit the json file under AIXPRT/Config/. Instructions to edit the configuration json are [here](https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/EditConfig.md). 
   
## 3. Results

When the test is complete, the benchmark saves the results to AIXPRT/Resultsin JSON format, and also generates CSV files with the name {ConfigName}_RESULTS_SUMMARY.csv
To submit results, please follow the instructions in AIXPRT/ResultSubmission.md or at https://www.principledtechnologies.com/benchmarkxprt/aixprt/2019/submit-results.php.


##### Sample results summary file <br/>

Each results summary file has three sections: SYSTEM INFORMATION, RESULTSSUMMARY and DETAILED RESULTS.<br/>
 1. SYSTEM INFORMATION <br/>
    This section provides basic information about the system under test. <br/>
    ![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/tensorflow_systemInfo.png)

 2. RESULTSSUMMARY <br/>
    AIXPRT measures inference latency and throughput for image recognition (ResNet-50) and object detection (SSD-MobileNet) tasks. Batching tasks allows AI applications to achieve 
higher levels of throughput, but higher throughputmay come at the expense of increased latency per task. In real-time or near real-time use cases like performing image recognition 
on individual photos being captured by a camera, lowerlatency is important to enable better user experience. In other cases, like performing image recognition on a large library of 
photos, higher throughput through batching images or concurrent instancesmay allowfaster completion of the overall workload. The achieveoptimal latency and/or throughput levels, 
AI applications often tune batch sizes and/or concurrent instances according to a system’s hardware capabilities, such as the number of availableprocessor cores and threads.
To represent a spectrumof commontunings, AIXPRT tests AI tasks in different batch sizes (1 –32 is the default in this package) that are relevant tothe target test system. 
AIXPRT thenreports the maximum throughput and minimum latency for image recognition (ResNet-50) and object detection (SSD-MobileNet v1) usages.<br/>
The AIXPRT results summary (example below) makes iteasier to quickly identify relevant comparisons between systems. <br/>

 ![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/results_summary.png)


 3. DETAILED RESULTS <br/>
   This section shows the throughput and latency results for each AItask configuration testedbythebenchmark. 
AIXPRT runs each AI task (e.g. ResNet-50, Batch1, on CPU) multiple times andreports the average inference throughput and correspondinglatencypercentiles.

![alt text](https://github.com/BenchmarkXPRT/AIXPRT/tree/master/Tensorflow/AIXPRT/Harness/assets/detailed_results.png)


## 4. Known issues
- On Windows, when input paths (either to AIXPRT or OpenVINO install directory) contains spaces, they must be enclosed in double quotations when being passed to the batch script. For ease of use, please enclose all input paths in double quotations, eg: 

   ```compile_AIXPRT_sources.bat "C:\Users\[user]\AIXPRT\" "C:\Intel\computer_vision_sdk\"```

- If your system hosts HDDLr, please do not connect NCS1 or 2. As yet, inference on NCS cannot be done on a system hosting an HDDLr

- Apt update commands in install/.sh scripts may not execute properly, **causing installation issues**

     For instance, when `sudo apt update` fails with `E: Could not get lock /var/lib/apt/lists/lock`. 
     - **Fix**: Run `sudo rm /var/lib/dpkg/lock /var/cache/apt/archives/lock /var/lib/apt/lists/lock` before running setup scripts
     
- **Windows**: On systems with Intel HD Graphics 620/630 with outdated drivers, the following warnings appear

   ```
   Running ssd_mobilenet batch1 fp32 GPU
  
   warning: Linking two modules of different data layouts: '' is 'e-m:e-i64:64-f80:128-n8:16:32:64-S128' whereas '<origin>' is 'e-   i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32'```
   warning: Linking two modules of different target triples: ' is 'x86_64-pc-windows-msvc' whereas '<origin>' is 'igil_64_GEN9'```

Updating with latest drivers should resolve this warning.






