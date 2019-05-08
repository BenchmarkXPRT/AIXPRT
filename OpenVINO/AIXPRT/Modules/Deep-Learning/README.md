
## 1. Introduction
This module contains workloads to evaluate the system performance of use cases related to Image Classification and Object detection using OpenVINO.
It has workloads “resnet50_v1” and "ssd-mobilenet" and can run Single Batch, Multi-Batch and Multi Instance scenarios.

## 2. System Requirements

* **Operating Systems**: Ubuntu 16.04 LTS, Windows 10
* **CPU**: 
	  6th to 8th generation Intel Core and Intel Xeon processors 
    Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics 
* **GPU**:
	  6th to 8th generation Intel Core processor with Iris® Pro graphics and Intel HD Graphics 
    6th to 8th generation Intel Xeon processor with Iris Pro graphics and Intel HD Graphics (excluding the e5 product family, which does not have graphics) 
* **VPU**: 
    Intel Movidius Neural Compute Stick, HDDL-r  ( Only Batch size 1 is supported, and on **Linux**)
     
## 3. Steps to Run Benchmark

### Installation And System Setup
#### 1. Download or Clone the AIXPRT Github repository.

``` git clone [url]```

#### 2. Install OpenVINO Distribution:
   * Install [OpenVINO](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-windows) full package version , following the instructions for [Windows](https://software.intel.com/en-us/articles/OpenVINO-Install-Windows) or [Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).
  NOTE : AIXPRT is developed and tested with OpenVINO 2019.R1 and 2018.R5
  
       **NB**: Please make sure to follow the instructions exactly as laid out in the OpenVINO install guide for your version.
       
#### 3. Build the workloads

##### (a) Linux
	
   * Install python3 packages
      ```
      sudo apt update
      sudo apt-get install python3-pip python3-numpy git
      pip3 install Pillow opencv-python
      ```
  
   * Give executable permissions to ```compile_AIXPRT_sources.sh```, ```install_myriad_bootrules.sh``` and ```install_GPU_VPU_HDDLR_dep.sh``` scripts located at ~/AIXPRT/install.
   * Make sure you have an active internet connection.
   
   
    ```
    cd ~/AIXPRT/install
    sudo ./compile_AIXPRT_sources.sh </path/to/AIXPRT/> </path/to/OpenVINO/>
    sudo ./install_GPU_VPU_HDDLR_dep.sh
    ```
    
    
   Above steps will build the workloads with installed openVINO.
   
   **Note**: If inference is on Compute sticks (NCS1, NCS2), run ```install_myriad_bootrules.sh``` instead of ```install_GPU_VPU_HDDLR_dep.sh```
    
    
##### (b) Windows

   
   * Install pillow, numpy, pywin32, wmi and opencv
      ```
      pip install pillow numpy pywin32 wmi opencv-python
      
      ```
      
   * Build the workloads: 
   
    ```
    cd AIXPRT/install
    compile_AIXPRT_sources.bat </path/to/AIXPRT> </path/to/OpenVINO>
    ```
    
   * On Windows, the ```python3``` command is not recognised  by default. Please make a copy of your python.exe and rename as python3.exe
   
#### Run benchmark
 1. Navigate to directory:
 
    ```
    cd AIXPRT/Harness
    ```  
 2. Run the benchmark:
 
    ```
    python3 index.py
    ```
    
    * On Linux, If the above command fails , please try running with  ```sudo ``` 
    
   
***Note***: Above step will run a the benchmark with configuration with CPU as target, fp32 precision on batches 1,2,4,8,16,32,64 and 128 for resnet-50 and ssd-mobilenet.
   If you wish to change the configuration , please edit the json file under AIXPRT/Config/. Instructions to edit the configuration json are [here](TODO: Add path to /AIXPRT/EditConfig.md). 
   
### Results

Benchmark runs and finishes pointing to the results location. 
All the results are located at AIXPRT/Results/ after the benchmark run. 


## 4. Known issues
- On Windows, when input paths (either to AIXPRT or openVINO install directory) contains spaces, they must be enclosed in double quotations when being passed to the batch script. For ease of use, please enclose all input paths in double quotations, eg: 

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






