<h1 align="center"><img src="https://github.com/BenchmarkXPRT/AIXPRT/blob/master/AIXPRT-header.png" alt="AIXPRT Header" /></h1>
<h4 align="center">
  <i>
    An easy-to-use public AI benchmark designed and developed by
  <a href="https://www.principledtechnologies.com/benchmarkxprt/">BenchmarkXPRT community</a>
   </i>
</h4>

<hr>


## About
[AIXPRT](https://www.principledtechnologies.com/benchmarkxprt/aixprt/) is an AI benchmark tool that makes it easier to evaluate a system's machine learning inference performance by running common image-classification, object-detection, and recommender system workloads.

AIXPRT includes support for the Intel OpenVINO, TensorFlow, and NVIDIA TensorRT toolkits to run image-classification and object-detection workloads with the ResNet-50 and SSD-MobileNet v1 networks, as well as a Wide and Deep recommender system workload with the Apache MXNet toolkit. The test reports FP32, FP16, and INT8 levels of precision. Test systems must be running Ubuntu 18.04 LTS or Windows 10, and the minimum CPU and GPU requirements vary by toolkit. You can find more detail on hardware and software requirements in the installation package's ReadMe files.




## Support
|<b> SDK/Frameworks </b> |OpenVINO|TensorRT|TensorFlow|MXNet
|--| ------ | -------|--------  | ----|
|<b> Hardware </b> |Intel CPU,<br/> Intel GPU,<br/> Intel Neural Compute Stick 2,<br/> Intel Vision Accelerator|NVIDIA GPU,<br/> NVIDIA Xavier <br/> |AMD CPU,<br/> AMD GPU,<br/> Intel CPU,<br/> NVIDIA GPU,<br/> NVIDIA Xavier| AMD CPU,<br/> Intel CPU,<br/> NVIDIA GPU|
|<b> OS </b> |Ubuntu 18.04,<br/> Windows 10| Ubuntu 18.04,<br/> Windows 10 |Ubuntu 18.04,<br/> Windows 10| Ubuntu18.04 |




## Run the latest version of AIXPRT
Please use the [package selector tool](https://www.principledtechnologies.com/benchmarkxprt/aixprt/guide.php) to download the appropriate one for your test system. A ReadMe file is provided along with each package with instructions for how to set up and run the benchmark.




## Results
  1. Below is how a snapshot of sample result summary. More details about the results can be found in the package's README.md file.
  <h1 align="center"><img src="https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/Harness/assets/results_summary.png" /></h1>

  2. For already available results visit[ AIXPRT results](https://www.principledtechnologies.com/benchmarkxprt/aixprt/2019/results) page.
  3. To submit results to our page, please follow these [instructions](https://www.principledtechnologies.com/benchmarkxprt/aixprt/2019/submit-results.php).




## Contribution guidelines

   - #### Instructions for downloading the AIXPRT repository

   The AIXPRT repository contains large files, over 50MB in size, so the package, git-lfs, must be installed and the repository must be cloned. (A zip file of the repository will not include the large files.)  

Install git lfs and clone.
Instructions are found at https://packagecloud.io/github/git-lfs/install and are listed in the following 3 steps

   1.  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   2.  sudo apt-get install git-lfs
   3.  git clone https://github.com/BenchmarkXPRT/AIXPRT.git (You may need to enter your credentials for each large file)
   4.  For environment setup, follow the steps in the README.md file in the Modules/Deep-Learning/ directory for each platform

   - #### Add a workload
1. Workloads on this git repository are grouped by framework. To begin, please pick a framework for the new workload. If it's a new framework, create a new folder with the framework name.
2. Follow the guidelines provided in [this](https://github.com/BenchmarkXPRT/AIXPRT/blob/master/Tensorflow/AIXPRT/Add_Edit_Workload.md) document to edit an existing workload or add a new one.
3. Once the workload is ready, contact BenchmarkXPRTsupport@principledtechnologies.com with your submission.


## Resources

* [Understanding AIXPRT batch size](https://www.principledtechnologies.com/benchmarkxprt/blog/2019/08/08/understanding-aixprt-batch-size/)
* [Understanding AIXPRT precision settings](https://www.principledtechnologies.com/benchmarkxprt/blog/2019/09/05/understanding-the-basics-of-aixprt-precision-settings/)
* [Understanding concurrent instances in AIXPRT](https://www.principledtechnologies.com/benchmarkxprt/blog/2019/09/12/understanding-concurrent-instances-in-aixprt/)
* [Understanding AIXPRT results](https://www.principledtechnologies.com/benchmarkxprt/blog/2019/08/01/understanding-aixprt-results/)
* [Navigating the AIXPRT results table](https://www.principledtechnologies.com/benchmarkxprt/blog/2019/05/30/improvements-to-the-aixprt-results-table/)
