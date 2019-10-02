### Steps to add a workload

1. Create a folder with your workload name. Add "bin" and  "data" folders to it. Also add workload_details.json which would describe your workload. Here is a sample [workload_details.json](TODO : Add the url to one workload_details.json).
 In the json, all the keys must remain same , only values must change according to your workload. ID's of the workloads are in the [resultSchema](TODO: add url to /Harness/ResultSchemaFiles/Deep-Learning/result_schema.json) file in Harness.

2. Add your workload  to resultSchema File present in harness so that the results table generated is consistent across. Choose a workload ID following the series.

3. Create a python script file which can start your workload and add its name as a value to "script" key in workload_details.json file. Harness will run the workload using this script file.

4. All the binary and script files mush be placed in bin/ folder . And all the source that is used to generate the bin should be at bin/src/ .

5. Any data used in the workload ( example : images , input textFiles , input sound files etc) should be placed in data/ folder

6. Please follow the [Measurment Methodology](TODO : add url to /Harness/assets/measurment_method.pdf) and [Result Caluculation](TODO : add url to /Harness/assets/result_calculation.pdf) to obtain a workload result.

7. Call the result API in harness to generate the workload result from your script file after running the workload.
  API ==> [resultapi.py](TODO : add the url to "createResultJson" method in /Harness/resultsapi.py)

8. Verify if it works

```
cd AIXPRT/Harness
python3 index.py

```
Once the run is completed, the application closes. At this time, please go to /AIXPRT/Results/{ConfigFileName} folder to find the results in a json format. The result file name will be of the format ‘<Deep-Learning>_result_<time stamp>.json’.

9. Now push your changes to upstream of your branch ! Done !


NOTE : A config file will be generated at AIXPRT/Config/{config_name}.json after the first run of index.py . One can edit this config file to run the specific workload in a specific way .

[How to edit config](TODO : add url to  AIXPRT/EditConfig.md file )

### Steps to edit a workload

Users are allowed to edit the scripts of workload but are required to share the scripts along with the results. Below are the steps to edit a workload

1. Each workload has a script file at AIXPRT/Modules/Deep-Learning/workloads/{workloadName}/bin/{run_task.py}. This script takes care of triggering the workload script.

2. One can choose to run a custom script at this point by editing it.

3. Please make sure the workload scripts follow the guidelines as described in the "Steps to add a workload" section of this document.

## UML
![alt text](TODO: add the url to  AIXPRT/Harness/assets/HarnessUML.pdf)
