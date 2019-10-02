#!/usr/bin/python

# ============================================================================
# Copyright (C) 2018 BenchmarkXPRT Development Community
# Licensed under the BENCHMARKXPRT DEVELOPMENT COMMUNITY MEMBER LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License by contacting Principled Technologies, Inc.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing grants and
# restrictions under the License.
#============================================================================

import os
import json
import constants
import utils
import validation

def writeResult(workloadID,resultMap,workloadName):

    workloadDirName = ""
    moduleName = ""
    newResult={}
    with open(constants.BENCHMARK_DETAILS_JSON_PATH) as data_file:
         data = json.load(data_file)
         for module in data:
             for workload in data[module]:
                 if workload["name"]==workloadName:
                     moduleName = module
                     workloadDirName = workload["dir_name"]
                     if not (workload.get("workload_info",None)==None):
                        #  Add default workload info to the generated info so that generated info is given more importance
                         workload["workload_info"].update(resultMap["Result"]["workload run information"])
                         resultMap["Result"]["workload run information"].update(workload["workload_info"])
                     break
    data_file.close()
    resultFileName =(workloadName.replace(" ", "_"))+".json"
    resultDir = os.path.join(constants.INSTALLED_MODULES_PATH,moduleName,"workloads",workloadDirName,"result")
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
    resultFile = os.path.join(resultDir,resultFileName)
    if not os.path.isfile(resultFile):
        try:
            with open(resultFile, "w") as f:
                f.write(json.dumps(resultMap,indent=4, sort_keys=True))
            f.close()
            return True
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                return False
    else:
        try:
            with open(resultFile) as data_file:
                  newResult = json.load(data_file)
                  #check if there is already some result for this workload is written . If there is some result ,
                  # then append the new result to the existing file.
                  # Ex: If a file for alexnet already exist with batch size 2 and a new result comes in with
                  # batch size 8 , then append this result to existing folder instead of creating a new folder.
                  if(newResult["Result"]["workloadID"]==(resultMap.get("Result",None)).get("workloadID")):
                      for result in resultMap.get("Result",None).get("results"):
                            (newResult["Result"]["results"]).append(result)
            data_file.close()
        except OSError as exc: # Guard against race condition
            print("Exception while reading existing result file")
            if exc.errno != errno.EEXIST:
                raise
                return False
        try:
            with open (resultFile,"w") as f:
                f.write(json.dumps(newResult,indent=4, sort_keys=True))
            f.close()
            return True
        except OSError as exc: # Guard against race condition
            print("Exception while trying to write to existing result file")
            if exc.errno != errno.EEXIST:
                raise
                return False
    return False

def hasMeasurments(result):
    hasOneMeasurments = False
    if not (result.get("system_throughput",None)==None):
        if(result.get("system_throughput_units",None)==None):
            print("system_throughput units should be system_throughput_units")
            return False
        hasOneMeasurments = True
    if not (result.get("system_latency",None)==None):
        if(result.get("system_latency_units",None)==None):
            print("system_latency units should be system_latency_units")
            return False
        hasOneMeasurments = True
    if not (result.get("accuracy",None)==None):
        if(result.get("accuracy_units",None)==None):
            print("accuracy units should be accuracy_units")
            return False
        hasOneMeasurments = True
    return hasOneMeasurments


def createResultJson(workloadName,workloadInput, results):
        """API to create a json file with the given input parameters.
        Also collect information form workload_details.json with the provided workloadName
        and adds this information to results json file.

        Args:
            workloadName: Name of the workload as described in the workload_details.json
            workloadInput: Dictionary of Inputs used to run the workload. "architechture" is a required feild and
                           the rest of the feilds are options and change with each workload.
                Ex :
                    workloadInput={
                        "Tensorflow_version": tf.__version__,
                        "Input Images Source": "ILSVRC 2012",
                        "Images Count":"Batch size",
                        "architecture":"gpu"
                        }
            results : Dictionary of the result as showen in the example below.
                        all feilds are required. "additional info" array cal be left empty if there isnt any additional info that needs to be saved.
                        NOTE : value of "measurment" needs to be a float value at all time
                        Ex:
                        results=[
                            {
                                "label":"Batch 2",
                                "measurement":216.42,
                                "units":"milliseconds",
                                "additional info":[]
                            }
        Returns:
            True once the json is written , False if the API fails to do so.

        Errors:
            Prints out error mesages with the failure reason

            TODO : Improve the code by throwing expections for failures instead of printing error messages
        """

        if not (validation.validateWorkloadName(workloadName)):
            print(workloadName)
            # raise ValueError('\nworkloadName is invalid\n')
            print("\nworkloadName is invalid\n")
            return
        if (workloadInput.get("architecture",None)==None):
            # raise ValueError('\nPlease provide the architecture on which the workload ran (cpu/gpu?)\n'
            print("\nPlease provide the architecture on which the workload ran (cpu/gpu?)\n")
            return
        if (workloadInput.get("precision",None)==None):
            # raise ValueError('\nPlease provide the precision on which the workload ran , if your workload has independent of precision then add NA \n'
            print("\nPlease provide the precision on which the workload ran , if your workload has independent of precision then add NA \n")
            return
        for result in results:
            if (result.get("label",None)==None):
                print("Each result item must have a tag label")
                return False
            if not (hasMeasurments(result)):
                print("Each result item must have a measurment associated with it with one of the below tags \n")
                print("system_throughput and system_throughput_units")
                print("system_latency and system_latency_units")
                return False
        workloadID=utils.getWorkloadID(workloadName)
        resultMap={
                    "Result":{
                             "workloadName":workloadName,
                             "workloadID":workloadID,
                             "workload run information":workloadInput,
                             "results":results
                            }
                    }
        return(writeResult(workloadID,resultMap,workloadName))
