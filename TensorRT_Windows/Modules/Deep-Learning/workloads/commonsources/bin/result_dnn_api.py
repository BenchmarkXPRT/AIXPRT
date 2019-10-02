#
# Author: Ramesh Chukka; Created on Feb 22, 2017
#
# Functions for workloads to write the results using result API from Harness
# Note: This module uses resultsapi package from Harness
#

import sys
import os
import platform
import csv
import json
import subprocess
import re


###################################################################
## Some utility functions for writing the results as expected by the common API
###################################################################
## returnInputMap()
def returnInputMap(framework, imgSource, imgCount, arch, model, version, precision, iterations):
    workloadInput = {}
    cudaVersion = getCudaVersion()
    cudnnVersion = "CUDNN " + getCudnnVersion()
    workloadInput["Framework"] =  framework+" "+str(version)
    workloadInput["Framework version"] =  version
    workloadInput["inference_runtime"] =  framework+" "+str(version)
    workloadInput["accelerator_lib"] = [cudaVersion, cudnnVersion]
    workloadInput["Input Images Source"] = imgSource
    workloadInput["Images Count"] =  imgCount
    workloadInput["architecture"] =  arch
    workloadInput["model"] =  model
    workloadInput["precision"] =  precision
    workloadInput["total_requests"] =  iterations

    return(workloadInput)

## returnBatchsizeResults()
def returnBatchsizeResults(batchSize, ker_throughput, ker_latency, iterCount, misc,  throughput_units, latency_units, sys_throughput, system_latency,
pct50, pct90, pct95, pct99, max_time, min_time, concurrent):
    results = []
    insideResults = {}
    insideResults["label"] = 'Batch ' + str(batchSize)
    insideResults["system_throughput"] = sys_throughput
    insideResults["system_throughput_units"] = throughput_units
    # Optional Fields
    additional_info = []
    insideResults["additional info"] = additional_info
    additional_info_details = {}
    additional_info_details["batch"] = batchSize
    additional_info_details["total_requests"] = iterCount
    additional_info_details["concurrent_instances"] = concurrent
    additional_info_details["time_units"] = "milliseconds"
    additional_info_details["Extra"] = misc
    additional_info_details["50_percentile_time"] = pct50
    additional_info_details["90_percentile_time"] = pct90
    additional_info_details["95_percentile_time"] = pct95
    additional_info_details["99_percentile_time"] = pct99
    additional_info.append(additional_info_details)
    results.append(insideResults)

    return(results)

## returnAccuracyResults()
def returnAccuracyResults(top, percentValue, std, iterCount, misc, prec):
    results = []
    insideResults = {}
    insideResults["label"] = 'Top-' + str(top)
    insideResults["accuracy"] =   percentValue
    insideResults["accuracy_units"] = 'percentage'
    # Optional Fields
    additional_info = []
    insideResults["additional info"] = additional_info
    additional_info_details = {}
    additional_info_details["Type"] = top
    additional_info_details["Accuracy"] = percentValue
    additional_info_details["ImageCount"] = iterCount
    additional_info_details["std"] = std
    additional_info_details["prec"] = prec
    additional_info_details["Extra"] = misc
    additional_info.append(additional_info_details)
    results.append(insideResults)

    return(results)

## returnWorkloadName()
def returnWorkloadName(key):
    dict = {}
    dict["resnet-50"] = "ResNet-50"
    dict["ssd-mobilenet_v1"] = "SSD-MobileNet-v1"
    return dict.get(key)


# Get the version of cuda available on system
def getCudaVersion():
    if platform.system() == "Linux":
        command = "cat /usr/local/cuda/version.txt"
        cudaVersion = subprocess.check_output(command, shell=True).strip().decode('utf_8')
        return cudaVersion
    elif platform.system() == "Windows":
        command = "nvcc --version"
        all_info = subprocess.check_output(command, shell=True).strip().decode('utf_8')
        if all_info:
            for line in all_info.split("\n"):
                if "Cuda compilation tools" in line:
                    return line
    return (" ")

#Get the version of cudnn installed if available
def getCudnnVersion():

    if platform.system() == "Linux":
        #Paths where cudnn can be installed
        cudnnPath1 = "/usr/local/cuda/include/cudnn.h"
        cudnnPath2 = "/usr/include/cudnn.h"

        all_info = ""
        CUDNN_MAJOR = ""
        CUDNN_MINOR = ""
        CUDNN_PATCHLEVEL = ""
        if(os.path.exists(cudnnPath1)):
            command = "cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2"
            all_info = subprocess.check_output(command, shell=True).strip().decode('utf_8')
        if(os.path.exists(cudnnPath2)):
            command = "cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2"
            all_info = subprocess.check_output(command, shell=True).strip().decode('utf_8')

        if all_info:
            for line in all_info.split("\n"):
                if "CUDNN_MAJOR" in line:
                    CUDNN_MAJOR =  re.sub( ".*CUDNN_MAJOR.* ", "", line,1)
                    break
            for line in all_info.split("\n"):
                if "CUDNN_MINOR" in line:
                    CUDNN_MINOR =  re.sub( ".*CUDNN_MINOR.* ", "", line,1)
                    break
            for line in all_info.split("\n"):
                if "CUDNN_PATCHLEVEL" in line:
                    CUDNN_PATCHLEVEL =  re.sub( ".*CUDNN_PATCHLEVEL.* ", "", line,1)
                    break
            CudnnVersion = CUDNN_MAJOR+"."+CUDNN_MINOR+"."+CUDNN_PATCHLEVEL
            return CudnnVersion
            
    elif platform.system() == "Windows":
        return " "
    



## writeResultsToAPI(name, input, results)
def writeResultsToAPI(name, input, results):
    #print "Inside writeResultsToAPI of dnn_pi"
    resultsapi.createResultJson(name, input, results)

## Here is a sample function to show how the above API can be used in a workload
## This example creates a result entry for nvCaffe workload for a given Topology
## writeBatchResultsToAPI()
def writeBatchResultsToAPI():
    # Add the result API code here
    resInImgPerSec = imgPerSec
    forwardTime = timeInMillisec
    workLoadName = returnWorkloadName(modelPrefix)
    inputString = returnInputMap("TensorRT", "ILSVRC 2012", "Batch size", "", modelPrefix+'Net', 'v4')
    resultsString = returnBatchsizeResults(batchSize, resInImgPerSec, units, forwardTime, iterCount)
    # Now wrtie the info to the API
    writeResultsToAPI(workLoadName, inputString, resultsString)

## writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext)
def writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext):
    # Add the result API code here
    resInImgPerSec = topResult
    forwardTime = Std
    composedWorkloadName = model+'_'+ext+'_validation'
    workLoadName = returnWorkloadName(composedWorkloadName)
    inputString = returnInputMap("TensorRT", "ILSVRC 2012", "Accuracy", "gpu", model+'net','v4')
    resultsString = returnAccuracyResults(accType, topResult, Std, iterCount, model+'net')
    # Now wrtie the info to the API
    writeResultsToAPI(workLoadName, inputString, resultsString)

path = os.environ["APP_HOME"]+"/Harness/"
sys.path.insert(0, path)
import resultsapi	#For MLXBench results API
