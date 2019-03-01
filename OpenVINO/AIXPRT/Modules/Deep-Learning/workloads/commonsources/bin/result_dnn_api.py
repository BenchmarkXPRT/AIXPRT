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

import sys
import os
import platform
import csv
import json



###################################################################
## Some utility functions for writing the results as expected by the common API
###################################################################
## returnInputMap()
def returnInputMap(framework, version, imgSource, imgCount, arch, model , precision , iterations ,acceleratorLib=[]):
    # For now it is hard coded
    workloadInput = {}
    workloadInput["framework"] =  framework+" "+version
    workloadInput["input images source"] = imgSource
    workloadInput["images count"] =  imgCount
    workloadInput["precision"] =  precision
    workloadInput["iterations"] =  iterations
    workloadInput["architecture"] =  arch
    workloadInput["model"] =  model
    workloadInput["accelerator_lib"]= acceleratorLib
    return(workloadInput)

## returnBatchsizeResults()
def returnBatchsizeResults(batchSize, resultInImgPerSec, timeInMsec, iterCount, modelName, aarch, standardDev, concurrent_instances, perc_99, perc_95, perc_90, perc_50, min_time, max_time):
    results = []
    insideResults = {}
    insideResults["label"] = 'Batch ' + str(batchSize)
    #insideResults["Batch"] = str(batchSize)
    insideResults["system_throughput"] =   resultInImgPerSec
    insideResults["system_throughput_units"] =   'imgs/sec'

    # Optional Fields
    additional_info = []
    insideResults["additional info"] = additional_info
    additional_info_details = {"concurrent_instances": concurrent_instances, "total_requests": iterCount, "50_percentile_time": perc_50, "90_percentile_time": perc_90,
                               "95_percentile_time": perc_95, "99_percentile_time": perc_99,"time_units":"milliseconds"}


    additional_info.append(additional_info_details)
    results.append(insideResults)

    return(results)

def returnAccuracyResults(top, percentValue, std, iterCount, modelName, prec, arch):
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
    additional_info_details["iter_count"] = iterCount
    additional_info_details["std"] = std
    additional_info_details["prec"] = prec
    additional_info_details["modelName"] = modelName
    additional_info.append(additional_info_details)
    results.append(insideResults)

    return(results)

## returnWorkloadName()
def returnWorkloadName(key):
    dict = {}
    dict["resnet-50"] = "ResNet-50"
    dict["ssd_mobilenet"] = "SSD-MobileNet-v1"

    return dict.get(key)

## writeResultsToAPI(name, input, results)
def writeResultsToAPI(name, input, results):
    #print "Inside writeResultsToAPI of dnn_pi"
    resultsapi.createResultJson(name, input, results)

## writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext)
def writeAccuracyResultsToAPI(accType, topResult, Std, iterCount, model, ext):
    # Add the result API code here
    resInImgPerSec = topResult
    forwardTime = Std
    composedWorkloadName = model+'_'+ext+'_validation'
    workLoadName = returnWorkloadName(composedWorkloadName)
    inputString = returnInputMap("OpenVINO", "ILSVRC 2012", "Accuracy", "gpu", model+'net', '')
    resultsString = returnAccuracyResults(accType, topResult, Std, iterCount, model+'net')
    # Now write the info to the API
    #print inputString
    #print resultsString
    writeResultsToAPI(workLoadName, inputString, resultsString)

sys.path.insert(1, os.path.join(os.environ['APP_HOME'],'Harness'))
import resultsapi	#For MLXBench results API
