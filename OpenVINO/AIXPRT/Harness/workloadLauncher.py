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
import subprocess
import constants
import utils
from colors import *
import datetime
import time
import benchmarkVersion as bv
import shutil
import copy
import numpy as np
import gui.workload_ui as ui
import gui.workload_gui as w_gui
from threading import Thread


def checkWorkloadRunCapabilities(workload , workloadConfig):
    hardware = workloadConfig["hardware"]
    precession = workloadConfig["precision"]
    runType = workloadConfig["runtype"]
    if((hardware in workload["support"]) and (precession in workload["support"][hardware]["supported_prec"]) and (runType in workload["support"][hardware]["supported_runtype"])):
        return True
    if(("," in hardware)):
        # trying hetero mode . pass everything
        return True
    return False

def getWorkloadsToRun(config):
    selectedModule = config["module"]
    workloadsList = []
    with open(constants.BENCHMARK_DETAILS_JSON_PATH) as data_file:
        data = json.load(data_file)
        # for each module (eg Deep Learning) from the default config json
        for module in data:
            print(selectedModule)
            # If requested module and default module is same
            if(module== selectedModule):
                # For each workload in default deep learning config json
                for workload in data[module]:
                    # For each requested workload from user config
                    for workloadConfig in config["workloads_config"]:
                        # making sure that we only run the workloads in config
                        if(workload["name"] == workloadConfig["name"]):
                            # making sure that workload can run the given config . Harness will not run the workload if it doesnt have capability
                            if(checkWorkloadRunCapabilities(workload , workloadConfig)):
                                utils.colorPrint(colors.OKBLUE , ("\t%s \n"%(workload["name"])) ,colors.ENDC)
                                utils.colorPrint(colors.HEADER , ("\t\t %s \n"%workload["description"]) ,colors.ENDC)
                                workload["requested_config"]= workloadConfig
                                workloadsList.append(copy.deepcopy(workload))
    return workloadsList

def get_current_time_stamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%M')
    return st


def runConfig(config):
    workloads = getWorkloadsToRun(config)
    applicationIterations = config["iteration"]
    workloadDelays = config["delayBetweenWorkloads"]
    isDemo =  config["isDemo"]
    os.environ["DEMO"] = str(isDemo)
    i = 1
    while (i <= applicationIterations):
        threadObject = runWorkloads(config["module"],workloads,workloadDelays,isDemo)
        i = i +1
        generateResults(config["module"],workloads,config,threadObject,isDemo)
    return

#Function to run the workloads. This method runs all the workloads installed.
#Takes in a list of workloads to make sure that , we try to run the installed workloads and also ingnore any other folders which are not actual
#workloads in /workloads folder . Ex: in OpenVX workloads folder , there is a directory named "common" which is not an actula workload.
def runWorkloads(module,workloadList,workloadDelays,isDemo):

    # Delete any prior json results files saved from previous run
    for workload in workloadList:
        individualWorkloadResultFiles = os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"result")
        # delete any individual results file if exists
        if os.path.exists(individualWorkloadResultFiles):
            for jsonFile in sorted(os.listdir(individualWorkloadResultFiles)):
               if jsonFile.endswith(".json"):
                   # print("didnt delete json")
                   os.remove(os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"result",jsonFile))
    for workload in workloadList:
        utils.colorPrint(colors.HEADER,("\nRunning %s \n"%workload["name"]),colors.ENDC)
        path = os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"bin",workload["script"])
        filename, file_extension = os.path.splitext(path)
        workload_details = os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"workload_details.json")
        with open(workload_details, "w") as f:
            f.write(json.dumps(workload, indent=4, sort_keys=True))
        f.close()
        time.sleep(workloadDelays)
        if(isDemo):
            workloadDir = workload["dir_name"]
            hardware = workload["requested_config"]["hardware"]
            precision = workload["requested_config"]["precision"]
            workloadName = workload["name"]
            resultPath = os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads",workloadDir,"result",workloadName+".json")
            
            wgui = w_gui.workloadUI(workloadName,workloadDir,hardware,precision)
            
            file_id = 0
            for file in  os.listdir(os.path.join(constants.INSTALLED_MODULES_PATH,module,"packages","input_images")):
                subprocess.call(["python3",path])
                wgui.updateUI(resultPath)

                file_id+=1
            time.sleep(1)

        else:
            subprocess.call(["python3",path])
    # if(isDemo):
        # return thread

    return None


def collectResults(module,workloadList,config,isDemo):
        print("Generating Results...")
        individualWorkloadResultList=[]
        schemaResultList = []
        result_schema = ""
        frameworks = set([])
        for workload in workloadList:
            individualWorkloadResultFiles = os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"result")
            if os.path.exists(individualWorkloadResultFiles):
                resutGenereated = False
                for jsonFile in sorted(os.listdir(individualWorkloadResultFiles)):
                    if jsonFile.endswith(".json"):
                        resutGenereated = True
                        with open(os.path.join(individualWorkloadResultFiles,jsonFile)) as data_file:
                                data = json.load(data_file)
                                individualWorkloadResultList.append(data["Result"])
                if not resutGenereated:
                  utils.colorPrint(colors.FAIL,("%s did not generate a result file"%workload["name"]),colors.ENDC)
        if not (config["runtype"] == "validation"):
            result_schema = os.path.join(constants.HARNESS_PATH,"ResultSchemaFiles",module,"result_schema.json")
        else:
            result_schema = os.path.join(constants.HARNESS_PATH,"ResultSchemaFiles",module,"validation_result_schema.json")
        if os.path.exists(result_schema):
            with open(result_schema) as schema_file:
                schema = json.load(schema_file)
                for schemaWorkload in schema["Workloads"]:
                    for workload in individualWorkloadResultList:
                        throughputList = []
                        if (schemaWorkload["workloadName"] == workload["workloadName"]):
                            # Some workloads in C++ are wrting results without using resultsapi , so there are some case missmatch .
                            #This is ugly , need to fix it better
                            schemaWorkload["workloadID"] = workload["workloadID"]
                            schemaWorkload["workload run information"] = workload["workload run information"]
                            if not (workload["workload run information"].get("framework",None)==None):
                                frameworks.add(workload["workload run information"]["framework"])
                            for result in workload["results"]:
                                if (config["runtype"] == "performance"):
                                    resultAdded = False
                                    for schemaResult in schemaWorkload["results"]:
                                        if schemaResult["label"] == result["label"]:
                                            resultAdded = True
                                                #because additional info is an optional check before you try to access it
                                            if (result.get("additional info",None)==None):
                                                result["additional info"] = ""
                                            schemaResult["additional info"] = result["additional info"]
                                            if not (result.get("system_throughput",None)==None):
                                                schemaResult["system_throughput"] = result["system_throughput"]
                                                if(isDemo):
                                                    throughputList.append(result["system_throughput"])
                                                    schemaResult["system_throughput"] = sum(throughputList)/len(throughputList)
                                                schemaResult["system_throughput_units"] = result["system_throughput_units"]
                                            if not (result.get("system_latency",None)==None):
                                                schemaResult["system_latency"] = result["system_latency"]
                                                schemaResult["system_latency_units"] = result["system_latency_units"]
                                    if not resultAdded:
                                        if (result.get("system_throughput",None)==None):
                                            result["system_throughput"] = ""
                                            result["system_throughput_units"] = ""
                                        if (result.get("system_latency",None)==None):
                                            result["system_latency"] = ""
                                            result["system_latency_units"] = ""
                                        schemaWorkload["results"].append(result)
                                    
                                else:
                                    for schemaResult in schemaWorkload["results"]:
                                        if schemaResult["label"] == result["label"]:
                                            #because additional info is an optional check before you try to access it
                                            if (result.get("additional info",None)==None):
                                                result["additional info"] = ""
                                            schemaResult["additional info"] = result["additional info"]
                                            if not (result.get("accuracy",None)==None):
                                                schemaResult["accuracy"] = result["accuracy"]
                                                schemaResult["accuracy_units"] = result["accuracy_units"]

                    schemaResultList.append(schemaWorkload)
        else:
            utils.colorPrint(colors.FAIL,("Could not find result_schema.json failed to generate results"),colors.ENDC)

        if not os.path.exists(constants.OVERALL_RESULTS_DIR):
            os.makedirs(constants.OVERALL_RESULTS_DIR)
        ts= get_current_time_stamp()
        result_ts= "Result_"+module+"_"+str(ts)+".json"
        result_ts=result_ts.replace(" ", "_")
        result_ts=result_ts.replace(":", "_")
        systemInfo = {
            "Application Version":bv.getBenchmarkVersionNumber(constants.AppName),
            "Frameworks Used" : ', '.join(frameworks)
        }
        systemInfoFile = os.path.join(constants.HARNESS_PATH,"SystemInfo.json")
        with open(systemInfoFile) as f:
            info = json.load(f)
            systemInfo.update(info)
        notes = [
        "* Notes to be added"
        ]
        resultMap = {
            "Result": {
                module: {
                    "System Info":systemInfo,
                    "moduleID":utils.getModuleID(module),
                    "isDemo":isDemo,
                    "Workloads":schemaResultList,
                    "notes":notes
                }
            }

        }
        resultsDirectory = constants.OVERALL_RESULTS_DIR
        if not os.path.exists(os.path.dirname(resultsDirectory)):
            try:
                os.makedirs(os.path.dirname(resultsDirectory))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        resultConfigDirectory = os.path.join(constants.OVERALL_RESULTS_DIR,config["config_file_name"],result_ts)
        if not os.path.exists(os.path.dirname(resultConfigDirectory)):
            try:
                os.makedirs(os.path.dirname(resultConfigDirectory))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(resultConfigDirectory, "w") as f:
            f.write(json.dumps(resultMap, indent=4, sort_keys=True))
        f.close()
        # delete the generated individual result files .If a workload didnt generate result throw an error
        for workload in workloadList:
            individualWorkloadResultFiles = os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"result")
            if os.path.exists(individualWorkloadResultFiles):
                for jsonFile in sorted(os.listdir(individualWorkloadResultFiles)):
                    if jsonFile.endswith(".json"):
                        os.remove(os.path.join(constants.INSTALLED_MODULES_PATH,module,"workloads",workload["dir_name"],"result",jsonFile))
                        #print("Did not delete workload JSON")
            else:
                 utils.colorPrint(colors.FAIL,("%s did not generate a result file"%workload["name"]),colors.ENDC)

        return result_ts

def generateResults(module,workloadList,config,threadObject,isDemo):
    # if atleast 1 workload is run , then generate result
    if (len(workloadList)>0):
        resultFileName=collectResults(module,workloadList,config,isDemo)
        resultFile = os.path.join(os.environ['APP_HOME'],"Results",config["config_file_name"],resultFileName)
        # if threadObject is not None:
        #     os.environ['RUN_RESULT'] = resultFile
        #     # UI Window is still open. here is your chance to show results on UI
        #     time.sleep(1)
        #     threadObject.join()
        print("-------------------------------------------------------------------------------\n")
        utils.colorPrint(colors.OKGREEN,("Completed running %s module and results are generated at %s\n"%(module,resultFile)),colors.ENDC)
        print("--------------------------------------------------------------------------------\n")
        return True
    else:
        utils.colorPrint(colors.FAIL,("No workloads are selected to run"),colors.ENDC)
        return False
