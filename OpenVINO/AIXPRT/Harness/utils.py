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

import constants
import json
import os, platform, subprocess
import re

workloadList = []
#Function to auto complete the module name on hitting TAB key
def moduleNameCompleter(text, state):
    options = [x for x in constants.INSTALLED_MODULES_LIST if x.startswith(text)]
    try:
        return options[state]
    except IndexError:
        return None

def getInstalledWorkloadsInModule(module):
    installedWorkloads =sorted(os.listdir(constants.INSTALLED_MODULES_PATH+module+"workloads"))
    return installedWorkloads

def getModuleID(moduleName):
    with open(constants.BENCHMARK_DETAILS_JSON_PATH) as data_file:
         data = json.load(data_file)
         for module in data:
             if(module == moduleName):
                 for workload in data[module]:
                    #  First 2 digits of workloads is its module ID
                     moduleID = int((str(workload["id"]))[:2])
                     return moduleID
    return ("No Such Module "+moduleName)

def getWorkloadID(workloadName):
    with open(constants.BENCHMARK_DETAILS_JSON_PATH) as data_file:
         data = json.load(data_file)
         for module in data:
            for workload in data[module]:
                 if(workload["name"] == workloadName):
                    return workload["id"]
    return ("No Such Module "+moduleName)

def colorPrint(color,printString , end):
    if(constants.Windows):
        print(printString)
    else:
        print(color+(printString)+end)

def getCpuName():
    if platform.system() == "Windows":
        name = subprocess.check_output(["wmic","cpu","get", "name"]).strip().split("\n")[1]
        return name
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = (subprocess.check_output(command, shell=True).strip()).decode('utf_8')
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def getGpuName():
    gpuDeviceList = []
    if platform.system() == "Windows":
        return ""
    elif platform.system() == "Darwin":
        return ""
    elif platform.system() == "Linux":
        command = "lspci -nn | grep '\[03' | cut -d':' -f3-"
        all_info = (subprocess.check_output(command, shell=True).strip()).decode('utf_8')
        for line in all_info.split("\n"):
            driverID = find_between(line,"[",":")
            deviceID = find_between(line,":","]")
            gpuDeviceList.append(line)
    return gpuDeviceList

def getMachineType():
    return platform.machine()

def getOSplatform():
    return platform.platform()

def threads_per_core():
    if platform.system() == "Linux":
        command = "lscpu"
        all_info = (subprocess.check_output(command, shell=True).strip()).decode('utf_8')
        for line in all_info.split("\n"):
            if "Thread(s) per core" in line:
                return (line.rpartition(':')[2]).strip()
    return "-"


def generateSytemInfo():
    systemInfoFile = os.path.join(constants.HARNESS_PATH,"SystemInfo.json")
    GPUmap = {}
    i = 0
    for gpu in getGpuName():
        key = "GPU "+str(i)
        GPUmap[key] = gpu
        i+=1
    SystemInfo = {
        "CPU":getCpuName(),
        "Instruction Set Architecture":getMachineType(),
        "OS Platform":getOSplatform(),
        "Thread(s) per core (CPU)":threads_per_core()
        }
    SystemInfo.update(GPUmap)
    with open(systemInfoFile,"w") as f:
        f.write(json.dumps(SystemInfo, indent=4, sort_keys=True))
    f.close()
    return

def createDefaultConfig():
    configDirectory = os.path.join(constants.APP_HOME,"Config")
    print("Generating Config file")
    allDefaultConfigList = []
    modulePath = os.path.join(constants.APP_HOME,"Modules")
    moduleFolderName = ""
    for item in os.listdir(modulePath):
        moduleFolderName = item
        workloadsPath = os.path.join(modulePath,moduleFolderName,"workloads")
        for workload in os.listdir(workloadsPath):
            workload_details = os.path.join(workloadsPath,workload,"workload_details.json")
            if os.path.exists(workload_details):
                with open(workload_details) as data_file:
                     workload_details_map = json.load(data_file)
                     if(workload_details_map.get("default_run_config",None) == None):
                         print("Could not find a default_run_config for %s workload"%workload_details_map["name"])
                     else:
                         default_run_config = workload_details_map["default_run_config"]
                         if(default_run_config["runtype"].lower()=="performance"):
                             default_run_config["name"]=workload_details_map["name"]
                             allDefaultConfigList.append(default_run_config)
        configMap = {
            "runtype":"performance",
            "iteration":1,
            "delayBetweenWorkloads":0,
            "module":moduleFolderName,
            "workloads_config":allDefaultConfigList
            }
        if not os.path.isdir(configDirectory):
            try:
                os.makedirs(configDirectory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        configFileName = moduleFolderName+"_default_config.json"
        configJsonFile = os.path.join(constants.APP_HOME,"Config",configFileName)
        with open(configJsonFile, "w") as f:
            f.write(json.dumps(configMap, indent=4, sort_keys=True))
        f.close()
    return

def generateConfigJson():
    configDirectory = os.path.join(constants.APP_HOME,"Config")
    if os.path.isdir(configDirectory):
        configFileCount = 0
        for fname in os.listdir(configDirectory):
            if fname.endswith('.json'):
                configFileCount = configFileCount+1
        if(configFileCount>0):
            for fname in os.listdir(configDirectory):
                if fname.endswith('.json'):
                    print("running application with existing configurations at %s"%configDirectory)
                    return
        else:
            createDefaultConfig()
    else:
        createDefaultConfig()
    return

# Creates log directory if one is not available
def createLogDir():
    logDir = constants.LOG_DIR
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    return logDir
