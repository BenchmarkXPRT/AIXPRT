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

import argparse
import os
import readline
import sys
import json
from colors import *
import constants
import utils
import subprocess
import platform, re
import benchmarkVersion as bv
import workloadLauncher

# App name is the name of the application folder.
AppName = constants.AppName
# Application version number is fetched form version.json
versionNumber = bv.getBenchmarkVersionNumber(AppName)
#Generate system info for every run so that chnages apply if user changes sytem config
utils.generateSytemInfo()
#generates config if there isn't one in Application . Config.json is generates new everytime during application packaging.
# This method is used when tried to run when there is not config.json
utils.generateConfigJson()

print('Harness version : %s'%versionNumber)

#Main function (called at the end of this class)
def main(argv):
    buildBenchmarkDetailsJson()
    configsList = getListOfConfigFiles()
    for config in configsList:
        workloadLauncher.runConfig(config)


# Function to get the list of confugaration json files present in /Config folder
def getListOfConfigFiles():
    configDirectory = os.path.join(constants.APP_HOME,"Config")
    configFileList = []
    if os.path.isdir(configDirectory):
        for fname in os.listdir(configDirectory):
            if fname.endswith('.json'):
                with open(os.path.join(configDirectory,fname)) as data_file:
                     configData = json.load(data_file)
                     (name,ext) = os.path.splitext(fname)
                     configData["config_file_name"] = name
                configFileList.append(configData)
    return configFileList


# This map contains workload name and its id
installedMainAndSubWorkloadsMap = {}
#Function check the workload_details.json in each module and
# constructs a module_details.json to create a moduleContext.
def buildBenchmarkDetailsJson():
        data={}
        workloadsList=[]
        moduleWorkloadMap={}
        for installedModule in constants.INSTALLED_MODULES_LIST:
            #clear the list before adding workloads in a module ,
            # else it'll add the previous module's workloads to the rest of the modules.
            del workloadsList[:]
            # go to each workload and collect its information in a list
            workloadsPath = os.path.join(constants.INSTALLED_MODULES_PATH,installedModule,'workloads')
            for workloadFolder in sorted(os.listdir(workloadsPath)):
                jsonFile = os.path.join( workloadsPath,workloadFolder,'workload_details.json')
                if os.path.exists(jsonFile):
                    with open(jsonFile) as data_file:
                         installedWorkloadData = json.load(data_file)
                         workloadsList.append(installedWorkloadData)
                    data_file.close()
            moduleWorkloadMap[installedModule] = workloadsList
            with open(constants.BENCHMARK_DETAILS_JSON_PATH ,'w') as data:
                data.write(json.dumps(moduleWorkloadMap, indent=4,sort_keys=True))
            data.close()
        return

if __name__ == "__main__":
    main(sys.argv)
