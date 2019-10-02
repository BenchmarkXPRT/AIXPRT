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
import sys
sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Harness"))
import utils
import constants
import json

def workloadStarted(workloadName , workloadID , batchSize , precision , aarch,command):
    """Call back Function for workload start event.
    All the workloads in the application calls this function right before
    begining the performance timer
    Args:
        workloadName: Name of the workload as described in the workload_details.json of each workload.
        workloadID: ID of the workload as described in the workload_details.json of each workload.
        batchSize: Integer batch size number used to run the workload. Ex: 1 , 2 ,4 ,8 etc . If a workload doesnt have BatchSize , then pass in None
        precision : precision on which the workload ran . Ex fp32 . fp16 , int8
        aarch : Hardware architechture on which the workload is running. Ex : cpu , gpu , myriad , fpge
        command : final command to run the actual execution of the ML_worklaod
    Returns:
        updated command WRT to trace tool format
        command format : <trace tool command> + <actual cmd from APP>
    """

    return command

def workloadEnded(workloadName , workloadID , batchSize , precision , aarch):
        """Call back Function for workload end event.
        All the workloads in the application calls this function right after
        stopping the performance timer
        Args:
            workloadName: Name of the workload as described in the workload_details.json of each workload.
            workloadID: ID of the workload as described in the workload_details.json of each workload.
            batchSize: Integer batch size number used to run the workload. Ex: 1 , 2 ,4 ,8 etc . If a workload doesnt have BatchSize , then pass in None
            precision : precision on which the workload ran . Ex fp32 . fp16 , int8
            aarch : Hardware architechture on which the workload is running. Ex : cpu , gpu , myriad , fpge
        Returns:
            A boolena value true to denote that the workload started and some tool is using the call back
            Returns False when no tool is using the callback
        """
        return False

def workloadPaused(workloadName , workloadID , batchSize , precision , aarch):
    """Call back Function for workload pause event.
    All the workloads in the application calls this function right after
    stopping the performance timer
    Args:
        workloadName: Name of the workload as described in the workload_details.json of each workload.
        workloadID: ID of the workload as described in the workload_details.json of each workload.
        batchSize: Integer batch size number used to run the workload. Ex: 1 , 2 ,4 ,8 etc . If a workload doesnt have BatchSize , then pass in None
        precision : precision on which the workload ran . Ex fp32 . fp16 , int8
        aarch : Hardware architechture on which the workload is running. Ex : cpu , gpu , myriad , fpge
    Returns:
        A boolena value true to denote that the workload started and some tool is using the call back
        Returns False when no tool is using the callback
    """
    return False
