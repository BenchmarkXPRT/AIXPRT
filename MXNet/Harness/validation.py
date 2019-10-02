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

def validateWorkloadName(workloadName):
    with open(constants.BENCHMARK_DETAILS_JSON_PATH) as data_file:
        data = json.load(data_file)
        for module in data:
            for workload in data[module]:
                if(workload["name"] == workloadName):
                    return True
    return False
