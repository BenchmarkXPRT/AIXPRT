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
import os

def getBenchmarkVersionNumber(AppName):
    versionjson = os.path.join(constants.APP_HOME,'Harness','version.json')
    with open(versionjson) as version_file:
        data = json.load(version_file)
        version = 'versionNumber'
        version = data[version]
        return version
