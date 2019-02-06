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
import inspect
import platform

#Platform on which the application is running
PLATFORM = platform.platform()
Windows = False

if "windows" in PLATFORM.lower():
    Windows = True

#path to where this file is
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Home and harness is split from path
AppHome , Harness = os.path.split(path)

APP_HOME = AppHome

AppName = os.path.basename(AppHome)

os.environ["APP_HOME"] = APP_HOME

INSTALLED_MODULES_PATH = os.path.join(APP_HOME,"Modules")
HARNESS_PATH = os.path.join(APP_HOME,"Harness")
INSTALLED_MODULES_LIST =sorted(os.listdir(INSTALLED_MODULES_PATH))
BENCHMARK_DETAILS_JSON_PATH = os.path.join(APP_HOME,'Harness','benchmark_details.json')
TOTAL_MODULES_LIST = list()
OVERALL_RESULTS_DIR = os.path.join(APP_HOME,'Results')
CALIBRATION_FILE = os.path.join(HARNESS_PATH,"calibration.json")
LOG_DIR = os.path.join(APP_HOME,"Logs")
POWER_JSON = os.path.join(HARNESS_PATH,"power.json")
