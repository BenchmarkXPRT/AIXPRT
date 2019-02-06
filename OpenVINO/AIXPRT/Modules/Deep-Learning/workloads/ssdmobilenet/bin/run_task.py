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

model_name= "ssd_mobilenet"
dir_name= "ssdmobilenet"
dataset= "VOC2007_OD"

sys.path.insert(0, os.path.join(os.environ['APP_HOME'],"Modules","Deep-Learning","workloads","commonsources","bin"))
from run import object_detection_ssd
object_detection_ssd(model_name,dir_name,dataset)
