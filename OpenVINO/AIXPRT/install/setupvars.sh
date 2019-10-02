#!/bin/bash

# Copyright (c) 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INSTALLDIR="${INTEL_OPENVINO_DIR:-/opt/intel/ov_custom/}"
if [[ ! -d "${INSTALLDIR}" ]]; then
  # Script has not been processed by installer, so INSTALLDIR is not valid
  # Using autodetection assuming:
  # - current shell is "bash"
  # - location of the current script is in "INSTALLDIR/bin"
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  BASE_DIR="$( dirname "$SCRIPT_DIR" )"

  INSTALLDIR="${BASE_DIR}"
fi

export INTEL_OPENVINO_DIR="$INSTALLDIR"
export INTEL_CVSDK_DIR="$INTEL_OPENVINO_DIR"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -pyver)
    python_version=$2
    echo python_version = "${python_version}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

if [ -e $INSTALLDIR/openvx ]; then
    export LD_LIBRARY_PATH=$INSTALLDIR/openvx/lib:$LD_LIBRARY_PATH
fi

if [ -e $INSTALLDIR/deployment_tools/inference_engine ]; then
    export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
    system_type=$(ls $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/)
    IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/$system_type

    if [[ -e ${IE_PLUGINS_PATH}/arch_descriptions ]]; then
        export ARCH_ROOT_DIR=${IE_PLUGINS_PATH}/arch_descriptions
    fi

    export HDDL_INSTALL_DIR=$INSTALLDIR/deployment_tools/inference_engine/external/hddl
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_mac/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:$IE_PLUGINS_PATH:$DYLD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_mac/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
    else
        export LD_LIBRARY_PATH=/opt/intel/opencl:$HDDL_INSTALL_DIR/lib:$INSTALLDIR/deployment_tools/inference_engine/external/gna/lib:$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:$INSTALLDIR/deployment_tools/inference_engine/external/omp/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
    fi
fi

if [ -e "$INSTALLDIR/opencv" ]; then
    if [ -f "$INSTALLDIR/opencv/setupvars.sh" ]; then
        source "$INSTALLDIR/opencv/setupvars.sh"
    else
        export OpenCV_DIR="$INSTALLDIR/opencv/share/OpenCV"
        export LD_LIBRARY_PATH="$INSTALLDIR/opencv/lib:$LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="$INSTALLDIR/opencv/share/OpenCV/3rdparty/lib:$LD_LIBRARY_PATH"
    fi
fi

export PATH="$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer:$PATH"
export PYTHONPATH="$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer:$PYTHONPATH"

if [ -z "$python_version" ]; then
    if command -v python3.7 >/dev/null 2>&1; then
        python_version=3.7
        python_bitness=$(python3.7 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python3.6 >/dev/null 2>&1; then
        python_version=3.6
        python_bitness=$(python3.6 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python3.5 >/dev/null 2>&1; then
        python_version=3.5
        python_bitness=$(python3.5 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python3.4 >/dev/null 2>&1; then
        python_version=3.4
        python_bitness=$(python3.4 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    elif command -v python2.7 >/dev/null 2>&1; then      
        python_version=2.7
    elif command -v python >/dev/null 2>&1; then
        python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    fi
fi

OS_NAME=""
if [[ -f /etc/os-release ]]; then
    OS_NAME=$(lsb_release -i -s)
fi

if [ "$python_bitness" != "" ] && [ "$python_bitness" != "64" ] && [ "$OS_NAME" != "Raspbian" ]; then
    echo "[setupvars.sh] 64 bitness for Python" $python_version "is requred"
fi

if [ ! -z "$python_version" ]; then
    if [ "$python_version" != "2.7" ]; then
        # add path to OpenCV API for Python 3.x
        export PYTHONPATH="$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH"
    fi
    # add path to Inference Engine Python API
    export PYTHONPATH="$INTEL_OPENVINO_DIR/python/python$python_version:$PYTHONPATH"    
fi

echo "[setupvars.sh]  environment initialized"
