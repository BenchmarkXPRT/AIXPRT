#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

. /etc/os-release
DISTRO=$NAME

if [[ $DISTRO = "Clear Linux OS" ]]; then
    USE_PREBUILT_OPENVINO=TRUE
fi

NUM_ARGS=$#
usage() {
    echo "Usage:"
    echo -e "\t compile_AIXPRT_sources.sh </path/to/AIXPRT>"
    echo -e "\t compile_AIXPRT_sources.sh -dir </path/to/AIXPRT>"
    echo -e "\t compile_AIXPRT_sources.sh -h [PRINT HELP MESSAGE]"
    echo -e "Assumes:\n\t--- you have installed openVINO in /opt/intel/computer_vision_sdk/ or /opt/intel/openvino/ or a custom build path to /dldt folder"
    echo -e "\t--- you have cloned AIXPRT"
    exit 1
}

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

#============================= parse command line options and set AIXPRT install directory ===============================================
key="$1"
case $key in
    -h | -help | --help)
     usage
    ;;
    -dir | --dir)
    flag=$key
    AIXPRT_DIR="$2"
    ;;
esac


if [[ ${NUM_ARGS} -lt 1 ]]; then
  AIXPRT_DIR="$( dirname $PWD )"
  if [ -d ${AIXPRT_DIR} ] && [ -d "${AIXPRT_DIR}/Modules" ]; then
     echo -e "\e[1;33mAIXPRT install directory not provided. Installing to ${AIXPRT_DIR}\e[0m"
  else
     echo -e "\e[1;31mCannot determine AIXPRT distribution directory. Please pass the source directory with -dir </path/to/AIXPRT/folder>.\e[0m"
     usage
  fi

  if [[ -z $USE_PREBUILT_OPENVINO ]]; then
     OPENVINO_DIR="/opt/intel/computer_vision_sdk"
     if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
        echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
     else
        OPENVINO_DIR="/opt/intel/openvino"
        if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
           echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
        else
           echo -e "\e[1;31mCannot find OpenVINO in default install locations /opt/intel/openvino or /opt/intel/computer_vision_sdk\e[0m"
           usage
        fi
     fi
  fi

fi



if [[ ${NUM_ARGS} == 1 ]]; then
   AIXPRT_DIR=$key
  if  [ ! -d ${AIXPRT_DIR} ]; then
     echo -e "\e[1;31m\n\nTarget folder ${key} does not exist.\n\e[0m"
     usage
  fi

  if [[ -z $USE_PREBUILT_OPENVINO ]]; then
     OPENVINO_DIR="/opt/intel/computer_vision_sdk"
     if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
        echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
     else
        OPENVINO_DIR="/opt/intel/openvino"
        if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
           echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
        else
           echo -e "\e[1;31mCannot find OpenVINO in default install locations /opt/intel/openvino or /opt/intel/computer_vision_sdk\e[0m"
           usage
        fi
     fi
  fi

elif [[ ${NUM_ARGS} == 2 ]]; then
  AIXPRT_DIR="$1"
  if  [ ! -d ${AIXPRT_DIR} ]; then
     echo -e "\e[1;31m\n\nTarget folder ${key} does not exist.\n\e[0m"
     usage
  fi

  if [[ -z $USE_PREBUILT_OPENVINO ]]; then
     OPENVINO_DIR="$2"
     #If custom build is provides , first restucture the custom build at /opt/int/ov_custom/ so that the rest of the script works.
     if [[ $OPENVINO_DIR == *"dldt"* ]]; then
      source ${PWD}/restructure-installation.sh $OPENVINO_DIR $AIXPRT_DIR
      OPENVINO_DIR=/opt/intel/ov_custom
      echo -e "\e[1;32mUsing dldt source to build AIXPRT. Final dldt installation will be at /opt/intel/ov_custom\e[0m"
     fi
     if [ ! -d ${OPENVINO_DIR} ]; then
        echo -e "\e[1;31m\n\nOpenvino install directory ${OPENVINO_DIR} not found.\n\e[0m"
        usage
     fi
  fi

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "$flag" == "-dir" ] && [ ! -d ${AIXPRT_DIR} ]; then

   echo -e "\e[1;31m\n\nTarget folder ${AIXPRT_DIR} does not exist.\n\e[0m"
   usage

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "${1}" != "-dir" ]; then
   AIXPRT_DIR=${1}

   echo -e "\e[1;31mProvided install directory ${AIXPRT_DIR} does not exist.\n\e[0m"
   usage

elif [[ ${NUM_ARGS} -gt 2 ]]; then
   echo -e "\e[1;31m\nCannot parse inputs\n\e[0m"
   usage
fi

# Last sanity check
if [ ! -d "${AIXPRT_DIR}/Modules" ]; then
   echo -e "\e[1;31mProvided install directory ${AIXPRT_DIR} does not have Modules subdir.\e[0m"
   usage
fi
if [[ -z $USE_PREBUILT_OPENVINO ]]; then
   if [ ! -d ${OPENVINO_DIR}/bin ]; then
      echo -e "\e[1;31m\n\nThe provided OpenVINO install directory must contain a 'bin' folder. Please check that you have installed openvino correctly.\n\e[0m"
      usage
  fi
fi


RUN_AGAIN="Then run the script again\n\n"
DASHES="\n\n==================================================\n\n"
CUR_PATH=$PWD

if [[ -z $USE_PREBUILT_OPENVINO ]]; then
   if [ -d "${OPENVINO_DIR}/deployment_tools/tools/" ]; then
      OPENVINO_BUILD="R1 and above"
   else
      OPENVINO_BUILD="R5"
   fi
 
   if [ ! -e $OPENVINO_DIR ]; then
      echo -e "\e[1;33m\nDid not find OpenVINO installed in ${OPENVINO_DIR}.\n\e[0m"
      echo -e "\e[1;0mPlease install OpenVINO distribution in /opt/intel\n\e[0m"
 
      exit 1
   fi

   OPENVINO_CV_DEP_DIR="${OPENVINO_DIR}/install_dependencies"
fi

AIXPRT_MODELS="${AIXPRT_DIR}/Modules/Deep-Learning/packages/models"
AIXPRT_PLUGIN="${AIXPRT_DIR}/Modules/Deep-Learning/packages/plugin/"
AIXPRT_BIN="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/"
AIXPRT_SOURCES="${AIXPRT_DIR}/Modules/Deep-Learning/workloads/commonsources/bin/src/"

#========================= Install dependencies =======================================================

# Step 1. Install Dependencies

printf "${DASHES}"
printf "Installing dependencies"
printf "${DASHES}"
#------------------------------------------------------------------------------------------------------
if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
    IFS='=' read -ra arr <<< "$(cat /etc/lsb-release | grep DISTRIB_RELEASE)" # get the release version
    RELEASE=${arr[1]}
fi

if [[ $DISTRO == "centos" ]]; then
    if command -v python3.5 >/dev/null 2>&1; then
        PYTHON_BINARY=python3.5
    fi
    if command -v python3.6 >/dev/null 2>&1; then
        PYTHON_BINARY=python3.6
    fi
    if [ -z "$PYTHON_BINARY" ]; then
        sudo -E yum install -y https://centos7.iuscommunity.org/ius-release.rpm
        #sudo -E yum install -y python36u easy_install python36u-pip
        sudo -E yum install -y python36u python36u-pip libgfortran3 build-essential libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base libpng12-dev python-pil
        sudo -E pip3.6 install virtualenv
        PYTHON_BINARY=python3.6
    fi
elif [[ $DISTRO == "ubuntu" ]]; then
    sudo -E apt -y install python3-pip libgfortran3 build-essential libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base
    PYTHON_BINARY=python3

    if [[ $RELEASE == "16.04" ]]; then
       echo -e "\e[0;32m Installing PIL and png packages for Ubuntu 16.04.\e[0m"
       sudo -E apt -y install libpng12-dev python-imaging

    else
       echo -e "\e[0;32m Installing PIL and png packages for Ubuntu 18.04.\e[0m"
       sudo -E apt -y install python-pil libpng-dev
    fi
elif [[ $DISTRO == "Clear Linux OS" ]]; then
    echo -e "\e[0;31m Clear Linux operating system.\e[0m"
else
   echo -e "\e[0;31mUnsupported operating system.\e[0m"
   exit
fi

#------------------------------------------------------------------------------------------------------

if [[ -z $USE_PREBUILT_OPENVINO ]]; then
   printf "${DASHES}"
   printf "Installing cv sdk dependencies\n\n"
   # cd ${OPENVINO_CV_DEP_DIR}

   # if [ "${OPENVINO_BUILD}" == "R5" ]; then
   #    sudo -E ./install_cv_sdk_dependencies.sh;
   # else
   #    sudo -E bash install_openvino_dependencies.sh
   # fi

   cd ${CUR_PATH}

   #========================= Setup Model Optimizer =======================================================
   # Step 2. Enter OpenVINO environment and Configure Model Optimizer

   printf "${DASHES}"
   printf "Setting OpenVINO environment and Configuring Model Optimizer"
   printf "${DASHES}"

   if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
      printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setvars.sh to set it. \n"

      if [ -e "${OPENVINO_DIR}/bin/setupvars.sh" ]; then # for Intel CV SDK package
         SETVARS_PATH="${OPENVINO_DIR}/bin/setupvars.sh"
      else
         echo -e "\e[0;31mError: setvars.sh is not found\n\e[0m"
         exit 1
      fi
      if ! source ${SETVARS_PATH} ; then
         printf "Unable to run ./setvars.sh. Please check its presence. ${RUN_AGAIN}"
         exit 1
      fi
   fi

   OPENVINO_DT_DIR="${OPENVINO_DIR}/deployment_tools"
   OPENVINO_IE_DIR="${OPENVINO_DT_DIR}/inference_engine/"
   OPENVINO_MO_DIR="${OPENVINO_DT_DIR}/model_optimizer/"
   MO_PATH="${OPENVINO_MO_DIR}/mo.py"

   if [ "${OPENVINO_BUILD}" == "R1 and above" ]; then
      MD_PATH="${OPENVINO_DT_DIR}/tools/model_downloader/downloader.py"
   else
      MD_PATH="${OPENVINO_DT_DIR}/model_downloader/downloader.py"
   fi

   printf "${DASHES}"
   printf "Install Model Optimizer dependencies"
   cd "${OPENVINO_MO_DIR}/install_prerequisites"
# bash install_prerequisites.sh "caffe"
else
   if [[ -d clearlinux ]]; then
      rm -rf clearlinux
   fi
   wget https://github.com/opencv/open_model_zoo/archive/2019_R2.zip --directory-prefix clearlinux
   unzip clearlinux/2019_R2.zip -d clearlinux
   rm clearlinux/2019_R2.zip

   MD_PATH=clearlinux/open_model_zoo-2019_R2/tools/downloader/downloader.py
   MO_PATH=/usr/share/openvino/model-optimizer/mo.py

   export PYTHONPATH=/usr/lib/python3.7/site-packages/openvino/inference_engine/:${PYTHONPATH}
fi

cd ${CUR_PATH}
#========================= Download and Convert Caffepre-trained models ===========================================
# Step 3a. Download and Convert Caffe pretrained Models

printf "${DASHES}"
printf "Downloading and Converting pretrained models"
printf "${DASHES}"

MODEL_NAMES=("resnet-50" "mobilenet-ssd")

PRECISION_LIST=("FP16" "FP32")
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

for idx in "${!MODEL_NAMES[@]}"
  do
    MODEL_DIR="$CUR_PATH/caffe_models"

    MODEL_NAME=${MODEL_NAMES[idx]}
    echo -e "\n\e[0;32m Generating IRs for ${MODEL_NAME}\n\e[0m"
    IR_PATH="$CUR_PATH/${MODEL_NAME}"
    if [ -d ${IR_PATH} ];then rm -r ${IR_PATH}; fi;

	if [ $MODEL_NAME == "mobilenet-ssd" ]
	     then
		printf "Run ${MD_PATH} --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"\n\n"
		MODEL_PATH="$MODEL_DIR/object_detection/common/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel"
		$PYTHON_BINARY ${MD_PATH} --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"
		MEAN_VALUES="data[127.5,127.5,127.5]"
		SCALE_VALUES="data[127.50223128904757]"
		MODEL_DEST="ssd_mobilenet"
		INPUT_SHAPE="[1,3,300,300]"

	elif [ $MODEL_NAME == "resnet-50" ]
	   then
		printf "Run ${MD_PATH} --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"\n\n"
		MODEL_PATH="$MODEL_DIR/classification/resnet/v1/50/caffe/resnet-50.caffemodel"
		$PYTHON_BINARY ${MD_PATH} --name "${MODEL_NAME}" --output_dir "${MODEL_DIR}"
		MEAN_VALUES="data[104.0,117.0,123.0]"
		SCALE_VALUES="data[1.0]"
		MODEL_DEST="resnet-50"
		INPUT_SHAPE="[1,3,224,224]"
	fi

	for PRECISION in ${PRECISION_LIST[@]}
	  do
		precision=${PRECISION,,}
		IR_MODEL_XML=${IR_PATH}/${MODEL_DEST}".xml"
		IR_MODEL_BIN=${IR_PATH}/${MODEL_DEST}".bin"
		IR_MODEL_AIXPRT_XML=${IR_PATH}/${MODEL_DEST}"_${precision}.xml"
		IR_MODEL_AIXPRT_BIN=${IR_PATH}/${MODEL_DEST}"_${precision}.bin"


		printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --model_name $MODEL_DEST --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES\n\n"
		$PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --model_name $MODEL_DEST --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES

		#------- rename generated IRs
		mv ${IR_MODEL_XML} ${IR_MODEL_AIXPRT_XML}
		mv ${IR_MODEL_BIN} ${IR_MODEL_AIXPRT_BIN}
	done


	if [ ! -e $IR_PATH ]; then
	    printf "\n\nTarget folder ${IR_PATH} does not exists"
	    exit 1
	else
	    cp -r ${IR_PATH}/* ${AIXPRT_MODELS}/${MODEL_DEST}/
	    rm -r ${IR_PATH}
	fi

done


#========================= Build Classification and Detection binaries ================================================================================
# Step 4. Build samples
printf "${DASHES}"
printf "Building AIXPRT sources ${AIXPRT_SOURCES}"
printf "${DASHES}"

if ! command -v cmake &>/dev/null; then
    echo -e "\e[0;32m\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it and run again.\n\e[0m"
    exit 1
fi

# copy sources here
AIXPRT_TMP_SRC="$CUR_PATH/AIXPRT_sources/"
BUILD_DIR="$CUR_PATH/AIXPRT_compiled"
COMPILED_APP_DIR="${BUILD_DIR}/intel64/Release"
if [ -d "${AIXPRT_TMP_SRC}" ]; then rm -Rf $AIXPRT_TMP_SRC; fi
if [ -d "${BUILD_DIR}" ]; then rm -Rf $BUILD_DIR; fi

mkdir ${AIXPRT_TMP_SRC}
cp -r ${AIXPRT_SOURCES}/* ${AIXPRT_TMP_SRC}

if [[ -z $USE_PREBUILT_OPENVINO ]]; then
  cp -r ${OPENVINO_DIR}/inference_engine/samples/thirdparty ${AIXPRT_TMP_SRC}
  cp -r ${OPENVINO_DIR}/inference_engine/samples/common ${AIXPRT_TMP_SRC}
else
  CLEARLINUX_SAMPLE_HDR_DIR=/usr/share/doc/inference_engine
fi

if [ ! -e "${BUILD_DIR}/intel64/Release/benchmark_app" ]; then
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    cmake -DCMAKE_BUILD_TYPE=Release ${AIXPRT_TMP_SRC} -DUSE_PREBUILT_OPENVINO=${USE_PREBUILT_OPENVINO} -DSAMPLE_HEADER_DIR=${CLEARLINUX_SAMPLE_HDR_DIR}
    make -j8
else
    printf "\n\nTarget folder ${BUILD_DIR} already exists. Skipping samples building."
    printf "If you want to rebuild samples, remove the entire ${BUILD_DIR} folder. ${RUN_AGAIN}"
fi

if [ ! -e $COMPILED_APP_DIR ]; then
   printf "\n\nTarget folder ${COMPILED_APP_DIR} does not exists.\n"
   exit 1
else
   cp ${COMPILED_APP_DIR}/benchmark_app ${AIXPRT_BIN}
#   cp ${COMPILED_APP_DIR}/image_classification ${AIXPRT_BIN}
#   cp ${COMPILED_APP_DIR}/image_classification_async ${AIXPRT_BIN}
#   cp ${COMPILED_APP_DIR}/object_detection_ssd ${AIXPRT_BIN}
#   cp ${COMPILED_APP_DIR}/object_detection_ssd_async ${AIXPRT_BIN}

   if [[ -z $USE_PREBUILT_OPENVINO ]]; then
     # these are found after compiling AIXPRT
     cp ${COMPILED_APP_DIR}/lib/libformat_reader.so ${AIXPRT_PLUGIN}/
     cp ${COMPILED_APP_DIR}/lib/libcpu_extension.so ${AIXPRT_PLUGIN}/
   fi
fi

#========================= Copy libraries ================================================================================
# Step 6. Finally copy plugins folder

#printf ${DASHES}
#printf "Copying OpenVINO Libraries"

PLUGIN_DIR="$CUR_PATH/plugin"
if [ -d "${PLUGIN_DIR}" ]; then rm -Rf $PLUGIN_DIR; fi
mkdir ${PLUGIN_DIR}

if [ $DISTRO == "centos" ];then
   OPERATING_SYSTEM="centos_7.4"
   OS_VERSION="centos7"
elif [ $DISTRO == "ubuntu" ];then
   OPERATING_SYSTEM="ubuntu_${RELEASE}" # Options: ubuntu_16.04 centos_7.4
   if [ ${RELEASE} == "16.04" ]; then
      OS_VERSION="ubuntu16"
   elif [ ${RELEASE} == "18.04" ]; then
      OS_VERSION="ubuntu18"
   fi
fi

if [[ -z $USE_PREBUILT_OPENVINO ]]; then
# Due to directory structure change in R1 and above releases
if [ "${OPENVINO_BUILD}" == "R1 and above" ]; then
   OPERATING_SYSTEM=""

   cpu_extension=$OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension.so
   avx2_extension=$OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so
   avx512_extension=$OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx512.so
   myriad_plugin=$OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so
   mkl_tiny_lnx=$OPENVINO_IE_DIR/external/mkltiny_lnx/lib/
   hddl_plugin=$OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so
   if [ -f "$cpu_extension" ]; then
    cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension.so $PLUGIN_DIR
   fi
   if [ -f "$avx2_extension" ]; then
    cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
   fi
   if [ -f "$avx512_extension" ]; then
    cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx512.so $PLUGIN_DIR
   fi
   if [ -f "$myriad_plugin" ]; then
    cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR
   fi
   if [ -d "$mkl_tiny_lnx" ]; then
    cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/* $PLUGIN_DIR
   fi
   

   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNN64.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/plugins.xml $PLUGIN_DIR

   

   # Copy tbb libraries
   if [ -d "${OPENVINO_IE_DIR}/external/tbb/lib/" ]; then
      find ${OPENVINO_IE_DIR}/external/tbb/lib/ -type f -name 'libtbb.so*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
      find ${OPENVINO_IE_DIR}/external/tbb/lib/ -type f -name 'libtbbmalloc.so*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
   fi

   # Copy omp libraries 

    if [ -d "${OPENVINO_IE_DIR}/external/omp/lib/" ]; then
      find ${OPENVINO_IE_DIR}/external/omp/lib/ -type f -name 'libiomp5.so*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
    fi

   # Copy HDDL libraries
   if [ -d "$hddl_plugin" ]; then
    find ${OPENVINO_IE_DIR}/external/hddl/lib/ -type f -name 'lib*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
    cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so $PLUGIN_DIR
   fi


   # openCV
   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_core*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_imgcodecs*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_imgproc*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
   # Copy the specific python version of ie_api.so plugin
   pythonVersion=$(python3 -c 'import platform; major, minor, patch = platform.python_version_tuple(); print(major+"."+minor)')
   pushd $OPENVINO_DIR/python/
   pythonDirectories=$(ls)
   for D in $pythonDirectories; do
    subDir=$OPENVINO_DIR/python/$D
    if [ -d $subDir ]; then
      if [[ $subDir == *$pythonVersion* ]]; then
         cp ${OPENVINO_DIR}/python/$D/openvino/inference_engine/ie_api.so ${PLUGIN_DIR}
      fi
   fi
   done
   popd

else

   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNN64.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR

   cp $OPENVINO_IE_DIR/external/omp/lib/libiomp5.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/libmkl_tiny_omp.so $PLUGIN_DIR

   # HDDL libraries
   if [ $DISTRO == "ubuntu" ];then
      cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so $PLUGIN_DIR
      find ${OPENVINO_IE_DIR}/external/hddl/lib/ -type f -name 'lib*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
   fi

   # openCV
   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_core*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_imgcodecs*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -name 'libopencv_imgproc*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   cp ${OPENVINO_DIR}/python/python3.5/${OS_VERSION}/openvino/inference_engine/ie_api.so ${PLUGIN_DIR}

fi

cp -r ${PLUGIN_DIR}/* ${AIXPRT_PLUGIN}/
fi

#printf ${DASHES}
#======================= Remove temp directories =================================================
# Step 7. Post-install Clean-up

cd ${CUR_PATH}
AIXPRT_compiled="${CUR_PATH}/AIXPRT_compiled"
AIXPRT_sources="${CUR_PATH}/AIXPRT_sources"
#plugin="${CUR_PATH}/plugin"
models="${CUR_PATH}/models"
caffe_models="${CUR_PATH}/caffe_models"

if [ -d "${AIXPRT_compiled}" ]; then rm -Rf ${AIXPRT_compiled}; fi
if [ -d "${AIXPRT_sources}" ]; then rm -Rf ${AIXPRT_sources}; fi
if [ -d "${plugin}" ]; then rm -Rf ${plugin}; fi
if [ -d "${models}" ]; then rm -Rf ${models}; fi
if [ -d "${caffe_models}" ]; then rm -Rf ${caffe_models}; fi


printf ${DASHES}
echo "OpenVINO Directory: ${OPENVINO_DIR}" > ${AIXPRT_DIR}/Harness/OpenVINO_BUILD.txt
echo -e "\e[1;32mSetup completed successfully.\e[0m"
printf ${DASHES}
exit 1
