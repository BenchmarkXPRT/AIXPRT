#!/bin/bash
IFS=':' read -ra OPENVINO_BUILD < "../Harness/OpenVINO_BUILD.txt"

OPENVINO_DIR=${OPENVINO_BUILD[1]}
if [ ! -d ${OPENVINO_DIR} ]; then
   echo -e "\e[1;31mCannot find OpenVINO installation directory ${OPENVINO_DIR}\e[0m"
   exit
fi

if [ ! $( pidof hddldaemon ) ]; then

  echo "Loading HDDL dependencies"
  sudo rm -rf /var/tmp/hddl_*
  source ${OPENVINO_DIR}/bin/setupvars.sh
  ${OPENVINO_DIR}/deployment_tools/inference_engine/external/hddl/bin/hddldaemon

fi
