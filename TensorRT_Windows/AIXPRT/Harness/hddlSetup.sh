#!/bin/bash

if [ ! $( pidof hddldaemon ) ]; then

  echo "Loading HDDL dependencies"
  sudo rm -rf /var/tmp/hddl_*
  source /opt/intel/computer_vision_sdk/bin/setupvars.sh
  /opt/intel/computer_vision_sdk/deployment_tools/inference_engine/external/hddl/bin/hddldaemon

fi
