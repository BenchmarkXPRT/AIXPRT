#!/bin/bash
DASHES="\n\n==========================================================\n\n"


#=================================== Get OpenVINO install directory ================================
# IFS=':' read -ra OPENVINO_BUILD < "../Harness/OpenVINO_BUILD.txt"

# OPENVINO_INSTALL_DIR=${OPENVINO_BUILD[1]}
OPENVINO_INSTALL_DIR=/opt/intel/openvino
if [ ! -d ${OPENVINO_INSTALL_DIR} ]; then
   echo -e "\e[1;31mCannot find OpenVINO installation directory ${OPENVINO_INSTALL_DIR}\e[0m"
   exit
fi

if [ ! -d ${OPENVINO_INSTALL_DIR} ]
  then
      echo -e "\e[0;32mOpenVINO installation not found in /opt/intel. Please install OpenVINO in /opt/intel before continuing.\e[0m"
      exit
fi

HDDL_INSTALL_DIR=${OPENVINO_INSTALL_DIR}/deployment_tools/inference_engine/external/hddl

#=========================== Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2 =============================
printf ${DASHES}
printf "Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2"
printf ${DASHES}

CUR_DIR=$PWD

# USB rules
sudo usermod -a -G users "$(whoami)"
if [ ! -e "${CUR_DIR}/97-usbboot.rules" ];then 
  echo -e "\e[0;31mUSB boot rules not found. Please include boot rules file\e[0m";
  exit
fi
sudo cp 97-usbboot.rules /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo ldconfig

echo -e "\e[1;32mIntel Compute Stick support added.\e[0m"
#=========================== Support Intel® Vision Accelerator Design with Intel® Movidius™ VPUs =============================
printf ${DASHES}
printf "Setting up support for Intel® Vision Accelerator Design with Intel® Movidius™ VPUs"
printf ${DASHES}

# Install dependencies
sudo apt install libusb-1.0-0 libboost-program-options1.58.0 libboost-thread1.58.0 libboost-filesystem1.58.0 libssl1.0.0 libudev1 libjson-c2
sudo usermod -a -G users "$(whoami)"
cd ${HDDL_INSTALL_DIR}

# Setup rules
sudo chmod +x ./generate_udev_rules.sh
sudo bash generate_udev_rules.sh /etc/udev/rules.d/98-hddlbsl.rules
sudo sed -i "s/\(.*i2c_i801$\)/#\1/g" /etc/modprobe.d/blacklist.conf
sudo modprobe i2c_i801

# Install drivers
kill -9 $( pidof hddldaemon autoboot)
cd ${HDDL_INSTALL_DIR}/drivers
sudo chmod +x ./setup.sh
sudo bash setup.sh install

sudo cp -av ${HDDL_INSTALL_DIR}/../97-myriad-usbboot.rules /etc/udev/rules.d/

sudo cp -av ${HDDL_INSTALL_DIR}/etc /
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo ldconfig
printf "VPU support added."
# # =========================== INSTALL intel-opencl driver for integrated GPU ============================= 
# # INSTALL intel-opencl driver for integrated GPU
# printf ${DASHES}
# printf " Installing OpenCL drivers for Integrated GPU."
# printf ${DASHES}

# cd "${OPENVINO_INSTALL_DIR}/install_dependencies"
# KERNEL_VERSION="$(uname -r | cut -c 1-4)" # Minimum kernel is 4.14
# if [ $(echo "${KERNEL_VERSION}" | cut -c 1) -lt "4" ]
#    then
#        bash install_4_14_kernel.sh
# fi

# if [ $(echo "${KERNEL_VERSION}" | cut -c 1) -eq "4" ] 
#    then 
#        if [ $(echo "${KERNEL_VERSION}" | cut -c 3-4) -lt "15" ]
#           then
#               bash install_4_14_kernel.sh
#        fi
# fi

# sudo bash install_NEO_OCL_driver.sh
# sudo apt-get -y install clinfo
# clinfo

# printf "\n\nOpenCL installation completed."
# printf ${DASHES}
# echo -e "\e[1;33mWARNING: \e[0mSystem reboot required."
# printf ${DASHES}
# exit 1


