#!/bin/bash -e
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -x

skip_zeroing_disk=0
if [ -e "$HOME/skip_zeroing_disk" ]; then
    echo "NOTE: will not zero disk at the end due to VMWare Fusion bug"
    echo "See: https://communities.vmware.com/t5/VMware-Fusion-Discussions/VMWare-Fusion-Pro-11-15-6-16696540-causes-macOS-crash-during/m-p/2284011#M139190"
    skip_zeroing_disk=1
fi

sudo apt update
sudo apt install -y build-essential wget ca-certificates
sudo apt-get --purge remove modemmanager  # required to access serial ports.

# Python setup (TVM support for Python 3.9+ is iffy)
sudo apt install python3.8
alias python=python3.8

# Install Arduino-CLI (latest version)
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s

# 3rd party board URLs
ADAFRUIT_BOARDS_URL="https://adafruit.github.io/arduino-board-index/package_adafruit_index.json"
ESP32_BOARDS_URL="https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_dev_index.json"
SPARKFUN_BOARDS_URL="https://raw.githubusercontent.com/sparkfun/Arduino_Boards/master/IDE_Board_Manager/package_sparkfun_index.json"
SEEED_BOARDS_URL="https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json"
SPRESENSE_BOARDS_URL="https://github.com/sonydevworld/spresense-arduino-compatible/releases/download/generic/package_spresense_index.json"
arduino-cli core update-index --additional-urls $ADAFRUIT_BOARDS_URL,$ESP32_BOARDS_URL,$SPARKFUN_BOARDS_URL,$SEEED_BOARDS_URL,$SPRESENSE_BOARDS_URL

# Install supported cores from those URLS
arduino-cli core install arduino:mbed_nano # Arduino Nano BLE
arduino-cli core install arduino:sam # Arduino Due
arduino-cli core install SPRESENSE:spresense --additional-urls $SPRESENSE_BOARDS_URL # Sony Spresense
arduino-cli core install adafruit:samd --additional-urls $ADAFRUIT_BOARDS_URL # Adafruit PyBadge
arduino-cli core install esp32:esp32 --additiona-urls $ESP32_BOARDS_URL # Adafruit FeatherS2

OLD_HOSTNAME=$(hostname)
sudo hostnamectl set-hostname microtvm
sudo sed -i.bak "s/${OLD_HOSTNAME}/microtvm.localdomain/g" /etc/hosts

# TVM deps
sudo apt install -y llvm

# ONNX deps
sudo apt install -y protobuf-compiler libprotoc-dev

# Clean box for packaging as a base box
sudo apt-get clean
if [ $skip_zeroing_disk -eq 0 ]; then
    echo "Zeroing disk..."
    EMPTY_FILE="$HOME/EMPTY"
    dd if=/dev/zero "of=${EMPTY_FILE}" bs=1M || /bin/true
    if [ ! -e "${EMPTY_FILE}" ]; then
        echo "failed to zero empty sectors on disk"
        exit 2
    fi
    rm -f "${EMPTY_FILE}"
else
    echo "NOTE: skipping zeroing disk due to command-line argument."
fi
