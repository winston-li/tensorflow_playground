#!/bin/bash

# stop on error
set -e
############################################

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE="en_US.UTF-8"
sudo locale-gen en_US.UTF-8

# install the required packages
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get -y install linux-headers-$(uname -r) linux-image-extra-`uname -r`

## Starting from TensorFlow 0.11.0, its prebuilt binary needs CUDA 8.0
# install CUDA 8.0
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
rm cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# get cuDNN 5.1
# it needs to register as nVidia developer prior to download via 
#   https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-linux-x64-v5.1-tgz
# download it first and share it via Dropbox URL here
CUDNN_FILE=cudnn-8.0-linux-x64-v5.1.tar
wget https://www.dropbox.com/s/zypujn2cbz2rvem/cudnn-8.0-linux-x64-v5.1.tar?dl=0 -O ${CUDNN_FILE}
tar xvf ${CUDNN_FILE}
rm ${CUDNN_FILE}
# move library files to /usr/local/cuda
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
rm -rf cuda

# set the appropriate library path
echo 'export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
' >> ~/.bashrc

# install python rerequisites
sudo apt-get install -y python-pip python-dev
sudo pip install pbr funcsigs numpy future

# install tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

# install monitoring programs
sudo wget https://git.io/gpustat.py -O /usr/local/bin/gpustat
sudo chmod +x /usr/local/bin/gpustat
sudo nvidia-smi daemon
sudo apt-get -y install htop

# reload .bashrc
exec bash

############################################
# run the test
# byobu				# start byobu + press Ctrl + F2 
# htop				# run in window 1, press Shift + F2
# watch --color -n1.0 gpustat -cp	# run in window 2, press Shift + <left>
# wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/models/image/mnist/convolutional.py
# python convolutional.py		# run in window 3

##### Validation: 
## (1) check nvidia gpu exist
# lspci | grep -i nvidia 
## (2) check cuda installation OK
# nvcc --version
## (3) check cudnn version
# cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

##### Azure NC-series GPU info:
# Winston@pathtrace1404:~$ lspci | grep -i nvidia
#   97f8:00:00.0 3D controller: NVIDIA Corporation GK210GL [Tesla K80] (rev a1)
# Winston@pathtrace1404:~$ gpustat
#   3pathtrace1404  Mon Dec  5 08:07:20 2016
#   [0] Tesla K80        | 36'C,   0 % |     0 / 11441 MB |
#
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 97f8:00:00.0
# Total memory: 11.17GiB
# Free memory: 11.11GiB

###
# Reference:
# (1) http://max-likelihood.com/2016/06/18/aws-tensorflow-setup/
# (2) https://medium.com/@giltamari/tensorflow-getting-started-gpu-installation-on-ec2-9b9915d95d6f
# (3) http://docs.nvidia.com/cuda/cuda-installation-guide-linux/
###
