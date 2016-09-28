#!/bin/bash

# stop on error
set -e
############################################
# install into /mnt/bin
sudo mkdir -p /mnt/bin
sudo chown ubuntu:ubuntu /mnt/bin

# install the required packages
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get -y install linux-headers-$(uname -r) linux-image-extra-`uname -r`

# install cuda
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

# get cudnn
# it needs to register as nVidia developer prior to download via 
#   https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/7.5/cudnn-7.5-linux-x64-v5.1-tgz
# download it first and share it via Dropbox URL here
CUDNN_FILE=cudnn-7.5-linux-x64-v5.1.tar
wget https://www.dropbox.com/s/pi6mndh5tpvm3k7/cudnn-7.5-linux-x64-v5.1.tar?dl=0 -O ${CUDNN_FILE}
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

# install anaconda
wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
bash Anaconda3-4.1.1-Linux-x86_64.sh -b -p /mnt/bin/anaconda3
rm Anaconda3-4.1.1-Linux-x86_64.sh 
echo 'export PATH="/mnt/bin/anaconda3/bin:$PATH"' >> ~/.bashrc

# install tensorflow
export TF_BINARY_URL='https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl'
/mnt/bin/anaconda3/bin/pip install $TF_BINARY_URL

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

###
# Reference:
# (1) http://max-likelihood.com/2016/06/18/aws-tensorflow-setup/
# (2) https://medium.com/@giltamari/tensorflow-getting-started-gpu-installation-on-ec2-9b9915d95d6f
###