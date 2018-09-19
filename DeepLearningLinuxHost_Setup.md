## Setup Environment

For best compatibility, CLI commands are recommanded through `ssh`, instead of GUI. 

### Step 1: Ubuntu 16.04

Refer [Ubuntu Site](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop-1604#0)


 * Install Ubuntu system update

    ```
    $ sudo apt-get update
    $ sudo apt-get upgrade
    ```
    * Note:

        You might meet version issues while installing some packages if not performed apt-get update & upgrade first. 
        To resolve that specific version issue, you might resolve it via specifying its dependence, for example,
        ```
        $ sudo apt-get install -y openssh-server (FAILED!)
            openssh-server : Depends: openssh-client (= 1:7.2p2-4)
        $ sudo apt-get install -y --allow-downgrades openssh-client=1:7.2p2-4
        $ sudo apt-get install -y openssh-server (OK!)

        $ sudo apt-get install nvidia-390 (FAILED!)
            nvidia-390 : Depends: lib32gcc1 but it is not going to be installed
                         Depends: libc6-i386 but it is not going to be installed
  	    $ sudo apt-get install libc6-i386
		    libc6-i386 : Depends: libc6 (= 2.23-0ubuntu3) but 2.23-0ubuntu9 is to be installed
	    # sudo apt-get install libc6=2.23-0ubuntu3
        $ sudo apt-get install nvidia-390 (OK!)
        ```
* Install OpenSSH server, allow remote ssh login
    ```
    $ sudo apt-get install -y openssh-server
    ```

### Step 2: nVidia Driver

* Remove *incompatible* open source nvidia driver from kernel
    ```
    $ sudo bash -c "echo -e \"blacklist nouveau\nalias nouveau off\" > /etc/modprobe.d/nvidia.conf"
    $ sudo update-initramfs -u
    $ reboot
    ```

* Stop X-Window
    ```
    $ sudo service lightdm stop
    ```

* Install Ubuntu NVidia driver 
    ```
    $ sudo add-apt-repository ppa:graphics-drivers/ppa
    $ sudo apt-get update
    $ sudo apt-get install nvidia-390
    $ sudo reboot
    ```
    * Note:

        * It's not recommended to download and install nvidia driver directly from https://www.nvidia.com/Download/index.aspx, still suffer from it from time to time...  
        * There is compatibility issue of nVidia 396 dirver and 1080Ti GPU, which causes looping X-Window login screen.
        * For accessing GCE or Azure VM without GUI, it's OK to install nvidia-396 driver (nvidia 390 driver is not applicable to CUDA 9.2)
        * You might check nvidia driver via 'nvidia-smi' or 'cat /proc/driver/nvidia/version'
        * To remove existing nvidia driver & cuda, try 'sudo apt-get purge nvidia*', then perform aforemnetioned steps to install needed driver again. 

### Step 3: Install Docker & NVidia runtime

* Install Docker Community Edition
    ```
    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    $ sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
    $ sudo apt-get update
    $ sudo apt-get install docker-ce
    ```
    * Note:
        refer to https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository

* Allow non-root user to run docker image
    ```
    $ sudo usermod -aG docker $USER
    ```

* Test hello world container
    ```
    // Log out and log back in so that your group membership is re-evaluated.
    $ docker run --rm hello-world
    ```

* Install nvidia-runtime
    ```
    $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    $ curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    $ sudo apt-get update
    $ sudo apt-get install nvidia-docker2
    $ sudo pkill -SIGHUP dockerd
    ```

* Validate runtime environment
    ```
    $ docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
    ```

### Step 4: Setup host level environment (Optional)

* Install Python Virtual Environment
    ```
    $ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

    add 3 lines to .bashrc:
        export PATH="~/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"

    # following packages used while installing python 3.6.5 
    $ sudo apt-get install zlib1g-dev libbz2-dev libreadline-dev libssl-dev libsqlite3-dev 
    $ pyenv install 3.6.5
  	$ pyenv virtualenv 3.6.5 pytorch-p365
  	$ pyenv activate pytorch-p365 
    ```

* Install PyTorch 
    ```
    (CUDA 9.2, nVidia driver version > 390)
  	$ pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl 
  	$ pip install torchvision
	OR 
    (CUDA 9.0, nVidia driver version 390)
	$ pip install torch torchvision
    
    $ pip install -r requirements.txt
    ```
    * Note: 
        * An example of <b>requirements.txt</b>
            ```
            numpy >= 1.14
            dash >= 0.26
            dash-core-components >= 0.28
            dash-html-components >= 0.11
            plotly >= 3.1
            Flask >= 1.0
            Flask-Caching >= 1.4.0
            Pillow >= 5.0
            scikit-image >= 0.13
            scikit-learn >= 0.19
            ```
        * It's crucial to assure the compatibility of CUDA version and nVidia driver version.
        * Check accompanying CUDA version packaged in PyTorch
            ```
            >>> import torch
            >>> print(torch.version.cuda)
            ```

* Install TensorFlow
    * Install CUDA Toolkit

        Do NOT go with CUDA installer package, use local run file instead
        ```
        (CUDA 9.1)
        $ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
        $ sudo sh cuda_9.1.85_387.26_linux.run
        $ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/1/cuda_9.1.85.1_linux
        $ sudo sh cuda_9.1.85.1_linux.run
        $ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/2/cuda_9.1.85.2_linux
        $ sudo sh cuda_9.1.85.2_linux.run
        $ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda_9.1.85.3_linux
        $ sudo sh cuda_9.1.85.3_linux.run 
        $ sudo bash -c "echo /usr/local/cuda-9.1/lib64/ > /etc/ld.so.conf.d/cuda.conf"
        $ sudo ldconfig
        $ echo "export PATH=\"/usr/local/cuda-9.1/bin:\$PATH\"" >> ~/.bashrc

        OR 

        (CUDA 9.2)
        $ wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
        $ sudo sh cuda_9.2.148_396.37_linux.run
        $ wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux
        $ sudo sh cuda_9.2.148.1_linux.run
        $ sudo bash -c "echo /usr/local/cuda-9.2/lib64/ > /etc/ld.so.conf.d/cuda.conf"
        $ sudo ldconfig
        $ echo "export PATH=\"/usr/local/cuda-9.2/bin:\$PATH\"" >> ~/.bashrc   
        ```
        * Note:
            * Choose NOT to install nVidia driver while installing CUDA Toolkit (asked while executing runfile)
            * https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation


    * Install CuDNN

        TO BE ADDED
    