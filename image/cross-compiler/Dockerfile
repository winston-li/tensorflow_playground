FROM sdthirlwall/raspberry-pi-cross-compiler

# Install some native build-time tools
# For Protobuf
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y autoconf automake libtool maven unzip
# For Bazel
RUN echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee /etc/apt/sources.list.d/webupd8team-java.list && \
    echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886 && \
    apt-get update && \
    echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y oracle-java8-installer && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config zip g++ zlib1g-dev

# For TensorFlow in Python 2.7
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python-pip python-numpy swig python-dev && pip install wheel