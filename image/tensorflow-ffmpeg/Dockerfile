FROM gcr.io/tensorflow/tensorflow

RUN apt-get update  && \ 
	apt-get install -y software-properties-common && \
	add-apt-repository -y ppa:mc3man/trusty-media && \
	apt-get update && apt-get -y dist-upgrade && \
	apt-get install -y ffmpeg

