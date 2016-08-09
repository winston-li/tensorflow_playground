FROM gcr.io/tensorflow/tensorflow

RUN apt-get update  && \ 
	apt-get install software-properties-common && \
	add-apt-repository ppa:mc3man/trusty-media && \
	apt-get update  && apt-get dist-upgrade && \
	apt-get install -y ffmpeg

