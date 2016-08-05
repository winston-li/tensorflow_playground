FROM gcr.io/tensorflow/tensorflow

# Setup environment variables
ENV HOME=/root \
	DEBIAN_FRONTEND=noninteractive \
	LANG=en_US.UTF-8 \
	LANGUAGE=en_US.UTF-8 \
	LC_ALL=en_US.UTF-8 \
	DISPLAY=:0.0 \
	DISPLAY_WIDTH=1280 \
	DISPLAY_HEIGHT=800

# Install dev toolkits
RUN apt-get update && apt-get install -y \
	fluxbox \
	socat \
	supervisor \
	x11vnc \
	rxvt-unicode \
	xvfb 

# Clone noVNC from github
RUN git clone https://github.com/kanaka/noVNC.git /root/noVNC && \
	git clone https://github.com/kanaka/websockify /root/noVNC/utils/websockify && \
	rm -rf /root/noVNC/.git && \
	rm -rf /root/noVNC/utils/websockify/.git && \
	ln -s /root/noVNC/vnc.html /root/noVNC/index.html


# Configure xterm font size
RUN echo -e "xterm**font: 9x15\nurxvt*font: 9x15" > /etc/X11/Xresources

# Add long-runing service config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Modify the launch script 'ps -p'
RUN sed -i -- "s/ps -p/ps -o pid | grep/g" /root/noVNC/utils/launch.sh

EXPOSE 8080
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]