# Tensor Flow Playgrounds in Various Virtual Environments 
[![Docker Repository on Quay](https://quay.io/repository/draft/tensorflow-novnc/status "Docker Repository on Quay")](https://quay.io/repository/draft/tensorflow-novnc) 
## Introduction: 
      Here are some lessons learned and working approaches after many trials-and-errors to build tensor flow experiment virtual environments on Mac OSX. The built playgrounds should be able to experiment with various input / output interactively. Many online articles and FAQs are actually not working correctly, at least on my MacBook. 
      Two kinds of virtual environments are verified, 
        (1) python virtual environment
        (2) docker/container
      This one covers two specific parts:
        (1) a tensorflow container with novnc, so that X11 GUI is accessible via browser. 
        (2) a tensorflow container with ffmpeg, so that matplotlib animation could embed on Jupyter notebook. 
        
      Appendix section contains other verified working alternatives.  

### Setup the container with X11 GUI on browser
        docker pull quay.io/draft/tensorflow-novnc
        docker run -v $PWD:/source -d -p 8080:8080 quay.io/draft/tensorflow-novnc
        open http://localhost:8080/vnc.html

        // It might be easier to experiment with CLI and see the output figures on browser altogether
        docker exec -it <tensorflow-novnc container id> bash

### Setup the container with matplotlib animation on Jupyter notebook
        docker pull quay.io/draft/tensorflow-ffmpeg
        docker run -it -p 8888:8888 -v ${PWD}:/source --rm quay.io/draft/tensorflow-ffmpeg
        // remember to add followings to embed matplotlib animations
            (1) %matplotlib inline
            (2) from matplotlib import rc
            (3) from IPython.display import HTML
            (4) rc('animation', html='html5') 
	            ...
            (5) type <your_animation_variable> in a new cell // it will show the embedded animation video

            Note: refer to
            (1) http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
            (2) http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/   

## Appendix:
#### 1. Python virutal environment 
    (1) install specific python version:
        brew update
        brew install pyenv 
        brew install pyenv-virtualenv
        pyenv install 3.5.2  // install Python 3.5.2
        
    (2) create a virtual environment and activate it:
        eval "$(pyenv init -)" // add this line to ~/.bash_profile if needed
        pyenv virtualenv 3.5.2 tf-py352  // create virtual environment (tf-py352) with Python 3.5.2
        pyenv activate tf-py352 // activate tf-py352 virtual environment
        
        Related commands:
        pyenv deactivate <virtual-env> // deactivate specified virtual environment
        pyenv uninstall <virtual-env> // delete specified virtual environment
        pyenv versions // list installed python versions
        pyenv virtualenvs // list created virtual environments

    (3) install libraries/modules within virtual environment: 
        pip install numpy matplotlib ipython jupyter
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl
	    pip install --upgrade $TF_BINARY_URL
        
    (4) setup matplotlib backends on Mac:
        // get matplotlib configuration file place
        >>> import matplotlib
        >>> matplotlib.matplotlib_fname()
        // change backend to TkAgg via editor
        
        Notes:
        <1> default backend "macos" doesn't work with python virtual environments. furthermore, the workarounds listed on matplotlib.org's FAQ don't work as well. (http://matplotlib.org/faq/virtualenv_faq.html)
        <2> backend "Qt4Agg" doesn't work as well. To use it, it's needed to have PySide, which will take quite a long time to build. Even worse, after time/space spent for setup build environment/tool (Xcode w/ right SDK version) and built it successfully, it just still not work...
    
    (5) experiment with python: 
        // when use matplotlib.pyplot, its output will show on an interactive window 
        
    (6) experiment with ipython:
        // when use matplotlib.pyplot, its output will show on an interactive window
        
    (7) experiment with jupyter notebook:
        jupyter notebook // show a new tab for the notebook on the default browser 
        // if you'd like the output figure shown inlined instead of a pop out interactive window, you should add following line.
        %matplotlib inline

        Note:
        As mentioned earlier in "Setup the container with matplotlib animation on Jupyter notebook" section, it needs extra steps to embed matplotlib animation on Jupyter notebook.
        brew install ffmpeg // install ffmpeg on mac
        
        // remember to add followings to embed matplotlib animations
        (1) %matplotlib inline
        (2) from matplotlib import rc
        (3) from IPython.display import HTML
        (4) rc('animation', html='html5') 
	            ...
        (5) type <your_animation_variable> in a new cell // it will show the embedded animation video
        
#### 2. Docker/container:
    (1) download official tensor flow Docker image & launch it:
        docker pull gcr.io/tensorflow/tensorflow
        <1> Jupyter notebook
            docker run -it -p 8888:8888 -v ${PWD}:/source --rm gcr.io/tensorflow/tensorflow
            // remember to add "%matplotlib inline" to show matplotlib figures
            
        <2> CLI
            docker run -it -p 8888:8888 -v ${PWD}:/source --rm --entrypoint bash gcr.io/tensorflow/tensorflow
            // can not show matplotlib figures
            
    (2) download custom Docker image with tensor flow & novnc:
        // customize official docker image by adding novnc, to enable X11 GUI access via browser
        // refer to earlier "Setup the container with X11 GUI on browser" section
