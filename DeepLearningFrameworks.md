# Deep Learning Frameworks Alternatives

### Description
-----
It lists some metrics and reviews of deep learning framework(s), with key factors highlighted for adoption considerations.  


### Open Source Deep Learning Software Comparison (in alphabetical order)
-----
| Software | License | Written in   | Interface Languages | GPU | Multi-CPU | Multi-GPU | CNN | RNN | AD | Pre-trained Models | Commercial Support  | Specialties | GitHub Stars (2016/8) | Creator |
| ---           | ---    | ---         | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Caffe         | BSD | C++ | Python, C++, MATLAB | O |   | O | O | O |   | O |   | Image Classification, Embedded Device Deployment | 11,760 | Berkeley Vision & Learning Center 
| CNTK          | Free | C++ | CLI & its Network Definiton Language | O | O | O | O | O | O |   |   | RNN Training | 5,948 | Microsoft |
| DeepLearning4J| Apache 2.0 | Java, Scala | Java, Scala | O | O | O | O | O | O | O | O | Hadoop/Spark Integration | 3,651 | Skymind |
| DSSTNE        | Apache 2.0 | C++ | CLI & its Network Definition language | O | O | O |   |   |  |  |   | Sparse Datasets | 3,145 | Amazon |
| Tensorflow    | Apache 2.0 | C++ | Python, C/C++ | O | O | O | O | O | O |   |   | TensorBoard for visualization, Embedded Device Deployment | 29,501 | Google |
| Theano        | BSD | Python | Python | O |   |   | O | O | O |   |   |   | 4,271 | Montreal University 
| Torch         | BSD | Lua | Lua, C | O |   | O | O | O | O | O |   |   | 5,125 | Ronan Collobert, et al. |

    [CNN]: Convolutional Neural Network
    [RNN]: Recurrent Neural Network 
    [AD]: Automatic Differentiation
          AD let you train deep models without having to crank out the backpropagation algorithms for arbitrary architecture. 
          It is important because you don’t want to hand-code a new variation of backpropagation every time you’re experimenting with a new arrangement of neural networks.
    

### Key Consideration Factors
    * Active Community
    * Prototyping and Production Support
    * Deployable at Things
    * Scalable 

### Reference
* https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software#cite_note-17
* https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software/Resources
* https://www.microsoft.com/en-us/research/microsoft-computational-network-toolkit-offers-most-efficient-distributed-deep-learning-computational-performance/
* http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html#dsstne
* https://github.com/zer0n/deepframeworks
