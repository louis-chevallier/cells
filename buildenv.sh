function myhome() {
    if [ -d '/home/wp02/' ] ; then
        local ret="home"
    else
        local ret="srv"
    fi
    echo $ret
}

function testtorch() {
    python -c "
from pylab import *; 
import numpy as np, matplotlib as plt, sys; 
import torch; 
print(torch.__version__)
"
}

function buildtheenv() {
    source buildenv.sh
    #set -vx
    ANA=${PWD}/conda 
    # 9.2
    PYTHON_VERSION_=${PYTHON_VERSION:-3.8}
    CUDA_VERSION_=${CUDA_VERSION:-10.1}

    #printf ${PYTHON_VERSION_} 

    source  /$(myhome)/wp24b/tools/bashrc
    envcuda ${CUDA_VERSION_}

    if [ 0"$1" == 0install ]; then
        rm -fr ${ANA}
        DIST=Miniconda3-latest-Linux-x86_64.sh
        SOURCE=/home/wp01/tmp/${DIST}
        bash ${SOURCE} -p $ANA -b -f
        PATH=${ANA}/bin:${PATH} 
        which python
        python --version
        #conda update -y --prefix ${ANA}  anaconda
        conda update conda -y
        #conda install -y python=${PYTHON_VERSION_}
        which python
        python --version

        conda install -y python=3.8.8
        conda install -y -c pytorch pytorch=1.8 torchvision=0.9.1 cudatoolkit=${CUDA_VERSION_}
        conda install -y -c conda-forge -c fvcore -c iopath fvcore iopath
        conda install -y -c anaconda scikit-learn
        conda install -y matplotlib
        pip install loguru
        pip install scikit-image face-alignment h5py
        conda install -y cloudpickle  cycler  dask  decorator  imageio  kiwisolver  matplotlib  networkx  numpy  
        conda install -y pandas  Pillow  pycparser  pygit  pyparsing  python-dateutil  pytz  PyWavelets  
        conda install -y PyYAML  scikit-image  scikit-learn  scipy  six  toolz imageio-ffmpeg  tqdm  sympy         
        pip install tensorboard pycocotools


        #conda install -y -c pytorch torchvision
        #conda install -y -c conda-forge opencv
        pip install opencv-python
        pip install chumpy
        # conda install -y scipy
        # #conda install -y sklearn
        pip install pynvml tensorboard
        pip install einops
        pip install ptflops
        pip install thop
        pip install torchsummary
        pip install albumentations 
        pip install torchscan
        #pip install kaggle-cli
        pip install git+https://github.com/kornia/kornia
        #conda install -c conda-forge detectron2
        #python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
        #python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
     fi
    #cuda92
    PATH=${ANA}/bin:${PATH}
    #set +vx
    testtorch
}
