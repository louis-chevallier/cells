removetrailingslash = ${1:/=}
upfolder = $(dir $(call removetrailingslash,${1}))
isnotgit = $(subst git,,$(call removetrailingslash,$(notdir $(call removetrailingslash,$(1)))))
goupuntilgit = $(if $(call isnotgit,$(1)),$(call goupuntilgit,$(call upfolder,$(1))),$(1))
MVDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GIT ?=$(call goupuntilgit,$(MVDIR))

$(warning   $(MAKEFILE_LIST)  )
$(warning   $(lastword $(MAKEFILE_LIST)))
$(warning  $(abspath $(lastword $(MAKEFILE_LIST))))
$(warning  $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
$(warning $(MVDIR))
_PYTHONPATH2=$(GIT)
FACELABREPO:=$(MVDIR)/../..
include $(GIT)/utils/inc.mk
export FACELABREPO
#$(warning PREF = $(PREF))
$(warning FACELABREPO=$(FACELABREPO))
FACELAB_LOG_LEVEL?=INFO
export DATA_FOLDER?=/home/wp24b

GGPU?=0,1
DEBUG=True
MODEL?=deeplab


startD : 
	ssh -X -q D9466.Idcc.lab -l chevallierl "cd $$PWD; DEBUG=False MODEL=deeplab make start1"

start : 
	ssh -X -q D9468.Idcc.lab -l chevallierl "cd $$PWD; DEBUG=True MODEL=unet make start1"

TARGET=train

start1 :
	hostname
	-source ~/.bashrc; spy;  source buildenv.sh ; hostname; buildtheenv; MINEKOLEVEL=0 PYTHONPATH=$(GIT)/utils:$(GIT) CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU) make $(TARGET)


PREF=$(PWD)/../../validation/grand_test/facelab2

cclean :
	rm -f /home/wp01/tmp/images/*.png
	rm -f *.trc


metric : 
	CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU)  $(SET) python metric.py 

#PTS=$(DATE).pts
PTS=m1.pts

#-m pdb

cell : kill cclean
	hostname
	CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU)  $(SET) python  cell13.py --debug $(DEBUG) --pts $(PTS) --synthetic True --model $(MODEL) 2>&1 | tee $(@)_$(DATE).trc

train : kill cclean
	hostname
	CVD=$(GGPU) NUMEXPR_MAX_THREADS=2 CUDA_VISIBLE_DEVICES=$(GGPU)  $(SET) python  training.py 2>&1 | tee $(@)_$(DATE).trc


prepdata : 
	CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU)  $(SET) python  tojson.py


kill :
	-source ~/.bashrc; spy; terminate training; terminate cell13; sleep 5 


submit :
	kg submit submission__unet_0009.csv -u lmdm99 -p <password> -c <competition> -m "<message>"
