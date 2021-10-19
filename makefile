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

GGPU?=0



start : start0

start0 :
	ssh -X -q D9466.Idcc.lab -l chevallierl "cd $$PWD; make start1"

TARGET=lmm

start1 :
	hostname
	-source ~/.bashrc; spy;  hostname; buildthenenv; MINEKOLEVEL=0 PYTHONPATH=$(GIT)/utils:$(GIT) CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU) make $(TARGET)


PREF=$(PWD)/../../validation/grand_test/facelab2

cell :
	CVD=$(GGPU) CUDA_VISIBLE_DEVICES=$(GGPU)  $(SET) python cell.py 




