CROPSIZE := 128

IMAGESETS = $(shell ls dataset)

all: $(IMAGESETS)

$(IMAGESETS):
	ISOS = $(shell ls dataset/$@ | grep -o 'ISOH*[0-9]*'); \
	for iso in $(ISOS); do \
		mkdir -p dataset_$(CROPSIZE)/$(FN)/iso; \
	done
