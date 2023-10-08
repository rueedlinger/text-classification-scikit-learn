DATA := data/
TRAIN := model.pkl
LANGUAGES := languages.csv
LANG := 

.PHONY: all
all: prepare


$(LANGUAGES):
	for lang in 1 2 3 4 ; do \
        echo $$number ; \
    done

.PHONY: prepare
prepare: $(LANGUAGES) $(DATA)

$(DATA):
	python lib/data.py

.PHONY: train
train: $(TRAIN)

$(train): prepare
	python lib/train.py

.PHONY: clean
clean:
	rm -rf $(DATA)
	rm -f $(TRAIN)
	rm -f $(LANGUAGES)