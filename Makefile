DATA := data/
TRAIN := model.pkl


.PHONY: all
all: prepare

.PHONY: prepare
prepare: $(DATA)

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