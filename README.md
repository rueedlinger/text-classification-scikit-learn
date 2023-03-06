# Text Classification Examples (Python)

## Setup
```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 py3.11
pyenv activate py3.11
```

## Download Training Data from Wikipedia

```bash
usage: dowload.py [-h] [--out OUT] [--file FILE] [--samples SAMPLES]
```

```bash
python lang-clf/dowload.py --out data --samples 100 --file iso-639-1.csv
```

## Train the Model

```bash
usage: train.py [-h] [--data DATA]
```

```bash
python lang-clf/train.py --data
```

## Use the Model
```bash
python lang-clf/predict.py
```