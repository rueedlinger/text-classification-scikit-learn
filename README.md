# Text Classification Examples (Python)

Simple example how to train a model (scikit-learn) which can predict the language of written text. The model uses samples from different wikipedia subdomains.

## Setup
```bash
pyenv install 3.11.0
pyenv virtualenv 3.11.0 py3.11
pyenv activate py3.11
pip install -r requirements.txt
```

## Download Training Data from Wikipedia

```
usage: data.py [-h] [--out OUT] [--file FILE] [--samples SAMPLES]
```

Download some text samples in different languages based on iso-639-1 code (see [iso-639-1.csv](iso-639-1.csv)) from wikipedia subdomains.

```bash
python lib/data.py --out data --samples 10 --file iso-639-1.csv
```

## Train the Model

```
usage: train.py [-h] [--data DATA]
```

```bash
python lib/train.py --data data
```

## Use the Model

```bash
python lib/predict.py
```