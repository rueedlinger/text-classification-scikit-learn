# Text Classification Examples (Python)

Simple example how to train a model (scikit-learn) which can predict the language of written text. The model uses samples from different wikipedia subdomains.

## Setup

The following modules are used:
- sklearn
- urllib3

```bash
pip install scikit-learn
pip install urllib3
```

Or you can just use [pyenv](https://github.com/pyenv/pyenv) to set up the project and install the modules with `pip install`.

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

Download some random text samples in different languages based on iso-639-1 code (see [iso-639-1.csv](iso-639-1.csv)) from 
wikipedia subdomains. The text samples are stored in the [data](data) directory.

```bash
python lib/data.py --out data --samples 100 --file iso-639-1.csv
```

The output might look like this:

```
language file: iso-639-1.csv
samples: 50
checking 6 WIKIPEDIA subdomains
checking de
checking en
checking fr
checking ja
checking es
checking zh
loading language data: de (1/6)
loading language data: en (2/6)
loading language data: fr (3/6)
loading language data: ja (4/6)
loading language data: es (5/6)
loading language data: zh (6/6)
```

## Train the Model

```
usage: train.py [-h] [--data DATA]
```

The next step is to build features from the text samples (`sklearn.feature_extraction.text.TfidfVectorizer`) 
and train the model (`sklearn.naive_bayes.MultinomialNB`) on the previous downloaded text samples.


To train the model just run:

```bash
python lib/train.py --data data
```
This step will split the text samples in a train and test set.
The model optimization is done using the train set with k-fold cross validation.
The last step is to persist fitted model with `pickle` as `model.pkl`.   

The output might look like this:

```
added ja samples (100)
added zh samples (100)
added de samples (99)
added fr samples (100)
added es samples (100)
added en samples (100)
labels:  6
samples:  599
train size (x, y):  359 359
test size (x, y):  240 240
create model definition Pipeline(steps=[('vect',
                 FeatureUnion(transformer_list=[('vec_word',
                                                 TfidfVectorizer(ngram_range=(1,
                                                                              2))),
                                                ('vec_char_wb',
                                                 TfidfVectorizer(analyzer='char_wb',
                                                                 ngram_range=(1,
                                                                              2))),
                                                ('vec_char',
                                                 TfidfVectorizer(analyzer='char',
                                                                 ngram_range=(1,
                                                                              2)))])),
                ('clf', MultinomialNB())])
evaluate train set with ['f1_weighted', 'accuracy'] and 5 folds
f1_weighted scores: [1.         1.         1.         0.97268673 0.97180649]
f1_weighted mean score: 0.9889
f1_weighted std 0.0136
f1_weighted score range (+/- 2 * std): [0.9617, 1.0161]
accuracy scores: [1.         1.         1.         0.97222222 0.97183099]
accuracy mean score: 0.9888
accuracy std 0.0137
accuracy score range (+/- 2 * std): [0.9614, 1.0162]
start fitting model
model fitting took 0.1158 seconds
top 10 features per label
de: ['vec_char_wb__r' 'vec_char_wb__n' 'vec_char__r' 'vec_char_wb__i'
 'vec_char__n' 'vec_char__i' 'vec_char_wb__e' 'vec_char__ ' 'vec_char__e'
 'vec_char_wb__ ']
en: ['vec_char_wb__i' 'vec_char__n' 'vec_char__t' 'vec_char_wb__a'
 'vec_char__i' 'vec_char__a' 'vec_char_wb__e' 'vec_char__e' 'vec_char__ '
 'vec_char_wb__ ']
es: ['vec_word__de' 'vec_char__n' 'vec_char__i' 'vec_char__o' 'vec_char_wb__a'
 'vec_char_wb__e' 'vec_char__a' 'vec_char__e' 'vec_char__ '
 'vec_char_wb__ ']
fr: ['vec_char__e ' 'vec_char__s' 'vec_char__i' 'vec_char_wb__a' 'vec_char__n'
 'vec_char__a' 'vec_char_wb__e' 'vec_char__e' 'vec_char__ '
 'vec_char_wb__ ']
ja: ['vec_char_wb__る' 'vec_char__る' 'vec_char_wb__。' 'vec_char__。'
 'vec_char_wb__、' 'vec_char__、' 'vec_char_wb__は' 'vec_char__は'
 'vec_char_wb__の' 'vec_char__の']
zh: ['vec_char__一' 'vec_char_wb__是' 'vec_char__是' 'vec_char_wb__ '
 'vec_char_wb__。' 'vec_char__。' 'vec_char_wb__的' 'vec_char__的'
 'vec_char_wb__，' 'vec_char__，']
number of features: 30623
              precision    recall  f1-score   support

          de     1.0000    1.0000    1.0000        40
          en     0.9524    1.0000    0.9756        40
          es     1.0000    1.0000    1.0000        40
          fr     1.0000    1.0000    1.0000        40
          ja     1.0000    1.0000    1.0000        40
          zh     1.0000    0.9500    0.9744        40

    accuracy                         0.9917       240
   macro avg     0.9921    0.9917    0.9917       240
weighted avg     0.9921    0.9917    0.9917       240

saving model
model size (Bytes): 3888074
model size (KiB): 3796.95
model size (MiB): 3.71
model size (GiB): 0.00
```

## Use the Model

In the last step we can try out our model.

```bash
python lib/predict.py
```

Let's provide a text sample (e.g "Hello my name is max").
```
loading model
loading model toke 0.8527 seconds
Please enter sentence: Hello my name is max.
```

The output might look like this:

```
---
Hello my name is max.
---
['en']
('en', 0.8238726600194233)
('de', 0.06591166352939336)
('es', 0.05989906412168451)
---
prediction took 0.0087 seconds
```

