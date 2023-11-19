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
python lib/data.py --out data --samples 50 --file iso-639-1.csv
```

The output might look like something like this:

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

After the training the model should be persisted with `pickle` as `model.pkl`.   

The output might look like this:

```
added ja samples (100)
added zh samples (100)
added de samples (98)
added fr samples (100)
added es samples (100)
added en samples (100)
labels:  6
samples:  598
train size (x, y):  358 358
test size (x, y):  240 240
create model definition Pipeline(steps=[('vect',
                 FeatureUnion(transformer_list=[('vec_word',
                                                 TfidfVectorizer(min_df=5,
                                                                 ngram_range=(1,
                                                                              5))),
                                                ('vec_char_wb',
                                                 TfidfVectorizer(analyzer='char_wb',
                                                                 min_df=5,
                                                                 ngram_range=(1,
                                                                              5)))])),
                ('clf', MultinomialNB())])
evaluate train set with ['f1_weighted', 'accuracy'] and 5 folds
f1_weighted scores: [1.       1.       1.       1.       0.985891]
f1_weighted mean score: 0.9972
f1_weighted std 0.0056
f1_weighted score range (+/- 2 * std): [0.9859, 1.0085]
accuracy scores: [1.         1.         1.         1.         0.98591549]
accuracy mean score: 0.9972
accuracy std 0.0056
accuracy score range (+/- 2 * std): [0.9859, 1.0085]
start fitting model
model fitting took 0.1480 seconds
top 10 features per label
de: ['vec_char_wb__r' 'vec_word__im' 'vec_char_wb__n' 'vec_char_wb__i'
 'vec_word__ein' 'vec_word__und' 'vec_word__ist' 'vec_word__der'
 'vec_char_wb__e' 'vec_char_wb__ ']
en: ['vec_char_wb__n' 'vec_char_wb__i' 'vec_char_wb__a' 'vec_word__in'
 'vec_word__and' 'vec_char_wb__e' 'vec_word__of' 'vec_word__is'
 'vec_word__the' 'vec_char_wb__ ']
es: ['vec_char_wb__n' 'vec_char_wb__o' 'vec_word__la' 'vec_word__el'
 'vec_word__es' 'vec_word__en' 'vec_char_wb__a' 'vec_char_wb__e'
 'vec_word__de' 'vec_char_wb__ ']
fr: ['vec_char_wb__t' 'vec_char_wb__n' 'vec_char_wb__a' 'vec_word__une'
 'vec_word__la' 'vec_word__le' 'vec_word__de' 'vec_word__est'
 'vec_char_wb__e' 'vec_char_wb__ ']
ja: ['vec_char_wb__あ' 'vec_char_wb__ン' 'vec_char_wb__ー' 'vec_char_wb__に'
 'vec_char_wb__る' 'vec_char_wb__。' 'vec_char_wb__、' 'vec_char_wb__は'
 'vec_char_wb__ ' 'vec_char_wb__の']
zh: ['vec_char_wb__市' 'vec_char_wb__。 ' 'vec_char_wb__一' 'vec_char_wb__是'
 'vec_char_wb__ ' 'vec_word__平方公里' 'vec_word__英語' 'vec_char_wb__。'
 'vec_char_wb__的' 'vec_char_wb__，']
number of features: 4992
              precision    recall  f1-score   support

          de     1.0000    1.0000    1.0000        40
          en     0.9524    1.0000    0.9756        40
          es     1.0000    0.9750    0.9873        40
          fr     1.0000    0.9750    0.9873        40
          ja     1.0000    1.0000    1.0000        40
          zh     1.0000    1.0000    1.0000        40

    accuracy                         0.9917       240
   macro avg     0.9921    0.9917    0.9917       240
weighted avg     0.9921    0.9917    0.9917       240

saving model
model size (Bytes): 2308413
model size (KiB): 2254.31
model size (MiB): 2.20
model size (GiB): 0.00
```

## Use the Model

In the last setp we can try out our model.

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

