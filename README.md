# PredictionIO Boston House Prices Template

## Requirement

* Spark 2.x

## Getting Started

### Create App on PIO

Create BHPApp application on PredictionIO:

```
pio app new --access-key BHP_TOKEN BHPApp
```

### Import Data

Import data from [boston\_house\_prices.csv]("https://raw.githubusercontent.com/scikit-learn/scikit-learn/0.19.0/sklearn/datasets/data/boston_house_prices.csv).

```
python data/import_eventserver.py
```

### Build Template

Build this template:

```
pio build
```

### Run Jupyter

Launch Jupyter notebook and open eda.ipynb. (or you can create a new notebook to analyze data)

```
PYSPARK_PYTHON=$PYENV_ROOT/shims/python PYSPARK_DRIVER_PYTHON=$PYENV_ROOT/shims/jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook" pio-shell --with-pyspark
```

### Run Train on Spark

Download Python code from eda.ipynb and put it to train.py.
To execute it on Spark, run `pio train` with --main-py-file option.

```
pio train --main-py-file train.py
```

### Deploy App

Run PredictionIO API server:

```
pio deploy
```

### Predict

Check predictions from deployed model:

```
curl -s -H "Content-Type: application/json" -d '{"attr0":5.1,"attr1":3.5,"attr2":1.4,"attr3":0.2}' http://localhost:8000/queries.json
```

