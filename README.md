### Create the environment:

```
conda create -n rapids-22.12 -c rapidsai -c conda-forge -c nvidia  \
    rapids=22.12 python=3.9 cudatoolkit=11.5
```

### Install requirements:
```
pip install -r requirements.txt
```

### Run main
Be sure to set the dataset path in src/main.py as DATASET_PATH.
```
cd src
python main.py
```
