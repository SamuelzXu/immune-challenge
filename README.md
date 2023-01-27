### Create the environment:

```
conda create -n rapids-22.12 -c rapidsai -c conda-forge -c nvidia  \
    rapids=22.12 python=3.9 cudatoolkit=11.5
conda activate
```

### Install requirements:
```
pip install -r requirements.txt
```
Note: If there is a problem installing sklearn (cupy relies on sklearn instead of scikit-learn), just keep trying to reinstall and it'll work eventally...
### Run main
Be sure to set the dataset path in src/main.py as DATASET_PATH.
```
cd src
python main.py
```
