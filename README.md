# WiFi2Seg

## Data Preprocessing
[Data Preprocessing](datas/preprocess/README.md)

## Split Dataset
It will split data path to xxx.json, and check there are corresponding timestamp file at all receiver folder
### split dataset to train and validation
```shell=
$ python3 split_data.py --raw_data_root /root/bindingvolume/CSI_UNCC/raw --glob_path env*/[FM][12]/*/* --mode split --name "train&val" --ratio 9:1
```
The json file will be like:
```
{
    'train': [xxx.npz, xxx.npz, xxx.npz],
    'val': [xxx.npz, xxx.npz, xxx.npz]
}
```
### dataset to test.json or val.json
```shell=
$ python3 split_data.py --raw_data_root /root/bindingvolume/CSI_UNCC/raw --glob_path test_set/[FM][12]/*/* --mode whole --name "test"
```
The json file will be like:
```
{
    'data': [xxx.npz, xxx.npz]
}
```

## Training VAE

## Training WiFi2Seg
```shell=
$ python3 train.py --configs configs.yaml --mode train -v test
```