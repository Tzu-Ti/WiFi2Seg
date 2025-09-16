# CSI Preprocessing: register with RGB data
## Check CSI collecting enough data
- for multi envs dataset, example:
```shell=
$ python3 check_csi.py --dataset_root ~/bindingvolume/CSI_dataset_multi_envs/raw --glob_path empty/M1/[ABCDE]
```
- for UNCC dataset, example:
```shell=
$ python3 check_csi.py --dataset_root ~/bindingvolume/CSI_UNCC/raw/ --glob_path env0/F1/*
```

## Parsing CSI
- for multi envs dataset, example:
```shell=
$ python3 parsing_csi.py --dataset_root ~/bindingvolume/CSI_dataset_multi_envs/raw --glob_path empty/M1/[ABCDE] --csi_subcarrier 2025 --csi_length 25 --num_workers 8 
```
- for UNCC dataset, example:
```shell=
$ python3 parsing_csi.py --dataset_root ~/bindingvolume/CSI_UNCC/raw --glob_path */[FM][12]/* --csi_subcarrier 2025 --csi_length 25 --num_workers 8
```