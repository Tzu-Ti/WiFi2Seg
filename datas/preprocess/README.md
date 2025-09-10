# Data Preprocessing
## Data directory format
```shell=!
CSI_dataset_multi_envs
|-- parsed
|
`-- raw
    |-- csi0
        |-- empty
            |-- M1
                |-- A
                    |-- timestamp
                        |-- timestamp.csi
        |-- env0
        |-- env1
        .
        .
        .
        |-- envxx
    |-- csi1|  
    .
    .
    .  
    |-- rgb|
        |-- empty
            |-- M1
                |-- A
                    |-- timestamp
                        |-- timestamp.jpg
        |-- env0
        |-- env1
        .
        .
        .
        |-- envxx
```

## Registration with CSI and RGB
### Check CSI collecting enough data
- for multi envs dataset, example:
```shell=
$ python3 check_csi.py --dataset_root ~/bindingvolume/CSI_dataset_multi_envs/raw --glob_path empty/M1/[ABCDE]
```
- for UNCC dataset, example:
```shell=
$ python3 check_csi.py --dataset_root ~/bindingvolume/CSI_UNCC/raw/ --glob_path env0/F1/*
```

### Parsing CSI
- for multi envs dataset, example:
```shell=
$ python3 parsing_csi.py --dataset_root ~/bindingvolume/CSI_dataset_multi_envs/raw --glob_path empty/M1/[ABCDE] --csi_subcarrier 2025 --csi_length 25 --num_workers 8 
```
- for UNCC dataset, example:
```shell=
$ python3 parsing_csi.py --dataset_root ~/bindingvolume/CSI_UNCC/raw --glob_path */[FM][12]/* --csi_subcarrier 2025 --csi_length 25 --num_workers 8
```

## Generate Segmentation
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation

## Generate skeleton keypoints
### Install OpenPose
- install cmake
```shell=
$ apt install cmake
$ apt install libprotobuf-dev protobuf-compiler
$ apt install libgoogle-glog-dev libopencv-dev
$ apt install libboost-all-dev
$ apt install libhdf5-dev
$ apt install libatlas-base-dev
```

- clone openpose
```shell=
$ git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
$ cd openpose/
$ git submodule update --init --recursive --remote
```

- build
```shell=
$ mkdir build && cd build
$ cmake .. -DBUILD_PYTHON=ON -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF
$ make -j `nproc`
```

- move .so to site-package
```shell=
$ cp build/python/openpose/pyopenpose.cpython-310-x86_64-linux-gnu.so $(virtualenv)/lib/python3.10/site-packages
```
### Generate skeleton by using openpose
```shell=
$ python3 generate_kp.py --model_folder openpose/models/ --dataset_root /root/bindingvolume/CSI_UNCC/raw --glob_path */[FM][12]/*
```