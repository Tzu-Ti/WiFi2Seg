# Generate Human Skeleton Keypoints
## Install OpenPose
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
## Generate skeleton by using openpose
```shell=
$ python3 generate_kp.py --model_folder openpose/models/ --dataset_root /root/bindingvolume/CSI_UNCC/raw --glob_path */[FM][12]/*
```