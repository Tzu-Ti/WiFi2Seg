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

## Data Preprocessing
### Registration with CSI and RGB
[CSI Preprocessing](csi/README.md)

### Generate Segmentation
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation

### Generate skeleton keypoints
[Generate Keypoints](skeleton/README.md)