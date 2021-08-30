# 3D Object Detection based on 3D LiDAR Point Clouds

## Demonstration (on a single GTX 2070 Super)

## 2. Getting Started
### 2.1. Requirement

```shell script
clone the repo
cd 3D-Object-Detection/
pip install -r requirements.txt
```

### 2.2. Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Velodyne point clouds _**(29 GB)**_
- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_ (For visualization purpose only)


Please make sure that you construct the source code & dataset directories structure as below.

### 2.3. How to run

#### 2.3.1. Visualize the dataset 

To visualize 3D point clouds with 3D boxes, let's execute:

```shell script
cd scripts/data_process/
python kitti_dataset.py
```


#### 2.3.2. Inference

The pre-trained model was pushed to this repo.

```
python test.py --gpu_idx 0 --peak_thresh 0.2
```

#### 2.3.3. Making demonstration

```
python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.2
python demo_front.py --gpu_idx 0 --peak_thresh 0.2
```

The kitti data for the demonstration can be downloaded from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip).


#### 2.3.4. Training

##### 2.3.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0
```

#### Tensorboard

- To track the training progress, go to the `logs/` folder and 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

- Then go to [http://localhost:6006/](http://localhost:6006/)

## References

[1] CenterNet: [Paper](https://arxiv.org/abs/1904.07850), [Code](https://github.com/xingyizhou/CenterNet) <br>
[2] RTM3D: [Paper](https://arxiv.org/abs/2001.03343) [Code](https://github.com/Banconxuan/RTM3D) <br>
[3] Complex-YOLO: [Paper](https://arxiv.org/pdf/1803.06199v2.pdf) [Code](https://github.com/ghimiredhikura/Complex-YOLOv3)

*3D LiDAR Point pre-processing:* <br>
[4] VoxelNet: [PyTorch Implementation](https://github.com/skyhehe123/VoxelNet-pytorch)

*Lidar-Image Projection
[5] [Github repo](https://github.com/darylclimb/cvml_project/tree/master/projections/lidar_camera_projection)
[6] [Github repo](https://github.com/navoshta/KITTI-Dataset)

## Folder structure

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── scripts/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```
