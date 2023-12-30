# KeyGNet: Learning Better Keypoints for Multi-Object 6DoF Pose Estimation
![teaser](./doc/keygnet.gif "keypoints optimized by KeyGNet")
> [Learning Better Keypoints for Multi-Object 6DoF Pose Estimation](https://arxiv.org/abs/2308.07827 "arxiv")
> Yangzheng Wu, Michael Greenspan
> WACV 2024
## Preliminary
### Dependencies
Install PyTorch prior to other python packages:
Install the rest dependencies by using conda:
```
conda env create -f environment.yml
conda activate keygnet
```
### Datasets
Download BOP core datasets from [BOP](https://bop.felk.cvut.cz/datasets/) website.

## Pre-trained keypoints for testing
Download the pre-trained keypoints from 'keypoints' folder and unzip it into the root of each dataset.

## Training
### Train KeyGNet
```
python train.py --dataset dname --batch_size 32 --keypointsNo 3 --lr 1e-3 --data_root "PathToData" --ckpt_root "PathToLogs"
```

## Testing
Test with the keyGNet keypoints by using [RCVPose](https://github.com/aaronWool/rcvpose), [PVNet](https://github.com/zju3dv/clean-pvnet), and [PVN3D](https://github.com/ethnhe/PVN3D) github repositories.

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{wu2024keygnet,
  title={Learning Better Keypoints for Multi-Object 6DoF Pose Estimation},
  author={Wu, Yangzheng and Greenspan, Michael},
  booktitle={WACV},
  year={2024}
}
```
