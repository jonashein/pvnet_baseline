# Towards Markerless Surgical Tool and Hand Pose Estimation: PVNet Baseline

- [Project page](http://medicalaugmentedreality.org/handobject.html) <!-- - [Paper](http://arxiv.org/abs/2004.13449) -->
- [Synthetic Grasp Generation](https://github.com/jonashein/grasp_generator)
- [Synthetic Grasp Rendering](https://github.com/jonashein/grasp_renderer)
- [Real Dataset Generation](https://github.com/jonashein/handobject_dataset_creator)
- [HandObjectNet Baseline](https://github.com/jonashein/handobjectnet_baseline)
- [PVNet Baseline](https://github.com/jonashein/pvnet_baseline)
- [Combined Model Baseline](https://github.com/jonashein/baseline_combination)

The structure of this project is described in [project_structure.md](project_structure.md).

## Table of Content

- [Setup](#setup)
- [Demo](#demo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Citations](#citations)

## Setup

### Download and install prerequisites
```sh
sudo apt-get install libglfw3-dev libglfw3
```

### Download and Install Code
```sh
git https://github.com/jonashein/pvnet_baseline.git
cd pvnet_baseline
conda env create --file=environment.yml
conda activate pvnet
```

Compile cuda extension for RANSAC voting under `lib/csrc/ransac_voting`:
```sh
cd lib/csrc/ransac_voting/
python setup.py build_ext --inplace
cd ../../../
```

### Download Synthetic Dataset
Download the synthetic dataset from the [project page](http://medicalaugmentedreality.org/handobject.html), 
or use the commands below:
```sh
cd data/
wget http://medicalaugmentedreality.org/datasets/syn_colibri_v1.zip
unzip -x syn_colibri_v1.zip
cd ../
```

Convert the dataset into the format expected by PVNet:
```sh
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/syn_colibri_v1/train.txt -o data/ -n syn_colibri_v1_train
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/syn_colibri_v1/val.txt -o data/ -n syn_colibri_v1_val
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/syn_colibri_v1/test.txt -o data/ -n syn_colibri_v1_test
```

### Download Real Dataset
Download the real dataset from the [project page](http://medicalaugmentedreality.org/handobject.html), 
or use the commands below:
```sh
cd data/
wget http://medicalaugmentedreality.org/datasets/real_colibri_v1.zip
unzip -x real_colibri_v1.zip
cd ../
```

Convert the dataset into the format expected by PVNet:
```sh
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/real_colibri_v1/train.txt -o data/ -n real_colibri_v1_train
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/real_colibri_v1/val.txt -o data/ -n real_colibri_v1_val
python3 pvnet_custom_dataset.py -m assets/drill_segmentation_textured_final.ply -d data/real_colibri_v1/test.txt -o data/ -n real_colibri_v1_test
```

## Demo

We provide pretrained models, which can be downloaded [here](https://drive.google.com/file/d/1shh_A7BhNTR2MirjH4PKSMU5GFinx66P/view?usp=sharing).
Unzip the downloaded archive, and copy the `model` directory of one of the checkpoints to the `data` directory.

Your directory structure should look like this:
```
pvnet_baseline/data/model/pvnet/custom/*.pth
```

Then run the commands listed [below](#evaluation) to evaluate the model.  

## Training

Pretrain a model on the synthetic dataset:
```sh
python train_net.py --cfg_file configs/syn_colibri_v1_train.yaml
```

Refine a model on the real dataset:
```sh
python train_net.py --cfg_file configs/real_colibri_v1_train.yaml
```
The training checkpoints and monitoring data will be stored at `data/model/` and `data/record/` respectively.


Losses and validation metrics can are monitored on tensorboard:
```sh
tensorboard --logdir data/record/pvnet
```

## Evaluation

Evaluate a pretrained model on the synthetic dataset:
```sh
python train_net.py --test --cfg_file configs/syn_colibri_v1_test.yaml
```

Evaluate a refined model on the real dataset:
```sh
python train_net.py --test --cfg_file configs/real_colibri_v1_test.yaml
```

After evaluating a model, the test set metrics can be computed by running:
```sh
python3 compute_metrics.py -m "data/record/metrics.pkl"
```

## Visualization

To visualize the keypoint estimates and render 3D views of the tool pose estimates, run:
```sh
python run.py --type visualize --test --cfg_file configs/real_colibri_v1_test.yaml --vis_out visualizations/
```

## Citations

If you find this code useful for your research, please consider citing:

* the publication that this code was adapted for
```
COMING SOON.
```

* the publication it builds upon and that this code was originally developed for
```
@inproceedings{peng2019pvnet,
  title={PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation},
  author={Peng, Sida and Liu, Yuan and Huang, Qixing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2019}
}
```
