# KEPT
KEPT: Knowledge-Enhanced Trajectory Prediction of Consecutive Driving Frames with Vision-Language Models
## Environment Setup
### Inference Environment
Create a conda vitual environment which is used for model inference
```bash
conda create -n inference python==3.11 -y
conda activate inference
pip install -r requirements_inference.txt
```
### Evaluation Environment
Create a conda vitual environment which is used for nuScenes dataset using and results evaluation
```bash
conda create -n eval python==3.8 -y
conda activate eval
pip install -r requirements_eval.txt
```
## Data preparation
### Data collection
Downlod our selected sequential scenes from (LINK1).And nuScenes dataset from (LINK2).the base data in this project are stored in JSON files with 0- prefix.
Extract coresponding token of images from nuscenes dataset.
```bash
python 0-extract_data_from_nuScenes.py --sample /data/nuscenes_dataset/v1.0-trainval/sample_data.json --input 0-sequential_scenes_val.json --output 0-sequential_scenes_sample_val.json
python 0-extract_data_from_nuScenes.py --sample /data/nuscenes_dataset/v1.0-trainval/sample_data.json --input 0-sequential_scenes_train.json --output 0-sequential_scenes_sample_train.json
```
### Data aligned
Find ego status in nuScenes dataset and merge all selected data in one JSON files.
```bash
python 0_1-merge_scenes&data.py --sample 0-sequential_scenes_sample_val.json --scenes 0-sequential_scenes_val.json --out 1-aligned_scenes_data_val.json
python 0_1-merge_scenes&data.py --sample 0-sequential_scenes_sample_train.json --scenes 0-sequential_scenes_train.json --out 1-aligned_scenes_data_train.json  
