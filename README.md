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
Create a conda vitual environment which is used for result evaluation
```bash
  conda create -n eval python==3.8 -y
  conda activate eval
  pip install -r requirements_eval.txt
```
## Data preparation
Downlod our selected sequential scenes from LINK1.All the base data are stored in JSON files with 0- prefix
Extract coresponding token from nuscenes dataset
```bash
python 0-extract_data_from_nuScenes.py --input 0-sequential_scenes_val.json --output 0-****_sample_val.json
python 0-extract_data_from_nuScenes.py --input 0-sequential_scenes_train.json --output 0-****_sample_train.json
```
