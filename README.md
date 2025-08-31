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
Create a conda vitual environment which is used for nuScenes dataset processing and results evaluation
```bash
conda create -n eval python==3.8 -y
conda activate eval
pip install -r requirements_eval.txt
```
## Data preparation
### Data collection
Downlod our selected sequential scenes from (LINK1).And nuScenes dataset from (LINK2).the base data in this project are stored in JSON files with `0-` prefix.
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
```
### Data format convert
Convert data format for next step lora finetuning
```bash
python 1_2-train_lora_data_format_converter.py --input 1-aligned_scenes_data_val.json --output 2-sequential_pretrain_data_with_status.json --status
python 1_2-train_lora_data_format_converter.py --input 1-aligned_scenes_data_val.json --output 2-sequential_pretrain_data_without_status.json
```
## Model finetuning
We use `LLamafactory`(LINK3) to fintune `Qwen2-vl-2B`(LINK4).

The first step lora uses 5 JSON files in `training/spatial-.json`

The second step lora uses 1 JSON file in `training/track_pretrain.json`

The third step lora uses `2-sequential_pretrain_data_without_status.json` and `2-sequential_pretrain_data_without_status.json`
## Model inference
``bash
conda activate inference
```
Using different commands to run model inference in different conditions.
```bash
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top1.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 1 --out /path/to/4-**output**.json
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top2.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 2 --out /path/to/4-**output**.json
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top3.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 3 --out /path/to/4-**output**.json
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top4.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 4 --out /path/to/4-**output**.json
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top1.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 1 --out /path/to/4-**output**.json --withstatus
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top2.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 2 --out /path/to/4-**output**.json --withstatus
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top3.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 3 --out /path/to/4-**output**.json --withstatus
python 3_4-KEPT_inference.py --model_dir /path/to/model --retrieval path/to/3-retrieval_results_top4.json --val /path/to/1-aligned_scenes_data_val.json --db /path/to/1-aligned_scenes_data_train.json --topk 4 --out /path/to/4-**output**.json --withstatus
```
