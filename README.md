# KEPT
KEPT: Knowledge-Enhanced Trajectory Prediction of Consecutive Driving Frames with Vision-Language Models
## Environment Setup
### Embedding and Retrieving Environment
Create a conda virtual environment which is used for model inference
```bash
conda create -n rag python==3.11 -y
conda activate rag
pip install -r requirements_rag.txt
```
### Inference Environment
Create a conda virtual environment which is used for model inference
```bash
conda create -n inference python==3.11 -y
conda activate inference
pip install -r requirements_inference.txt
```
### Evaluation Environment
Create a conda virtual environment which is used for nuScenes dataset processing and results evaluation
```bash
conda create -n eval python==3.8 -y
conda activate eval
pip install -r requirements_eval.txt
```
## Data preparation
### Data collection
Downlod our selected [sequential scenes](https://huggingface.co/datasets/larswang/kept_datasets/tree/main) and [nuScenes dataset](https://www.nuscenes.org/nuscenes). The base data in this project are stored in `JSON` files with `0-` prefix.

Extract coresponding token of images from nuscenes dataset.
```bash
python 0-extract_data_from_nuScenes.py --sample /data/nuscenes_dataset/v1.0-trainval/sample_data.json --input 0-sequential_scenes_val.json --output 0-sequential_scenes_sample_val.json
python 0-extract_data_from_nuScenes.py --sample /data/nuscenes_dataset/v1.0-trainval/sample_data.json --input 0-sequential_scenes_train.json --output 0-sequential_scenes_sample_train.json
```
### Data aligned
Find ego status in nuScenes dataset and merge all selected data in one `JSON` file with `1-` prefix.
```bash
python 0_1-merge_scenes&data.py --sample 0-sequential_scenes_sample_val.json --scenes 0-sequential_scenes_val.json --out 1-aligned_scenes_data_val.json
python 0_1-merge_scenes&data.py --sample 0-sequential_scenes_sample_train.json --scenes 0-sequential_scenes_train.json --out 1-aligned_scenes_data_train.json  
```
### Data format convert
Convert data format for next step lora finetuning and stored in `JSON` files with `2-` prefix
```bash
python 1_2-train_lora_data_format_converter.py --input 1-aligned_scenes_data_val.json --output 2-sequential_pretrain_data_with_status.json --status
python 1_2-train_lora_data_format_converter.py --input 1-aligned_scenes_data_val.json --output 2-sequential_pretrain_data_without_status.json
```
## Encoder Training
```bash
cd vfsf_training/
python train_contrastive_hard_negative.py
```
We also provide different editions of the encoder, namely the frequency-only `vfsf_frequency_only.py` and the spatial-only `vfsf_spatial_only`. The corresponding training codes are `train_contrastive_hard_negative_frequency.py` and `train_contrastive_hard_negative_spatial.py`.

## Embedding and Retrieving
### Embedding
Prepare the nuScenes dataset and align the data paths.
```bash
cd embedding/
python embedding_database.py
```
### Retrieving
```bash
cd retrieving/
python retrieving_hnsw.py
```
This step allows to get the Top-K retrieval results for the following inference.
During the embedding and retrieving pipelines, if the vision encoder is changed, it will be necessary to copy the correct edition of the encoder under the corresponding repository.


## Model finetuning
We use [LLamafactory](https://github.com/hiyouga/LLaMA-Factory) to fintune [Qwen2-vl-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

The first step lora dataset contains 5 `JSON` files named in the format of `spatial-.json`

The second step lora dataset contains 1 `JSON` file named `track_pretrain.json`

The third step lora dataset contains `2-sequential_pretrain_data_without_status.json` or `2-sequential_pretrain_data_with_status.json`, using for different inference condition.
## Model inference
```bash
conda activate inference
```
Using different commands to run model inference in different conditions. All the prediction results are stored in `JSON` files with `4-` prefix.
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
## Evaluate prediction results
```bash
conda activate eval
```
### Add valid results
Replace `--pred` with `4-` prefix files got in last step. Eval_avalible `JSON` is with `5-` prefix.
```
python eval/add_valid_res_in_json.py --basedata /path/to/1-aligned_scenes_data_val.json --pred path/to/4-**output**.json --output 5-**eval**.json
```
### Caculation
```bash
python eval_collision.py --input 5-**eval**.json --ignore-z
python eval_L2.py --input 5-**eval**.json
```
