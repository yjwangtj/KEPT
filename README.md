# KEPT
KEPT: Knowledge-Enhanced Trajectory Prediction of Consecutive Driving Frames with Vision-Language Models
# Inference Environment
conda create -n inference python==3.11 -y
conda activate inference
pip install -r requirements_inference.txt
# Evaluation Environment
conda create -n eval python==3.8 -y
conda activate eval
pip install -r requirements_eval.txt
