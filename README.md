## Rise LLM Assignment
This repository contains a python script for the RISE Research Engineer assignment. The main goal was to fine-tune a Transformer-based TokenClassification model for [MultiNERD Named Entity Recognition dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17) on A) All the Tags on English, and B) A subpart of the tags.

### How to install
All the requirements are listed in `requirments.txt`. Use Python `3.10` for installation with `pip install -r requirments.txt`.

### How to Run
#### Training
Using a setup with CUDA compatibility is recommended. Otherwise, fine-tuning the data is very slow. On a Nvidia V100, each epoch takes around 10 minutes.
Run `train.py -m <model> -e <A or B>` for running with a specific model and the experiment (A or B).

#### Evaluation
Run `train.py -m <model> -e <A or B> -t eval` for evaluation with a specific model and the experiment (A or B). This will use the folder <./<model>-<experiment>/model> for loading the latest checkpoint and running the evaluation.

### Metrics
Following is the table for both experiment metrics. To have a comparison with the baseline paper, here we also calculate Precision, Recall, F1, and Support. You can also see the confusion matrix for both experiments.

### Summary

###
