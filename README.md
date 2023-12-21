## Rise LLM Assignment
This repository contains a Python script for the RISE Research Engineer assignment. The main goal was to fine-tune a Transformer-based TokenClassification model for [MultiNERD Named Entity Recognition dataset](https://huggingface.co/datasets/Babelscape/multinerd?row=17) on A) All the Tags on English, and B) A subpart of the tags.

### How to install
All the requirements are listed in `requirments.txt`. Use Python `3.10` for installation with `pip install -r requirments.txt`.

### How to Run
#### Training
Using a setup with CUDA compatibility is recommended. Otherwise, fine-tuning the data is very slow. On a Nvidia V100, each epoch takes around 10 minutes.
Run `train.py -m <model> -e <A or B>` for running with a specific model and the experiment (A or B).

#### Evaluation
Run `train.py -m <model> -e <A or B> -t eval` for evaluation with a specific model and the experiment (A or B). This will use the folder <./<model>-<experiment>/model> for loading the latest checkpoint and running the evaluation.

### Metrics
Following is the table for both experiment metrics. To have a comparison with the baseline paper, here we also calculate Precision, Recall, F1, and Support.

### Summary
The results for all the 'B' categories are very similar. Although some of the numbers are higher than others, based on the stochasticity of the model training, there is no strong correlation to conclude here. I've used other models for this task and the results were similar. 
There are two main hypotheses for this outcome:
1. The pre-trained models have a high-quality embedding space which can fine-tune even for capacities with a small number of examples.
2. Annotating only the first token of each work (if there is more than one) helps the performance. This is especially true for categories with a small number of examples.
