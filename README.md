## Rise LLM Assignment
This repository containts python script for RISE Research Enineer assignment. The main goal was to fine-tune a Transformer-based TokenClassification model for (MultiNERD Named Entity Recognition dataset)[https://huggingface.co/datasets/Babelscape/multinerd?row=17] on A) All the Tags on English, and B) A subpart of the tags.

### How to install
All the requirements are listed in `requirments.txt`. Use Python `3.10` for installation with `pip install -r requirments.txt`.

### How to Run
Using a setup with CUDA compatibility is recommeneded. Otherwise, fine-tuning the data is very slow.

On a Nvidia V100, each epoch takes around 10 minutes.

### Metrics
Following is the table for both experiment metrics. To have a comparison with the baseline paper, here we also calculate Precision, Recall, F1, and Support. You can also see the confusion matrix for both experiments.

### Summary

###