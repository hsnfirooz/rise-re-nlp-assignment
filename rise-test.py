from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from typing import Callable
import evaluate
import transformers, torch
import logging
import numpy as np
import argparse
import utils
from preprocess_data import tokenize_and_align_labels, create_seq_tag_dict, filter_tags

from datasets.utils.logging import disable_progress_bar

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

argParser = argparse.ArgumentParser()

argParser.add_argument("-e", "--experiment", 
                       help="experiment tag",
                       choices=["A", "B"],
                       default="A")

argParser.add_argument("-m", "--model",
                       help="name of the model from HuggingFace Hub",
                       default="distilbert-base-cased")

argParser.add_argument("-t", "--training",
                       help="Training Or Evaluation",
                       choices=["train", "eval"],
                       default="train")

args = argParser.parse_args()

LANGUAGE = "en"
DATASET_NAME: str = "Babelscape/multinerd"

BATCH_SIZE = 32

EXPERIMENT = args.experiment

if torch.cuda.is_available():
    device = "cuda:0"
    logging.info(f"Set device to {device}.")
else:
    device = "cpu"
    logging.warning(f"Using {device} results in slow training!")

# Disable mapping tqdm for a clean output in slurm/sbatch.
disable_progress_bar()

dataset = load_dataset(DATASET_NAME)
logging.info(f"{DATASET_NAME} is loaded.")

pr_dataset = dataset.filter(lambda row: row['lang'] == LANGUAGE)
pr_dataset = pr_dataset.remove_columns(['lang'])
logging.info(f"Filtered all the data by \'{LANGUAGE}\' language.")

FILTER_TAGS_REQUIRED = False
if EXPERIMENT == "A":
    experiment_tag_list = utils.exp_a_tags
else:
    experiment_tag_list = utils.exp_b_tags
    FILTER_TAGS_REQUIRED = True

if FILTER_TAGS_REQUIRED:
    seq_tag_map = create_seq_tag_dict(experiment_tag_list)
    pr_dataset = pr_dataset.map(filter_tags, 
                                fn_kwargs={"org_tags_list": experiment_tag_list, 
                                           "tag_ids_mapping": seq_tag_map},
                                )
logging.info(f"Kept the required tags for experiment \'{EXPERIMENT}\'.")

MODEL_NAME = args.model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          add_prefix_space=True,)
logging.info(f"'{args.model}\' model has been loaded.")

OUTPUT_PATH = MODEL_NAME.split("/")[-1] + "-" + EXPERIMENT + "/model"

if tokenizer.is_fast:
     logging.info(f"Tokenizer is Fast! Great choice!")
else:
    logging.warning(f"Not a fast tokenizer! Leads to slow running.")

tokenized_pr_dataset = pr_dataset.map(tokenize_and_align_labels, 
                                      fn_kwargs={"tokenizer": tokenizer},
                                      batched=True)

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, 
                                                        num_labels=len(experiment_tag_list),
                                                        ignore_mismatched_sizes=True,
                                                        )

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
                OUTPUT_PATH,
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=2,
                weight_decay=0.005,
                torch_compile=torch.cuda.is_available(),
                )

def train(
        model: AutoModelForTokenClassification,
        args: TrainingArguments,
        dataset: DatasetDict,
        data_collator: DataCollatorForTokenClassification,
        tokenizer: AutoTokenizer,
        compute_metrics: Callable,
        device: str,
        SAVE_MODEL = True,
        TRAINING = True,

) -> None:
    trainer = Trainer(
        model.to(device),
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if TRAINING:
        trainer.train()
        logging.info(f"Training is Done!")

        metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        trainer.save_metrics("test", metrics)

        if SAVE_MODEL:
            trainer.save_model(OUTPUT_PATH)
    else:
        trainer._load_from_checkpoint(OUTPUT_PATH)
        eval_results = trainer.evaluate()
        print(eval_results)

if __name__ == "__main__":
    com_metrics = utils.prepare_compute_metrics(experiment_tag_list)

    do_train = args.training == "train"
    train(model, 
          training_args, 
          tokenized_pr_dataset, 
          data_collator, 
          tokenizer, 
          com_metrics, 
          device=device,
          TRAINING=do_train,
          )
