from transformers import AutoTokenizer

def tokenize_and_align_labels(row, 
                              tokenizer: AutoTokenizer,
                              label_all_tokens=False,
                              ) -> dict:
    
    tokenized_inputs = tokenizer(row['tokens'], 
                                 truncation=True, 
                                 is_split_into_words=True,
    )
    """
    As the tokenizer is trained on a different dataset, all the token IDs in our dataset
    needs to be aligend with the original tokenizer.

    This functiuon is borrowed from HuggungFace Tutorial.
    https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb

    Args:
        row (dict): a dictionary with the keys `tokens` and `ner_tags` for each row.
        tokenizer (AutoTokenizer): tokenizer to use in for aliging the labels.
        label_all_tokens (bool, optional): If set to True, token labels inherit the same label of their original word.
            If set to false, these sub-words are given a `-100` label instead to get ignored in the loss calculation.
            Defaults to False.

    Returns:
        tokenized_inputs (dict): A dictionary with tokenized inputs.
    """
    labels = []
    for i, label in enumerate(row['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Replace special token id with -100 to be ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_seq_tag_dict(tag_list: dict) -> dict:
    """
    This function is creating a mapping for changing the IDs to be sequentional. Non-sequentianl
        IDs cause the training to rise 'out-of-band' error.
    Args:
        tag_list (dict): a dictionary with label and corresponding id.

    Returns:
        seq_tag_ids (dict): a dictionary which maps unsequntional ids to sequentinal ids.
    """
    seq_tag_ids = {}
    for idx, value in enumerate(tag_list.values()):
        seq_tag_ids[value] = idx
    return seq_tag_ids

def filter_tags(row, 
                org_tags_list: dict,
                tag_ids_mapping: dict):
    """
    This function if filtering the un-used tags and replace 0 as the tag value for other classes.

    Args:
        row (dict): a dictionary with the keys `tokens` and `ner_tags` for each row.
        org_tags_list (dict): a dictionary with label and corresponding id.
        tag_ids_mapping (dict): a dictionary which maps unsequntional ids to sequentinal ids.

    Returns:
        row (dict): a processed dictionary with the keys `tokens` and `ner_tags` for each row.
    """
    tags_to_keep = list(org_tags_list.values())
    row['ner_tags'] = [0 if tag not in tags_to_keep else tag_ids_mapping[tag] for tag in row['ner_tags']]
    return row