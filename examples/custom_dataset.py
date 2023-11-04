# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_custom_dataset(dataset_config, tokenizer, split: str):
    dataset = datasets.load_dataset("Burakblm54/dialog-29k", split=split)

    prompt = (
        f"{{dialog}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(dialog=sample["text"]),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["text"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            "labels": prompt,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
