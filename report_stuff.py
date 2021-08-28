from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from transformers import BartTokenizer,T5Tokenizer
import wandb
from eval import Evaluator
from cli import parse_args
from data import get_dataset
from modeling import OurModel
import torch
import logging
from collections import OrderedDict
import json



logger = logging.getLogger(__name__)


def main():

    args = parse_args()

    if args.model_type == 'bart':
        num_extra_tokens = 101
        extra_tokens = [f'<extra_token_{i}>' for i in range(num_extra_tokens)]
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        tokenizer.add_tokens(extra_tokens)
        model = OurModel(args, tokenizer)
    elif args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = OurModel(args)
    else:
        raise ValueError("model_type should be one of [None,'bart','t5']")


    # print(f"model :{args.model_type}")
    # print(f"Total Params :{model.count_parameters(False)}")
    # print(f"Trainable Params :{model.count_parameters(True)}")

    train_dataset = get_dataset(args, tokenizer, evaluate=False)

    x=0




if __name__ == "__main__":
    main()
