from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

from transformers import BartTokenizer, BertTokenizer, T5Tokenizer, BertGenerationConfig, BertConfig
import wandb
from eval import Evaluator
from training import train, set_seed
from cli import parse_args
from data import get_dataset
from modeling import OurModel
import torch
import git
import shutil
import logging
from collections import OrderedDict
import json


logger = logging.getLogger(__name__)


def main():

    args = parse_args()

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device


    # Set seed
    set_seed(12)

    if args.model_type == 'bart' or args.model_type == 'bart-raw':
        num_extra_tokens = 101
        extra_tokens = [f'<extra_token_{i}>' for i in range(num_extra_tokens)]
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base') 
        tokenizer.add_tokens(extra_tokens)
        model = OurModel(args,tokenizer)
    elif args.model_type == 't5' or args.model_type == 't5-raw':
        tokenizer = T5Tokenizer.from_pretrained('t5-small',model_max_length=2048)   
        model = OurModel(args)    
    else:
        raise ValueError("model_type should be one of [None,'bart','t5']")

    if args.cont:
        model.from_pretrained(args.cont)


    model.to(args.device)

    logger.info("Evaluation parameters %s", args)

    evaluator = Evaluator(args, tokenizer)

    sweep = {}
    beam_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for bs in beam_sizes:
        # Evaluation
        result = evaluator.evaluate_beam(
            model, prefix=f"{args.model_type}_{args.cont.split('/')[-1]}_dev", beam_size=bs)
        sweep[bs] = result

    with open(os.path.join(args.output_dir, 'dev_beam_sweep.json'), 'w') as f:
        f.write(json.dumps(OrderedDict(sweep)))


if __name__ == "__main__":
    main()
