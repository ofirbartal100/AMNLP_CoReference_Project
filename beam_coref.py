from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from transformers import BartTokenizer, BertTokenizer, T5Tokenizer, BertGenerationConfig, BertConfig
import wandb
from utils import write_meta_data
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

import pickle


logger = logging.getLogger(__name__)


def main():

    args = parse_args()

    wandb_flag = False
    if wandb_flag:
        wandb.init(project='Coref', entity='ofirbartal100', config=args, notes="t5 - identity")
        args.output_dir += "/" + wandb.run.name

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    if args.predict_file is None:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(12)

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

    if args.cont:
        model.from_pretrained(args.cont)

    if wandb_flag:
        wandb.watch(model)

    model.to(args.device)

    logger.info("Evaluation parameters %s", args)

    evaluator = Evaluator(args, tokenizer)

    sweep = {}
    beam_sizes = [2]
    for bs in beam_sizes:
        # Evaluation
        result = evaluator.evaluate_beam(
            model, prefix=f"{args.model_type}_{args.cont.split('/')[-1]}_test", beam_size=bs)
        sweep[bs] = result

    with open(os.path.join(args.output_dir, 'test_beam_sweep.json'), 'w') as f:
        f.write(json.dumps(OrderedDict(sweep)))


if __name__ == "__main__":
    main()
