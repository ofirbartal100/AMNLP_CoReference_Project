from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6"

import logging
import os
import torch

from transformers import BartTokenizer,BertTokenizer,T5Tokenizer

from modeling import OurModel
from data import get_dataset
from cli import parse_args
from training import train, set_seed,eval

logger = logging.getLogger(__name__)


def main():
    
    args = parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.device = device

    # Set seed
    set_seed(12)


    if args.model_type == 'bert_generation':
        tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
        model = OurModel(args)
    elif args.model_type == 'bart' or args.model_type == 'bart-raw':
        num_extra_tokens = 101
        extra_tokens = [f'<extra_token_{i}>' for i in range(num_extra_tokens)]
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base') 
        tokenizer.add_tokens(extra_tokens)
        model = OurModel(args,tokenizer)
    elif args.model_type == 't5' or args.model_type == 't5-raw':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')   
        model = OurModel(args)    
    else:
        raise ValueError("model_type should be one of [None,'bart','t5']")
        

    if args.cont:
        model.from_pretrained(args.output_dir+'/'+args.cont)


    model.to(args.device)

    # Eval
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=True)

        tr_loss = eval(args, train_dataset, model)
        print(tr_loss)
        


if __name__ == "__main__":
    main()
