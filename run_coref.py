from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
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


logger = logging.getLogger(__name__)


def main():

    args = parse_args()

    args.train_file = '/home/ofirbartal/Projects/Coref/data/train.english.jsonlines'
    args.predict_file = '/home/ofirbartal/Projects/Coref/data/dev.english.jsonlines'
    args.gradient_accumulation_steps = 1
    args.max_seq_length = 800
    args.max_total_seq_len = 1500

    if 'bart' in args.model_type:
        args.gpu_id = 2
        args.train_file_cache = '/home/ofirbartal/Projects/Coref/data/train.bart.extended.pkl'
        args.predict_file_cache = '/home/ofirbartal/Projects/Coref/data/dev.bart.full.pkl'
        args.predict_file_cache_extended = '/home/ofirbartal/Projects/Coref/data/dev.bart.extended.pkl'
        args.warmup_steps = 4500
        args.adam_epsilon = 8e-07
        args.head_learning_rate = 0.00031
        args.learning_rate = 3.1e-05
        args.adam_beta2 = 0.93
        args.weight_decay = 0.007
        args.dropout_prob = 0.15
        args.adam_beta1 = 0.820

    elif 't5' in args.model_type:
        args.gpu_id = 0
        args.train_file_cache = '/home/ofirbartal/Projects/Coref/data/train.t5.extended.pkl'
        # args.train_file_cache = '/home/ofirbartal/Projects/Coref/data/train.t5.pkl'
        args.predict_file_cache = '/home/ofirbartal/Projects/Coref/data/dev.t5.full.pkl'
        args.predict_file_cache_extended = '/home/ofirbartal/Projects/Coref/data/dev.t5.extended.pkl'
        args.warmup_steps = 35000
        args.adam_epsilon = 0.0000018
        args.head_learning_rate = 3e-4
        args.learning_rate = 0.000458
        args.adam_beta2 = 0.62
        args.weight_decay = 0.02
        args.dropout_prob = 0.1
        args.adam_beta1 = 0.61

    wandb_flag = True
    if wandb_flag:
        wandb.init(project='Coref', entity='ofirbartal100', config=args, notes="".format(args.model_type))
        args.output_dir += "/" + wandb.run.name

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    if args.predict_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if not args.cont:
        if args.output_dir and os.path.exists(args.output_dir) and \
                os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir))

        if args.overwrite_output_dir and os.path.isdir(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    try:
        write_meta_data(args.output_dir, args)
    except git.exc.InvalidGitRepositoryError:
        logger.info("didn't save metadata - No git repo!")

    # Set seed
    set_seed(12)

    if args.model_type == 'bert_generation':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = OurModel(args)
    elif args.model_type == 'bart' or args.model_type == 'bart-raw':
        num_extra_tokens = 101
        extra_tokens = [f'<extra_token_{i}>' for i in range(num_extra_tokens)]
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_tokens(extra_tokens)
        model = OurModel(args, tokenizer)
    elif args.model_type == 't5' or args.model_type == 't5-raw':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = OurModel(args)
    else:
        raise ValueError("model_type should be one of [None,'bart','t5']")

    if args.cont:
        model.model.from_pretrained(args.output_dir+'/'+args.cont)

    if wandb_flag:
        wandb.watch(model)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    evaluator = Evaluator(args, tokenizer)

    args.predict_file_cache = args.predict_file_cache_extended
    evaluator_extended = Evaluator(args, tokenizer)
    # evaluator_extended =None

    
    # Training
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, evaluator, wandb_flag=wandb_flag,evaluator_extended=evaluator_extended)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    # results = {}

    # if args.do_eval :
    #     result = evaluator.evaluate(model, prefix="final_evaluation", official=True)
    #     results.update(result)
    #     return results

    # return results


if __name__ == "__main__":
    main()
