import wandb
import json
import os
import logging
import random
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from tqdm import tqdm, trange
from collections import OrderedDict, defaultdict

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer, evaluator, wandb_flag=False,evaluator_extended=None):
    """ Train the model """

    train_dataloader = BucketBatchSampler(
        train_dataset, max_total_seq_len=args.max_total_seq_len, batch_size_1=args.batch_size_1)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    head_params = ['coref', 'mention', 'antecedent']

    model_decay = [p for n, p in model.named_parameters() if
                   not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    model_no_decay = [p for n, p in model.named_parameters() if
                      not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in model.named_parameters() if
                  any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
    head_no_decay = [p for n, p in model.named_parameters() if
                     any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]

    head_learning_rate = args.head_learning_rate if args.head_learning_rate else args.learning_rate
    optimizer_grouped_parameters = [
        {'params': model_decay, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': model_no_decay, 'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': args.weight_decay},
        {'params': head_no_decay, 'lr': head_learning_rate, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    loaded_saved_optimizer = False
    # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #         os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    #     loaded_saved_optimizer = True

    # multi-gpu training (should be after apex fp16 initialization)
    # if not args.no_cuda:
    #     model = torch.nn.DataParallel(model)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    # if os.path.exists(args.model_name_or_path) and 'checkpoint' in args.model_name_or_path:
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         global_step = int(checkpoint_suffix)

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from global step %d", global_step)
    #         if not loaded_saved_optimizer:
    #             logger.warning("Training is continued from checkpoint, but didn't load optimizer and scheduler")
    #     except ValueError:
    #         logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(12)  # Added here for reproducibility (even between python 2 and 3)

    # If nonfreeze_params is not empty, keep all params that are
    # not in nonfreeze_params fixed.
    if args.nonfreeze_params:
        names = []
        for name, param in model.named_parameters():
            freeze = True
            for nonfreeze_p in args.nonfreeze_params.split(','):
                if nonfreeze_p in name:
                    freeze = False

            if freeze:
                param.requires_grad = False
            else:
                names.append(name)

        print('nonfreezing layers: {}'.format(names))

    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    # Added here for reproducibility
    set_seed(12)
    best_f1 = -1
    best_global_step = -1
    epoch_loss = 0.0

    results = [
        ("mention precision", 0),
        ("mention recall", 0),
        ("mention f1", 0),
        ("precision", 0),
        ("recall", 0),
        ("f1", 0),
        ("output_example", ''),
        ("output_example_decoded", [0]),
    ]
    results = OrderedDict(results)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(tensor.to(args.device) for tensor in batch)
            input_ids, input_attention_mask, gold_clusters, augmented_data, output_attention_mask = batch
            model.train()

            outputs = model(input_ids=input_ids, input_attention_mask=input_attention_mask,
                            data_augmentations=augmented_data, output_attention_mask=output_attention_mask)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # if args.n_gpu > 1 and not args.no_cuda:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

            tr_loss += loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # optimize
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # # Log metrics
                # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     logger.info(f"\nloss step {global_step}: {(tr_loss - epoch_loss) / args.logging_steps}")
                #     logging_loss = tr_loss

                # evaluate model
                # if args.do_eval and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                #     results = evaluator.evaluate(
                #         model, prefix=f'step_{global_step}', tb_writer=None, global_step=global_step)
                #     f1 = results["f1"]
                #     # save model checkpoint again
                #     # if f1 > best_f1:
                #     #     best_f1 = f1
                #     #     best_global_step = global_step
                #     #     # Save model checkpoint
                #     #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     #     if not os.path.exists(output_dir):
                #     #         os.makedirs(output_dir)
                #     #     # Take care of distributed/parallel training
                #     #     model_to_save = model.module if hasattr(model, 'module') else model
                #     #     model_to_save.save_pretrained(output_dir)
                #     #     tokenizer.save_pretrained(output_dir)

                #     #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     #     logger.info("Saving model checkpoint to %s", output_dir)

                #     #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     #     logger.info("Saving optimizer and scheduler states to %s", output_dir)
                #     logger.info(f"best f1 is {best_f1} on global step {best_global_step}")
                # save model checkpoint
                # if args.save_steps > 0 and global_step % args.save_steps == 0 and \
                #         (not args.save_if_best or (best_global_step == global_step)):
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     tokenizer.save_pretrained(output_dir)

                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)

        dev_loss = eval(args,evaluator.eval_dataset,model)
        if evaluator_extended:
            dev_loss_extended = eval(args,evaluator_extended.eval_dataset,model)


        if wandb_flag:
            wandb.log({"loss": (tr_loss - epoch_loss) / len(epoch_iterator)}, step=epoch)
            wandb.log({"dev_loss": dev_loss}, step=epoch)
            if evaluator_extended:
                wandb.log({"dev_loss_extended": dev_loss_extended}, step=epoch)

            # wandb.log({"mention recall": results["mention recall"]}, step=epoch)
            # wandb.log({"mention f1": results["mention f1"]}, step=epoch)
            # wandb.log({"mention precision": results["mention precision"]}, step=epoch)
            # wandb.log({"precision": results["precision"]}, step=epoch)
            # wandb.log({"recall": results["recall"]}, step=epoch)
            # wandb.log({"f1": results["f1"]}, step=epoch)

        epoch_loss = tr_loss

        if 0 < t_total < global_step:
            train_iterator.close()
            break

    with open(os.path.join(args.output_dir, f"best_f1.json"), "w") as f:
        json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

    return global_step, tr_loss / global_step


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def eval(args, train_dataset, model):
    """ Train the model """

    train_dataloader = BucketBatchSampler(
        train_dataset, max_total_seq_len=args.max_total_seq_len, batch_size_1=True)

    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(12)  # Added here for reproducibility (even between python 2 and 3)

    
    # Added here for reproducibility
    set_seed(12)

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(tensor.to(args.device) for tensor in batch[1])
            input_ids, input_attention_mask, gold_clusters, augmented_data, output_attention_mask = batch
            model.eval()

            outputs = model(input_ids=input_ids, input_attention_mask=input_attention_mask,
                            data_augmentations=augmented_data, output_attention_mask=output_attention_mask)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            tr_loss += loss.item()


    return  tr_loss / len(train_dataloader)
