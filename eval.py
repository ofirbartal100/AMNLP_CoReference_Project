import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from metrics import CorefEvaluator, MentionEvaluator
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from conll import evaluate_conll
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.tokenizer = tokenizer
        self.eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)
        self.eval_dataloader = BucketBatchSampler(self.eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)


    def evaluate(self, model, prefix="", tb_writer=None, global_step=None, official=False):

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) :
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(self.eval_dataset))
        model.eval()

        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        # losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        outputs = torch.zeros(1,5)
        for (doc_key, subtoken_maps), batch in tqdm(self.eval_dataloader):
            if prefix!= "final_evaluation" and random.random() > 1/5: # reduce eval to 10%
                continue

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids,input_attention_mask, gold_clusters ,data_augmentations, output_attention_mask = batch

            with torch.no_grad():
                # outputs = data_augmentations
                end  = int((input_ids==1).max(1).indices[0])
                outputs = model.generate(input_ids,input_attention_mask,data_augmentations, output_attention_mask, num_beams=2, early_stopping=True,min_length=end,max_length=2*end)#, bos_token_id=101, eos_token_id=102,pad_token_id=0
                
            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = (outputs.cpu().numpy(),)
            for output in zip(*(batch_np + outputs_np)):

                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())


                predicted_clusters = self.eval_dataset.reverse_augmentation(output[5]) # need our clusters
                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps

            
            if prefix!= "final_evaluation" :
                break
            

        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()
        # prec, rec, f1 = 0,len(input_ids[0])-len(outputs[0]),((input_ids[0,:min(len(input_ids),len(outputs[0]))] - outputs[0,:min(len(input_ids),len(outputs[0]))])**2).float().mean().item()

        # prec, rec, f1 = coref_evaluator.get_prf_all() # need for final models only.. not for training

        # results = [(key, sum(val) / len(val)) for key, val in losses.items()]
        results = [
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1),
            ("output_example",str(outputs[0])),
            # ("output_example_decoded",eval_dataset.print_augmentation(outputs[0].cpu().numpy())),
            ("output_example_decoded",self.eval_dataset.tokenizer.decode(outputs[0])),
            ("input_example_decoded",self.eval_dataset.tokenizer.decode(input_ids[0])),
        ]
        logger.info("***** Eval results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

        if self.eval_output_dir:
            output_eval_file = os.path.join(self.eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key} = {values:.3f}\n")
                    else:
                        writer.write(f"{key} = {values}\n")

        results = OrderedDict(results)
        results["experiment_name"] = self.args.model_type
        results["data"] = prefix
        with open(os.path.join(self.args.output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(results) + '\n')

        if official:
            with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
                f.write(json.dumps(doc_to_prediction) + '\n')
                f.write(json.dumps(doc_to_subtoken_map) + '\n')

            # if self.args.conll_path_for_eval is not None:
            #     conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
            #     official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            #     logger.info('Official avg F1: %.4f' % official_f1)

        return results


    def evaluate_beam(self, model, prefix="",  beam_size=2):

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) :
            os.makedirs(self.eval_output_dir)


        # Eval!
        logger.info("***** Running evaluation {} - beam size {} *****".format(prefix,beam_size))
        logger.info("  Examples number: %d", len(self.eval_dataset))
        model.eval()

        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        for (doc_key, subtoken_maps), batch in tqdm(self.eval_dataloader):

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids,input_attention_mask, gold_clusters ,data_augmentations, output_attention_mask = batch

            with torch.no_grad():
                end  = int((input_ids==1).max(1).indices[0])
                logger.info(f'input_size: {end} , shape: {input_ids.shape}')

                outputs = model.generate(input_ids,input_attention_mask,data_augmentations, output_attention_mask, num_beams=beam_size, early_stopping=True,min_length=end,max_length=2*end)#, bos_token_id=101, eos_token_id=102,pad_token_id=0
                
            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = (outputs.cpu().numpy(),)
            for output in zip(*(batch_np + outputs_np)):

                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())


                predicted_clusters = self.eval_dataset.reverse_augmentation(output[5]) # need our clusters
                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps


        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()
        # prec, rec, f1 = coref_evaluator.get_prf_all() # need for final models only.. not for training

        results = [
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1),
        ]

        logger.info("***** Eval results {} - beam size {}*****".format(prefix,beam_size))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")

        return OrderedDict(results)
