from transformers.tokenization_bart import BartTokenizer
from utils import extract_clusters2, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from metrics import CorefEvaluator, MentionEvaluator
from torch.utils.data import Dataset
from utils import flatten_list_of_lists
import json
import logging
import os
import pickle
from collections import namedtuple
from tqdm import tqdm
import torch
import random

# from consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF

NULL_ID_FOR_COREF = 0
MENTION_START, MENTION_END = '<extra_id_0>', '<extra_id_1>'

CorefExample = namedtuple(
    "CorefExample", ["token_ids", "clusters", "augmented_labels"])

logger = logging.getLogger(__name__)


class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=-1):
        self.tokenizer = tokenizer
        logger.info(f"Reading dataset from {file_path}")
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(
            file_path)
        self.max_seq_length = max_seq_length
        self.examples, self.lengths, self.num_examples_filtered = self._tokenize(
            examples)
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length.")


    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(
                    flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(
                    len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(
                    max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _augment_data(self, tokens, clusters):
        # start_mention = 5
        # end_mention = 6
        # not_mention_token = 3
        # or_token = 4

        if isinstance(self.tokenizer ,BartTokenizer):
            start_mention = 50265
        else:            
            start_mention = 32000

        end_mention = start_mention+1
        or_token = start_mention+2

        target = [[] for t in tokens]
        sorted_clusters = [ sorted(c,  key=lambda cc: cc[0]) for c in clusters]
        sorted_clusters = sorted(sorted_clusters, key=lambda c: c[0][0])
        for idx, cluster in enumerate(sorted_clusters):
            for m in cluster:
                if(m[0] == m[1]):
                    target[m[0]].append(('se', idx+start_mention+3))
                    # target[m[0]].append(('se', idx+7))
                else:
                    target[m[0]].append(('s', idx+start_mention+3))
                    # target[m[0]].append(('s', idx+7))
                    target[m[1]].append(('e', idx+start_mention+3))
                    # target[m[1]].append(('e', idx+7))

        i = 0
        j = 0
        augmented_data = []
        for i in range(len(target)):
            
            for mi, m in enumerate(target[i]):
                if mi > 0:
                    augmented_data.append(or_token)
                if 'e' in m:
                    augmented_data.append(m[1])
                    augmented_data.append(end_mention)

            for mi, m in enumerate(target[i]):
                if mi > 0:
                    augmented_data.append(or_token)
                if 's' in m:
                    augmented_data.append(start_mention)
                    augmented_data.append(m[1])
                

            augmented_data.append(tokens[i])


        # self.print_augmentation(augmented_data)
        # self.reverse_augmentation(augmented_data)

        f1 = self.eval_augmentation(augmented_data, clusters)
        if f1 < 0.99:
            print("augmentation bug")

        return augmented_data

    def print_augmentation(self, aug):
        s = ''

        start_mention = 5
        end_mention = 6
        not_mention_token = 3
        or_token = 4
        pad_token = 0

        for a in aug:
            if a > 6:
                s += str(a-7)
            else:
                if a == start_mention:
                    s += '('
                elif a == end_mention:
                    s += ')'
                elif a == not_mention_token:
                    s += '-'
                elif a == or_token:
                    s += '|'
                elif a == pad_token:
                    s += ' '
                else:
                    s += '?'
        return s

    def eval_augmentation(self, augmentation, gold_clusters):
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()

        gold_clusters = extract_clusters2(gold_clusters)
        mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(
            gold_clusters)
        gold_mentions = list(mention_to_gold_clusters.keys())

        predicted_clusters = extract_clusters2(
            self.reverse_augmentation(augmentation))  # need our clusters
        mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(
            predicted_clusters)
        predicted_mentions = list(mention_to_predicted_clusters.keys())
        mention_evaluator.update(predicted_mentions, gold_mentions)
        coref_evaluator.update(predicted_clusters, gold_clusters,mention_to_predicted_clusters, mention_to_gold_clusters)
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()
        return f1

    def reverse_augmentation(self, tokens):
        # start_mention = 5
        # end_mention = 6
        # not_mention_token = 3
        # or_token = 4
        pad_token = 0

        if isinstance(self.tokenizer ,BartTokenizer):
            start_mention = 50265
        else:            
            start_mention = 32000

        end_mention = start_mention+1
        or_token = start_mention+2


        clusters_dict = {}
        i = 0
        ci = 0

        while i < len(tokens):
            if tokens[i] == start_mention and i+1 < len(tokens) and tokens[i+1] >= start_mention+3:  # valid cluster id #(id
                if tokens[i+1] in clusters_dict:
                    clusters_dict[tokens[i+1]] += (('s', ci),)
                else:
                    clusters_dict[tokens[i+1]] = (('s', ci),)
                i += 1
                # ci += 1

            elif tokens[i] >= start_mention+3 and i+1 < len(tokens) and tokens[i+1] == end_mention:  # id)
                # ci += 1

                if tokens[i] in clusters_dict:
                    clusters_dict[tokens[i]] += (('e', ci),)
                else:
                    clusters_dict[tokens[i]] = (('e', ci),)
                i += 1


            # elif tokens[i] == or_token:
                # ci -= 1

            elif tokens[i] < start_mention:
                ci += 1

            i += 1

        clusters = []
        for key in clusters_dict:
            mentions = clusters_dict[key]
            cluster = []
            stack = []
            for m in mentions:
                if 'se' in m:
                    cluster.append((m[1], m[1]))
                elif 's' in m:
                    stack.append(m[1])
                elif 'e' in m:
                    e = m[1]
                    if len(stack) > 0:
                        s = stack.pop()
                        if e == s and len(stack) > 0:
                            cluster.append((stack.pop(), e))
                            stack.append(s)
                        else:
                            cluster.append((s, e))
            if len(cluster)>0:
                clusters.append(tuple(cluster))

        return clusters

    def _tokenize2(self, examples):#normal tokanization
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers in tqdm(examples):
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = []

            token_ids = []
            for idx, word in enumerate(words):
                word_idx_to_start_token_idx[idx] = len(token_ids)
                tokenized = self.tokenizer.encode(
                    " " + word, add_special_tokens=False)

                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)

                word_idx_to_end_token_idx[idx] = len(token_ids)

            token_ids.append(self.tokenizer.eos_token_id)

            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]

            if new_clusters != []:
                augmented = self._augment_data(token_ids, new_clusters)

            if 0 < self.max_seq_length < len(augmented):
                num_examples_filtered += 1
                continue

            lengths.append(len(token_ids))

            if new_clusters != []:
                coref_examples.append(((doc_key, end_token_idx_to_word_idx), CorefExample(
                    token_ids=token_ids, clusters=new_clusters, augmented_labels=augmented)))

        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(self, batch, max_length):
        padded_batch = []
        for example in batch:
            encoded_dict = self.tokenizer.encode_plus(
                example[0], pad_to_max_length=True, max_length=max_length, return_tensors='pt')
            clusters = self.pad_clusters(example.clusters)

            # add padding to augmented data as well
            encoded_dict_augmented = self.tokenizer.encode_plus(
                example[2], pad_to_max_length=True, max_length=max_length, return_tensors='pt')

            example = (encoded_dict["input_ids"], encoded_dict['attention_mask'], torch.tensor(
                clusters), encoded_dict_augmented["input_ids"], encoded_dict_augmented['attention_mask'])
            padded_batch.append(example)
        tensored_batch = tuple(torch.stack([example[i].squeeze(
        ) for example in padded_batch], dim=0) for i in range(len(example)))
        return tensored_batch

    def _tokenize(self, examples):#extended tokenization
        seperetor = '.'
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers in tqdm(examples):
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = []

            token_ids = []
            seperators = []
            for idx, word in enumerate(words):
                word_idx_to_start_token_idx[idx] = len(token_ids)
                tokenized = self.tokenizer.encode(
                    " " + word, add_special_tokens=False)

                if word==seperetor:
                    sep=[len(end_token_idx_to_word_idx)]
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                if word==seperetor:
                    sep.append(len(end_token_idx_to_word_idx))
                    seperators.append(sep)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)

                


            base_clusters = [[(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            token_ids = self.extend_token_ids(token_ids,seperators)
            for token_id in token_ids:
                trimmed_clusters = trim_clusters(base_clusters, token_id[1], token_id[2])
                new_clusters = trimmed_clusters
                if new_clusters != []:
                    augmented = self._augment_data(token_id[0], new_clusters)

                if 0 < self.max_seq_length < len(augmented):
                    num_examples_filtered += 1
                    continue

                lengths.append(len(token_id[0]))

                if new_clusters != []:
                    coref_examples.append(((doc_key, end_token_idx_to_word_idx), CorefExample(
                        token_ids=token_id[0], clusters=new_clusters, augmented_labels=augmented)))

        return coref_examples, lengths, num_examples_filtered

    def __extend_token_ids(self, token_ids):
        if self.model == 't5':
            dot = [3, 5]
            eos = 1
        else:
            dot = [479]
            eos = 2

        idxs = find_sub_list(dot, token_ids) 
        sentences = []

        start = 0
        for idx in idxs:
            end = idx[1]+1
            sentences.append((token_ids[start:end],start,end))
            start = end

        extended_tokens = [(token_ids, 0, len(token_ids))]
        num_scentences = len(sentences)
        for i in range(num_scentences-1):
            token_sentence = sentences[i][0].copy()
            sub = random.randint(i+2,num_scentences)
            for j in range(i+1,sub):
                token_sentence.extend(sentences[j][0].copy())
            token_sentence.append(eos)
            extended_tokens.append((token_sentence, sentences[i][1], sentences[j][2]+1))
            
        return extended_tokens
        
    def extend_token_ids(self,token_ids, idxs):
        eos = token_ids[-1]
        sentences = []

        start = 0
        for idx in idxs:
            end = idx[1]
            sentences.append((token_ids[start:end],start,end))
            start = end

        extended_tokens = [(token_ids, 0, len(token_ids))]
        num_scentences = len(sentences)
        for i in range(num_scentences-1):
            token_sentence = sentences[i][0].copy()
            sub = random.randint(i+2,num_scentences)
            for j in range(i+1,sub):
                token_sentence.extend(sentences[j][0].copy())
            token_sentence.append(eos)
            extended_tokens.append((token_sentence, sentences[i][1], sentences[j][2]+1))
            
        return extended_tokens


def get_dataset(args, tokenizer, evaluate=False):
    read_from_cache, file_path = False, ''
    if evaluate and os.path.exists(args.predict_file_cache):
        file_path = args.predict_file_cache
        read_from_cache = True
        # read_from_cache = False

    elif (not evaluate) and os.path.exists(args.train_file_cache):
        file_path = args.train_file_cache
        read_from_cache = True
        # read_from_cache = False

    # read_from_cache = False
    if read_from_cache:
        logger.info(f"Reading dataset from {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    file_path, cache_path = (args.predict_file, args.predict_file_cache) if evaluate else (
        args.train_file, args.train_file_cache)
    # file_path, cache_path = (args.predict_file, args.predict_file_cache)

    coref_dataset = CorefDataset(file_path, tokenizer, max_seq_length=args.max_seq_length)
    with open(cache_path, 'wb') as f:
        pickle.dump(coref_dataset, f)

    return coref_dataset


def trim_clusters(clusters, start, end):
    return list(filter(lambda lst: len(lst)>1 , [list(filter(lambda x: x[1] < (end-start) and x[0] >= 0,map(lambda x: (x[0]-start, x[1]-start), cluster))) for cluster in clusters]))


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results