import torch
from torch.nn import Module
from transformers import  BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.configuration_t5 import T5Config
from transformers.configuration_bart import BartConfig

from transformers.utils import logging
logger = logging.get_logger(__name__)


class OurModel(Module):
    def __init__(self, args, tokenizer=None):
        super().__init__()

        self.model_type = args.model_type

        if self.model_type == 'bart':
            pret_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            self.model = BartForConditionalGeneration(config=BartConfig.from_pretrained('facebook/bart-base',dropout=args.dropout_prob,max_position_embeddings=4096,classifier_dropout=args.dropout_prob))
            own_state_dict = self.model.state_dict()
            for (name, p),(name2, p2) in zip(self.model.named_parameters(),pret_model.named_parameters()):
                if name == name2 and p.shape==p2.shape:
                    own_state_dict[name2].copy_(p2)
                else:
                    print(name)
            
            self.model.resize_token_embeddings(len(tokenizer))
        elif self.model_type == 'bart-raw':
            self.model = BartForConditionalGeneration(config=BartConfig.from_pretrained('facebook/bart-base',dropout=args.dropout_prob,max_position_embeddings=4096,classifier_dropout=args.dropout_prob))
            self.model.resize_token_embeddings(len(tokenizer))
        elif self.model_type == 't5':
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small', config=T5Config(n_positions=4096,decoder_start_token_id=0,dropout_rate=args.dropout_prob))
        elif self.model_type == 't5-raw':
            self.model = T5ForConditionalGeneration(config=T5Config(n_positions=4096, decoder_start_token_id=0,dropout_rate=args.dropout_prob))
            
        else:
            raise ValueError("model_type should be one of [None,'bart','t5']")

        if args.freeze_shared:
            for name, p in self.model.named_parameters():
                if "shared" in name:
                    p.requires_grad = False


    def from_pretrained(self, dir):
        if self.model_type == 'bart' or self.model_type == 'bart-raw':
            self.model = BartForConditionalGeneration.from_pretrained(dir)
        elif self.model_type == 't5' or self.model_type == 't5-raw':
            self.model = T5ForConditionalGeneration.from_pretrained(dir)

    def save_pretrained(self, dir):
        self.model.save_pretrained(dir)

    def count_parameters(self,trainable=True):
        return sum(p.numel() for p in self.model.parameters() if (p.requires_grad) or (not trainable))

    def forward(self, input_ids, input_attention_mask, data_augmentations, output_attention_mask):
        return self.model(input_ids=input_ids, attention_mask=input_attention_mask, labels=data_augmentations)
        

    
    def generate(self, input_ids, input_attention_mask, data_augmentations, output_attention_mask, **kwargs):
        with torch.no_grad():
            return self.model.generate(input_ids, **kwargs)[:, 1:]#  generation has a proceeding token
