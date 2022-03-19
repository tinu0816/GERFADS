
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer as Tokenizer, BartForConditionalGeneration

import textwrap
import argparse
from argparse import Namespace

# In[2]:


#import seaborn as sns
#from pylab import rcParams
#import matplotlib.pyplot as plt
#from matplotlib import rc

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from transformers import (
    AdamW,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
    )
import gc


#MODEL_NAME = "timo/timo-BART-german"
#langs = ['de', 'en']
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, langs, model_max_length=1548)
#print(tokenizer)

def freeze_params(model):
    for layer in model.parameters():
        layer.requires_grade=False

def shift_tokens_right(input_ids, pad_token_id):
    prev_output_tokens=input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1)-1).unsqueeze(-1)
    prev_output_tokens[:, 0]=input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:,:-1]
    return prev_output_tokens


MODEL_NAME = "facebook/bart-base"
tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

class GermanModel(pl.LightningModule):
    def __init__(self, learning_rate, tokenizer):
        super().__init__()
        hparams = argparse.Namespace()
        hparams.freeze_encoder = True
        hparams.freeze_embeds = True
        hparams.eval_beams = 4

        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.learning_rate = learning_rate
        self.save_hyperparameters("learning_rate")
        self.save_hyperparameters(hparams)
        self.scaler = torch.cuda.amp.GradScaler()

        if self.hparams.freeze_encoder:
            freeze_params(self.model.model.encoder)
        if self.hparams.freeze_embeds:
            self.freeze_embeds()

    def freeze_embeds(self):
        freeze_params(self.model.model)
        for d in [self.model.model.encoder, self.model.model.decoder]:
          freeze_params(d.embed_positions)
          freeze_params(d.embed_tokens)
        
    def forward(self, input_ids, decoder_input_ids, attention_mask, use_cache, labels=None):
        output = self.model(
            input_ids,
            decoder_input_ids = decoder_input_ids,
            attention_mask = attention_mask,
            use_cache = use_cache,
            labels = labels
        )
        torch.cuda.empty_cache()
        torch.no_grad()
        return output.loss, output.logits

    def on_after_backward(self) -> None:
        print("on_after_backwards")
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            #log.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
            
        src_ids, src_mask = batch["text_input_ids"], batch["text_attention_mask"]
        tgt_ids = batch["labels"]

        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)
        loss, outputs = self(src_ids,
                    attention_mask=src_mask,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                    labels=tgt_ids)
        torch.cuda.empty_cache()
        del src_ids, src_mask, batch, tgt_ids, decoder_input_ids
        self.scaler.scale(loss)
        if torch.isnan(loss) or torch.isinf(loss):
            print("nan or inf detected. skipping")
            return None
        #self.scaler.update()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask=labels,
            labels=labels
          )
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def generate(self, text, eval_beams, early_stopping=True, max_len=40):
        
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams = eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)for w in generated_ids]
    
        
    def generate_text(self, text, eval_beams, early_stopping=True, max_len=40):
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams = eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)for w in generated_ids]
    



def prepare_hparams():
    hparams = argparse.Namespace()
    hparams.freeze_encoder = True
    hparams.freeze_embeds=True
    hparams.eval_beams = 3
    hparams.gradient_clip_val = 1
    hparams.learning_rate = 2e-5
    return hparams



def prepare_trained_model():

    path = "C:/Users/marti/checkpoints/bart_dialogs_en_trained.ckpt"
    hparams = prepare_hparams()
    model = GermanModel(2e-5, tokenizer)
    trained_model = model.load_from_checkpoint(
        tokenizer = tokenizer,
        checkpoint_path = path
    )
    trained_model.freeze()
    return trained_model


def summarize(model, text):

    text_encoding = tokenizer(text, 
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_attention_mask = True,
            return_tensors="pt"
            )

    generated_ids = model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=4,
        decoder_start_token_id=tokenizer.pad_token_id,
        #repetition_penalty=2.5,
        #length_penalty=1.0,
        early_stopping=True
    )
    preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for gen_id in generated_ids]
    return "".join(preds)

