'''
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from model.transformer import Transformer
from config import *

model = Transformer(src_pad_idx, 
                    trg_pad_idx, 
                    trg_sos_idx, 
                    enc_voc_size, 
                    dec_voc_size, 
                    d_model, 
                    n_head, 
                    max_len, 
                    ffn_hidden, 
                    n_layers, 
                    drop_prob,
                    device).to(device)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

def primitive(model, optimizer, src, trg):
    model.train()
    optimizer.zero_grad()
    output = model(src, trg[:, :-1])
    
    out_file = open("record.txt", "w+")
    torch.set_printoptions(threshold=np.inf)
    
    print("Original Output Shape: {}".format(output.shape))
    print("Original Output: \n{}\n".format(output), file=out_file)
    print("Original Output Shape: {}\n".format(output.shape), file=out_file)

    output_reshape = output.contiguous().view(-1, output.shape[-1])
    trg = trg[:, 1:].contiguous().view(-1)

    print("reshaped: {}, {}".format(output_reshape.shape, trg.shape))
    print("Reshaped Output: \n{}\n".format(output_reshape), file=out_file)
    print("Reshaped Output Shape: {}\n".format(output_reshape.shape), file=out_file)

    out_file.close()
    print("##################################")
    print("# Finish Running.                #")
    print("# See 'record.txt' for Details.  #")
    print("##################################")

mtx_A = torch.randint(1, 100, (128, 41))
mtx_B = torch.randint(1, 100, (128, 43))
primitive(model, optimizer, mtx_A, mtx_B)
'''

"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

from config import *
from model.transformer import Transformer
#from util.bleu import idx_to_word, get_bleu
#from util.epoch_timer import epoch_time

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)
                    
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def run(best_loss):
    train_loss = train(model, train_iter, optimizer, criterion, clip)

if __name__ == '__main__':
    run(best_loss=inf)