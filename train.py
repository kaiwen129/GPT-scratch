import torch
from torch import nn 
from torch.nn import functional as F
import torch.distributions as dist

import transformer
from transformer import GenTRF

import numpy as np

import random, tqdm, gzip

torch.set_default_device('cuda')

EMBEDDING_DIM = 128
BATCH_SIZE = 32
NUM_BATCHES = 40000
NUM_TOKENS = 256 # enwik8 uses 9 to 240

LEARNING_RATE = 0.0001
ATTENTION_HEADS = 8
DEPTH = 8
SEQ_LENGTH = 256
SAMPLE_SEQ_LENGTH = 256 # generate 256 new characters when testing
TEMPERATURE = 0.5

SAVE_PATH = "model/gentrf.pth"

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    # Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def sample_batch(data, length=SEQ_LENGTH, batch_size=BATCH_SIZE):
    """
    Randomly creates `batch_size` inputs of length `length` by choosing arbitrary subsequences from `data` (single sequence of tokens).

    Also creates target vector (input shifted one to the right) for each input.

    Returns inputs and targets as two matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    starts = torch.randint(size=(batch_size,), low=0, high=data.size()[0] - length - 1)

    inputs = [data[i:i + length] for i in starts]
    targets = [data[i+1:i+length+1] for i in starts]

    # concatenate into input matrix X and target matrix T
    X = torch.cat([x[None, :] for x in inputs], dim=0).to(torch.long)
    T = torch.cat([t[None, :] for t in targets], dim=0).to(torch.long)

    X, T = X.to(device), T.to(device)

    return X, T

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_data, val_data, te_data = enwik8('data/enwik8')

    model = GenTRF(EMBEDDING_DIM, ATTENTION_HEADS, SEQ_LENGTH, NUM_TOKENS, DEPTH)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.to(device)

    optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

    for i in tqdm.trange(NUM_BATCHES):
        optimizer.zero_grad()

        X, T = sample_batch(tr_data)
        X, T = X.to(device), T.to(device)
        
        output = model(X)
        loss = F.nll_loss(output.transpose(2,1), T)

        loss.backward()

        optimizer.step()

    torch.save(model.state_dict(), SAVE_PATH)
    print("model saved")

def sample(lnprobs, temp=1.0):
    """
    Sample element from `lnprobs` with temperature `temp`

    :param temp: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    """
    if temp == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temp, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def sample_seq(x, length=SAMPLE_SEQ_LENGTH):
    '''
    Using the current model, generates new characters from a `seq_length` long input x.

    :x: input as a string of characters
    '''
    model = GenTRF(EMBEDDING_DIM, ATTENTION_HEADS, SEQ_LENGTH, NUM_TOKENS, DEPTH)
    model.load_state_dict(torch.load(SAVE_PATH))

    seq = torch.tensor([ord(c) for c in x])

    with torch.no_grad():
        for i in range(length):
            output = seq[-SEQ_LENGTH:]
            output = model(seq[None, :])

            c = sample(output[0, -1, :], temp=TEMPERATURE)
            seq = torch.cat([seq[1:], c[None]], dim=0)

    output_str = ''.join(chr(i) for i in seq)

    print(x, " ...... ", output_str)


### Testing ###

#sample_seq("The Battle of Verdant Fields, fought on a crisp autumn day, was a clash of titans that echoed through the pages of history. The armies assembled, banners unfurled against the azure sky, as the sun cast a golden hue over the verdant landscape. Swords clashe")

train()

