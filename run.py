"""RNN Beam Search example

I refered the following code for Beam Search.
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
"""
import argparse
import copy
from heapq import heappush, heappop
import math
import os
import time

import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from models import EncoderRNN, DecoderRNN, Seq2Seq


# utils {{{
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_n_best(decoded_seq, itos):
    for rank, seq in enumerate(decoded_seq):
        print(f'Out: Rank-{rank+1}: {" ".join([itos[idx] for idx in seq])}')
# }}}


# train {{{
def train(model, itr, optimizer, criterion):
    print('Start training')
    model.train()
    epoch_loss = 0
    for batch in itr:
        src = batch.src # (T, bs)
        trg = batch.trg # (T, bs)

        optimizer.zero_grad()

        output = model(src, trg)

        output_size = output.shape[-1]

        output = output[1:].view(-1, output_size)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(itr)
# }}}

# evaluate {{{
def evaluate(model, itr, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in itr:
            src = batch.src
            trg = batch.trg

            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(itr)
# }}}

# BeamSearchNode {{{
class BeamSearchNode(object):
    def __init__(self, h, prev_node, wid, logp, length):
        self.h = h
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward
# }}}

# beam search {{{
def beam_search_decoding(decoder,
                         enc_last_h,
                         beam_width,
                         n_best,
                         sos_token,
                         eos_token,
                         max_dec_steps):
    """Beam Seach Decoding for RNN

    Args:
        decoder: An RNN decoder model
        enc_last_h: A sequence of encoded input. (n_layers, bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input

    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    n_best_list = []
    bs = enc_last_h.shape[1]

    # Decoding goes sentence by sentence.
    # So this process is very slow compared to batch decoding process.
    for batch_id in range(bs):
        # Get last encoder hidden state
        decoder_hidden = enc_last_h[:, batch_id].unsqueeze(1).contiguous() # (n_layers, 1, H)

        # Prepare first token for decoder
        decoder_input = torch.tensor([sos_token]).long().to(DEVICE) # (1)

        # Number of sentence to generate
        end_nodes = []

        # starting node
        node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)

        # whole beam search node graph
        nodes = []

        # Start the queue
        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0

        # Start beam search
        while True:
            # Give up when decoding takes too long
            if n_dec_steps > max_dec_steps:
                break

            # Fetch the best node
            score, _, n = heappop(nodes)
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == eos_token and n.prev_node is not None:
                end_nodes.append((score, id(n), n))
                # If we reached maximum # of sentences required
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # Get top-k from this decoded result
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (1, bw), (1, bw)
            # Then, register new top-k nodes
            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1) # (1)
                logp = topk_log_prob[0][new_k].item() # float log probability val

                node = BeamSearchNode(h=decoder_hidden,
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes, (-node.eval(), id(node), node))
            n_dec_steps += beam_width

        # if there are no end_nodes, retrieve best nodes (they are probably truncated)
        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        # Construct sequences from end_nodes
        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = [n.wid.item()]
            # back trace from end node
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse

            n_best_seq_list.append(sequence)

        n_best_list.append(n_best_seq_list)

    return n_best_list
# }}}

# batch beam search {{{
def batch_beam_search_decoding(decoder,
                               enc_last_h,
                               beam_width,
                               n_best,
                               sos_token,
                               eos_token,
                               max_dec_steps):
    """Batch Beam Seach Decoding for RNN

    Args:
        decoder: An RNN decoder model
        enc_last_h: A sequence of encoded input. (n_layers, bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input

    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    n_best_list = []
    bs = enc_last_h.shape[1]

    # Get last encoder hidden state
    decoder_hidden = enc_last_h # (n_layers, bs, H)

    # Prepare first token for decoder
    decoder_input = torch.tensor([sos_token]).repeat(1, bs).long().to(DEVICE) # (1, bs)

    # Number of sentence to generate
    end_nodes_list = [[] for _ in range(bs)]

    # whole beam search node graph
    nodes = [[] for _ in range(bs)]

    # Start the queue
    for bid in range(bs):
        # starting node
        node = BeamSearchNode(h=decoder_hidden[:, bid], prev_node=None, wid=decoder_input[:, bid], logp=0, length=1)
        heappush(nodes[bid], (-node.eval(), id(node), node))

    # Start beam search
    fin_nodes = set()
    history = [None for _ in range(bs)]
    n_dec_steps_list = [0 for _ in range(bs)]
    while len(fin_nodes) < bs:
        # Fetch the best node
        decoder_input, decoder_hidden = [], []
        for bid in range(bs):
            if bid not in fin_nodes and n_dec_steps_list[bid] > max_dec_steps:
                fin_nodes.add(bid)

            if bid in fin_nodes:
                score, n = history[bid] # dummy for data consistency
            else:
                score, _, n = heappop(nodes[bid])
                if n.wid.item() == eos_token and n.prev_node is not None:
                    end_nodes_list[bid].append((score, id(n), n))
                    # If we reached maximum # of sentences required
                    if len(end_nodes_list[bid]) >= n_best:
                        fin_nodes.add(bid)
                history[bid] = (score, n)
            decoder_input.append(n.wid)
            decoder_hidden.append(n.h)

        decoder_input = torch.cat(decoder_input).to(DEVICE) # (bs)
        decoder_hidden = torch.stack(decoder_hidden, 1).to(DEVICE) # (n_layers, bs, H)

        # Decode for one step using decoder
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) # (bs, V), (n_layers, bs, H)

        # Get top-k from this decoded result
        topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (bs, bw), (bs, bw)
        # Then, register new top-k nodes
        for bid in range(bs):
            if bid in fin_nodes:
                continue
            score, n = history[bid]
            if n.wid.item() == eos_token and n.prev_node is not None:
                continue
            for new_k in range(beam_width):
                decoded_t = topk_indexes[bid][new_k].view(1) # (1)
                logp = topk_log_prob[bid][new_k].item() # float log probability val

                node = BeamSearchNode(h=decoder_hidden[:, bid],
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes[bid], (-node.eval(), id(node), node))
            n_dec_steps_list[bid] += beam_width

    # Construct sequences from end_nodes
    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    for bid in range(bs):
        if len(end_nodes_list[bid]) == 0:
            end_nodes_list[bid] = [heappop(nodes[bid]) for _ in range(beam_width)]

        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes_list[bid], key=lambda x: x[0]):
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse

            n_best_seq_list.append(sequence)

        n_best_list.append(copy.copy(n_best_seq_list))

    return n_best_list
# }}}


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def main():
    # ArgumentParser {{{
    parser = argparse.ArgumentParser()
    # hyper parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--enc_embd_size', type=int, default=256)
    parser.add_argument('--dec_embd_size', type=int, default=256)
    parser.add_argument('--rnn_h_size', type=int, default=512)
    parser.add_argument('--n_enc_layers', type=int, default=2)
    parser.add_argument('--n_dec_layers', type=int, default=2)
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--dec_dropout', type=float, default=0.5)
    # other parameters
    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--n_best', type=int, default=5)
    parser.add_argument('--max_dec_steps', type=int, default=1000)
    parser.add_argument('--export_dir', type=str, default='./ckpts/')
    parser.add_argument('--model_name', type=str, default='s2s')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--skip_train', action='store_true')
    opts = parser.parse_args()
    # }}}

    SOS_token = '<SOS>'
    EOS_token = '<EOS>'
    SRC = Field(tokenize=tokenize_de,
                init_token=SOS_token,
                eos_token=EOS_token,
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token=SOS_token,
                eos_token=EOS_token,
                lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    print(f'Number of training examples: {len(train_data.examples)}')
    print(f'Number of validation examples: {len(valid_data.examples)}')
    print(f'Number of testing examples: {len(test_data.examples)}')

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    print(f'Unique tokens in source (de) vocabulary: {len(SRC.vocab)}')
    print(f'Unique tokens in target (en) vocabulary: {len(TRG.vocab)}')

    train_itr, valid_itr, test_itr =\
            BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=opts.batch_size,
                device=DEVICE)

    enc_v_size = len(SRC.vocab)
    dec_v_size = len(TRG.vocab)

    encoder = EncoderRNN(opts.enc_embd_size, opts.rnn_h_size, enc_v_size, opts.n_enc_layers, opts.enc_dropout, DEVICE)
    decoder = DecoderRNN(opts.dec_embd_size, opts.rnn_h_size, dec_v_size, opts.n_dec_layers, opts.dec_dropout, DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    if opts.model_path != '':
        model.load_state_dict(torch.load(opts.model_path))

    if not opts.skip_train:
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        best_valid_loss = float('inf')
        for epoch in range(opts.n_epochs):
            start_time = time.time()

            train_loss = train(model, train_itr, optimizer, criterion)
            valid_loss = evaluate(model, valid_itr, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                print('Update model!')
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(opts.export_dir, f'{opts.model_name}-model.pt'))
            else:
                print('Model was not updated. Stop training')
                break

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    TRG_SOS_IDX = TRG.vocab.stoi[TRG.init_token]
    TRG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(test_itr):
            src = batch.src # (T, bs)
            trg = batch.trg # (T, bs)
            print(f'In: {" ".join(SRC.vocab.itos[idx] for idx in src[:, 0])}')

            _out, h = model.encoder(src) # (T, bs, H), (n_layers, bs, H)
            # decoded_seqs: (bs, T)
            start_time = time.time()
            decoded_seqs = beam_search_decoding(decoder=model.decoder,
                                                enc_last_h=h,
                                                beam_width=opts.beam_width,
                                                n_best=opts.n_best,
                                                sos_token=TRG_SOS_IDX,
                                                eos_token=TRG_EOS_IDX,
                                                max_dec_steps=opts.max_dec_steps)
            end_time = time.time()
            print(f'for loop beam search time: {end_time-start_time:.3f}')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)

            start_time = time.time()
            decoded_seqs = batch_beam_search_decoding(decoder=model.decoder,
                                                      enc_last_h=h,
                                                      beam_width=opts.beam_width,
                                                      n_best=opts.n_best,
                                                      sos_token=TRG_SOS_IDX,
                                                      eos_token=TRG_EOS_IDX,
                                                      max_dec_steps=opts.max_dec_steps)
            end_time = time.time()
            print(f'Batch beam search time: {end_time-start_time:.3f}')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)


if __name__ == '__main__':
    main()
