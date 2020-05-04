"""RNN Beam Search example

I refered the following.
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
"""
from heapq import heappush, heappop
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F


# EncoderRNN {{{
class EncoderRNN(nn.Module):
    def __init__(self, embd_size, h_size, vocab_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.rnn = nn.GRU(embd_size, h_size, batch_first=False)

    def forward(self, x):
        # x: (L, bs, H)
        embedded = self.embedding(x) # (L, bs, E)
        output, hidden = self.rnn(embedded) # (L, bs, H), (1, bs, H)

        return output, hidden
# }}}

# DecoderRNN {{{
class DecoderRNN(nn.Module):
    def __init__(self, embd_size, h_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.h_size = h_size
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.rnn = nn.GRU(embd_size, h_size)
        self.fout = nn.Linear(h_size, vocab_size)

    def forward(self, x, hidden):
        # x: (T, bs), hiden: (bs, H)
        embedded = self.embedding(x)  # (1, 1, E) = (T, bs, E)
        out, hidden = self.rnn(embedded, hidden) # (1, 1, H)=(T, bs, H), (1, bs, H)

        out = self.fout(out.squeeze(0)) # (bs, V)=(1,V)
        out =  F.log_softmax(out, dim=1)
        return out, hidden
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
def beam_search_decoding(decoder, enc_out, beam_width, n_best, max_dec_steps=1000):
    """Beam Seach Decoding for RNN

    Args:
        decoder: An RNN decoder model
        enc_output: A sequence of encoded input. (T, bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input

    Returns:
        res_sequences: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    res_sequences = []
    bs = enc_out.shape[1]

    # Decoding goes sentence by sentence.
    # So this process is very slow compared to batch decoding process.
    for batch_id in range(bs):
        # Get last encoder hidden state
        decoder_hidden = enc_out[-1][batch_id].unsqueeze(0).unsqueeze(0) # (T, bs, H) = (1, 1, H)

        # Prepare first token for decoder
        decoder_input = torch.tensor([SOS_token]).long().to(DEVICE).unsqueeze(0) # (T, bs) = (1, 1)

        # Number of sentence to generate
        end_nodes = []

        # Starting node
        node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)

        # Whole beam search node graph
        nodes = []

        # Start the queue
        heappush(nodes, (-node.eval(), node))
        n_dec_steps = 0

        # Start beam search
        while True:
            # Give up when decoding takes too long
            if n_dec_steps > max_dec_steps:
                break

            # Fetch the best node
            score, n = heappop(nodes)
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == EOS_token and n.prev_node != None:
                end_nodes.append((score, n))
                # If we reached maximum # of sentences required
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            # Decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # Get top-k from this decoded result
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (1, bw), (1, bw)
            next_nodes = []
            # Then, register new top-k nodes
            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1, -1) # (1, 1)
                logp = topk_log_prob[0][new_k].item() # float log probability val

                node = BeamSearchNode(h=decoder_hidden,
                                      prev_node=n,
                                      wid=decoded_t,
                                      logp=n.logp+logp,
                                      length=n.length+1)
                heappush(nodes, (-node.eval(), node))
            n_dec_steps += beam_width

        # If there are no end_nodes, retrieve best nodes (they are probably truncated)
        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        # Construct sequences from end_nodes
        for score, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = []
            sequence.append(n.wid)
            # back trace from end node
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid)
            sequence = sequence[::-1] # reverse

            res_sequences.append(sequence)

    return res_sequences
# }}}


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    batch_size = 4
    embd_size = 64
    h_size = 128
    vocab_size = 1000
    beam_width = 5
    n_best = 3

    x = torch.zeros(MAX_LENGTH, batch_size).long().to(DEVICE) # dummy input

    encoder = EncoderRNN(embd_size, h_size, vocab_size).to(DEVICE)
    out, _h = encoder(x) # (T, bs, H), (1, 16, H)
    decoder = DecoderRNN(embd_size, h_size, vocab_size).to(DEVICE)
    decoded_seqs = beam_search_decoding(decoder, out, beam_width=beam_width, n_best=n_best)
    for seq in decoded_seqs:
        print([x.item() for x in seq])


if __name__ == '__main__':
    main()
