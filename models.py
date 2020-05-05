import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# EncoderRNN {{{
class EncoderRNN(nn.Module):
    def __init__(self, embd_size, h_size, v_size, n_layers, dropout, device):
        super(EncoderRNN, self).__init__()
        self.h_size = h_size
        self.v_size = v_size
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, h_size, num_layers=n_layers, dropout=dropout)

    def forward(self, x):
        # x: (L, bs, H)
        embedded = self.embedding(x) # (L, bs, E)
        output, hidden = self.rnn(embedded) # (L, bs, H), (1, bs, H)

        return output, hidden
# }}}

# DecoderRNN {{{
class DecoderRNN(nn.Module):
    def __init__(self, embd_size, h_size, v_size, n_layers, dropout, device):
        super(DecoderRNN, self).__init__()
        self.h_size = h_size
        self.v_size = v_size
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, h_size, num_layers=n_layers, dropout=dropout)
        self.fout = nn.Linear(h_size, v_size)

    def forward(self, x, hidden):
        # x: (bs), hiden: (1, bs, H)
        x = x.unsqueeze(0) # (1, bs)
        embedded = self.embedding(x)  # (1, 1, E) = (T, bs, E)
        out, hidden = self.rnn(embedded, hidden) # (T, bs, H), (0, bs, H)

        out = self.fout(out.squeeze(0)) # (bs, V)=(1,V)
        out =  F.log_softmax(out, dim=1)
        return out, hidden
# }}}

# Seq2Seq {{{
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.h_size == decoder.h_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size]
        trg = [trg len, batch size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_v_size = self.decoder.v_size

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_v_size).to(self.device) # (trg_len, bs, trg_vocab)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        _out, hidden = self.encoder(src) # (T, bs, H), (1, bs, H)

        #first input to the decoder is the <sos> tokens
        inp = trg[0, :] # (bs) # first token

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            # output: (bs, trg_vocab)
            # hidden, cell: (2, bs, hid)
            output, hidden = self.decoder(inp, hidden)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #get the highest predicted token from our predictions
            top1 = output.argmax(1) # (bs)

            teacher_force = random.random() < teacher_forcing_ratio
            inp = trg[t] if teacher_force else top1

        return outputs
    # }}}
