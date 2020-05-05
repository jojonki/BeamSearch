import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# EncoderRNN {{{
class EncoderRNN(nn.Module):
    def __init__(self, embd_size, enc_h_size, dec_h_size, v_size, device):
        super(EncoderRNN, self).__init__()
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.v_size = v_size
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, enc_h_size, bidirectional=True)
        self.f_concat_h = nn.Linear(enc_h_size*2, dec_h_size)

    def forward(self, x):
        # x: (T, bs, H)
        embedded = self.embedding(x) # (T, bs, E)
        output, hidden = self.rnn(embedded) # (T, bs, 2H), (2, bs, H)
        hidden = torch.tanh(self.f_concat_h(torch.cat((hidden[-2], hidden[-1]), dim=1))) # (bs, H)

        return output, hidden.squeeze(0) # (T, bs, 2H), (bs, H)
# }}}

# Attention {{{
class Attention(nn.Module):
    def __init__(self, enc_h_size, dec_h_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_h_size * 2) + dec_h_size, dec_h_size)
        self.v = nn.Linear(dec_h_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outs):
        """
        dec_hidden: (bs, decH)
        enc_outs: (T, bs, encH*2)
        """
        bs = enc_outs.shape[1]
        src_len = enc_outs.shape[0]

        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1) # (bs, T, decH)
        enc_outs = enc_outs.permute(1, 0, 2) # (bs, T, encH*2)

        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outs), dim=2))) # (bs, T, encH*2+decH)
        attention = self.v(energy).squeeze(2) # (bs, T)

        return F.softmax(attention, dim=1)
# }}}

# DecoderRNN {{{
class DecoderRNN(nn.Module):
    def __init__(self, embd_size, dec_h_size, v_size, device):
        super(DecoderRNN, self).__init__()
        self.dec_h_size = dec_h_size
        self.v_size = v_size
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU(embd_size, dec_h_size)
        self.fout = nn.Linear(dec_h_size, v_size)

    def forward(self, x, hidden, _enc_outs):
        """
        x: (bs)
        hiden: (bs, H)
        _enc_outs: only used for AttnDecoderRNN and not used here.
        """
        x = x.unsqueeze(0) # (1, bs)
        hidden = hidden.unsqueeze(0) # (1, H)
        embedded = self.embedding(x)  # (1, 1, E) = (T, bs, E)
        out, hidden = self.rnn(embedded, hidden) # (T, bs, H), (1, bs, H)

        out = self.fout(out.squeeze(0)) # (bs, V)=(1,V)
        out =  F.log_softmax(out, dim=1)
        return out, hidden.squeeze(0) # (T, bs, H), (bs, H)
# }}}

# AttnDecoderRNN {{{
class AttnDecoderRNN(nn.Module):
    def __init__(self, embd_size, enc_h_size, dec_h_size, v_size, attn, device):
        super(AttnDecoderRNN, self).__init__()
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.v_size = v_size
        self.attn = attn
        self.device = device

        self.embedding = nn.Embedding(v_size, embd_size)
        self.rnn = nn.GRU((embd_size+enc_h_size*2), dec_h_size)
        self.fout = nn.Linear(dec_h_size+enc_h_size*2+embd_size, v_size)

    def forward(self, x, dec_hidden, enc_outs):
        """
        x: (bs),
        dec_hidden: (bs, decH)
        enc_outs: (T, bs, encH*2)
        """
        x = x.unsqueeze(0) # (1, bs)
        embedded = self.embedding(x) # (1, bs, E)
        a = self.attn(dec_hidden, enc_outs).unsqueeze(1) # (bs, 1, T)
        enc_outs = enc_outs.permute(1, 0, 2) # (bs, T, encH*2)
        weighted = torch.bmm(a, enc_outs) # (bs, 1, encH*2)
        weighted = weighted.permute(1, 0, 2) # (1, bs, encH*2)
        rnn_input = torch.cat((embedded, weighted), dim=2) # (1, bs, E+encH*2)

        out, dec_hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(0)) # (1, bs, H), (1, bs, H)
        # seq_len, n_layers and n_directions will always be 1 in this decoder, therefore:
        assert (out == dec_hidden).all()

        embedded = embedded.squeeze(0) # (bs, E)
        out = out.squeeze(0) # (bs, H)
        weighted = weighted.squeeze(0) # (bs, encH*2)
        pred = self.fout(torch.cat((out, weighted, embedded), dim=1)) # (bs, V)

        return pred, dec_hidden.squeeze(0) # (bs, V), (bs, H)
# }}}

# Seq2Seq {{{
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.enc_h_size == decoder.dec_h_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

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
        enc_outs, hidden = self.encoder(src) # (T, bs, encH*), (bs, H)

        #first input to the decoder is the <sos> tokens
        inp = trg[0, :] # (bs) # first token

        for t in range(1, trg_len):
            output, hidden = self.decoder(inp, hidden, enc_outs) # (bs, V), (bs, H)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #get the highest predicted token from our predictions
            top1 = output.argmax(1) # (bs)

            teacher_force = random.random() < teacher_forcing_ratio
            inp = trg[t] if teacher_force else top1

        return outputs
    # }}}
