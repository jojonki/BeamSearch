import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Queue import PriorityQueue
from heapq import heappush, heappop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class EncoderRNN(nn.Module):# {{{
    def __init__(self, embd_size, h_size, output_size):
        super(EncoderRNN, self).__init__()
        self.h_size = h_size
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embd_size)

        self.rnn = nn.GRU(embd_size, h_size, batch_first=False)

    def forward(self, x):
        # x: (L, bs, H)
        embedded = self.embedding(x) # (L, bs, E)
        output, hidden = self.rnn(embedded) # (L, bs, H), (1, bs, H)

        return output, hidden
# }}}

class DecoderRNN(nn.Module):# {{{
    def __init__(self, embd_size, h_size, output_size):
        '''
        Illustrative decoder
        '''
        super(DecoderRNN, self).__init__()
        self.h_size = h_size
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embd_size,
                                      )

        self.rnn = nn.GRU(embd_size, h_size, batch_first=False)
        # self.dropout_rate = dropout
        self.fout = nn.Linear(h_size, output_size)

    def forward(self, x, hidden):
        # x: (T, bs), hiden: (bs, H)
        embedded = self.embedding(x)  # (1, 1, E) = (T, bs, E)
        out, hidden = self.rnn(embedded, hidden) # (1, 1, H)=(T, bs, H), (1, bs, H)

        out = self.fout(out.squeeze(0)) # (bs, V)=(1,V)
        out =  F.log_softmax(out, dim=1)
        return out, hidden
# }}}


class BeamSearchNode(object):# {{{
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

def _b(target_tensor, decoder_hiddens, encoder_outputs=None):# {{{
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == EOS_token and n.prev_node != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                logp = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + logp, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wid)
            # back trace
            while n.prev_node != None:
                n = n.prev_node
                utterance.append(n.wid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch# }}}

def greedy_decode(decoder_hidden, encoder_outputs, target_tensor):# {{{
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch# }}}


def beam_search(decoder, enc_out, beam_width=2):# {{{
    # enc_out: (T, bs, H)

    bs = enc_out.shape[1]
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(bs):
        decoder_hidden = enc_out[-1][idx].unsqueeze(0).unsqueeze(0) # (T, bs, H) = (1, 1, H) get last h
        # encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([SOS_token]).unsqueeze(0) # (1, 1) = (T, bs)

        # Number of sentence to generate
        endnodes = []
        # number_required = min((topk + 1), topk - len(endnodes))
        number_required = beam_width

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)
        nodes = []

        # start the queue
        heappush(nodes, (-node.eval(), node))
        qsize = 0

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 200: break

            # fetch the best node
            score, n = heappop(nodes)
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == EOS_token and n.prev_node != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # PUT HERE REAL BEAM SEARCH OF TOP
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (1, bw), (1, bw)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1, -1) # (1, 1)
                logp = topk_log_prob[0][new_k].item() # float log probability val

                node = BeamSearchNode(h=decoder_hidden, prev_node=n, wid=decoded_t, logp=n.logp+logp, length=n.length+1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                heappush(nodes, (score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [heappop(nodes) for _ in range(beam_width)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wid)
            # back trace from end node
            while n.prev_node != None:
                n = n.prev_node
                utterance.append(n.wid)
            utterance = utterance[::-1] # reverse

            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch# }}}


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50

bs = 16
embd_size = 64
H = 128
V = 1000
x = torch.zeros(MAX_LENGTH, bs).long()
encoder = EncoderRNN(embd_size, H, V)
out, h = encoder(x) # (T, bs, H), (1, 16, H)
decoder = DecoderRNN(embd_size, H, V)
beam_search(decoder, out)

if __name__ == '__main__':
    main()
