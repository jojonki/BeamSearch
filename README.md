# BeamSearch
This is a sample code of beam search decoding for pytorch. `run.py` trains a translation model (de -> en). 

There are two beam search implementations.
- `beam_search_decoding` decodes sentence by sentence. Although this implementation is slow, this may help your understanding for its simplicity.

- `batch_beam_search_decodingl` decodes sentences as a batch and faster than `beam_search_decoding` (see the execution time in the below log). I believe that current implementation is not reasonable since there are lot of `for loop` implementations and there are much space for batch processing.

Both ouputs of two implementaions must be the same.


## How to Use?
```
// from training
% python run.py

// Or load pretrained model
% python run.py --model_path ./ckpts/s2s-model.pt --skip_train
Number of training examples: 29000
Number of validation examples: 1014
Number of testing examples: 1000
Unique tokens in source (de) vocabulary: 7855
Unique tokens in target (en) vocabulary: 5893
In: <SOS> . schnee den über laufen hunde mittelgroße zwei <EOS>
for loop beam search time: 1.717
Out: Rank-1: <SOS> two dogs dogs run across the grass . <EOS>
Out: Rank-2: <SOS> two dogs dogs run through the grass . <EOS>
Out: Rank-3: <SOS> two dogs dogs running through the snow . <EOS>
Out: Rank-4: <SOS> two dogs dogs run through the snow . <EOS>
Out: Rank-5: <SOS> two dogs dogs run across the snow . <EOS>
Batch beam search time: 1.168
Out: Rank-1: <SOS> two dogs dogs run across the grass . <EOS>
Out: Rank-2: <SOS> two dogs dogs run through the grass . <EOS>
Out: Rank-3: <SOS> two dogs dogs running through the snow . <EOS>
Out: Rank-4: <SOS> two dogs dogs run through the snow . <EOS>
Out: Rank-5: <SOS> two dogs dogs run across the snow . <EOS>
In: <SOS> . <unk> mit tüten gehsteig einem auf verkauft frau eine <EOS>
for loop beam search time: 2.943
Out: Rank-1: <SOS> a woman is on the sidewalk on a sidewalk . <EOS>
Out: Rank-2: <SOS> a woman is on on a sidewalk sidewalk . <EOS>
Out: Rank-3: <SOS> a woman is walking on a sidewalk sidewalk . <EOS>
Out: Rank-4: <SOS> a woman is walking on the sidewalk sidewalk . <EOS>
Out: Rank-5: <SOS> a woman is on the sidewalk on a sidewalk . . <EOS>
Batch beam search time: 1.862
Out: Rank-1: <SOS> a woman is on the sidewalk on a sidewalk . <EOS>
Out: Rank-2: <SOS> a woman is on on a sidewalk sidewalk . <EOS>
Out: Rank-3: <SOS> a woman is walking on a sidewalk sidewalk . <EOS>
Out: Rank-4: <SOS> a woman is walking on the sidewalk sidewalk . <EOS>
Out: Rank-5: <SOS> a woman is on the sidewalk on a sidewalk . . <EOS>
In: <SOS> . bushaltestelle einer an sitzt anzug im mann ein <EOS> <pad> <pad>
for loop beam search time: 3.354
Out: Rank-1: <SOS> a man in a suit is sitting at a podium . <EOS>
Out: Rank-2: <SOS> a man in a suit is sitting at a table . <EOS>
Out: Rank-3: <SOS> a man in a suit is sitting at a cafe . <EOS>
Out: Rank-4: <SOS> a man in a suit sitting at a podium . <EOS>
Out: Rank-5: <SOS> a man in a is sitting at a podium . <EOS>
Batch beam search time: 2.204
Out: Rank-1: <SOS> a man in a suit is sitting at a podium . <EOS>
Out: Rank-2: <SOS> a man in a suit is sitting at a table . <EOS>
Out: Rank-3: <SOS> a man in a suit is sitting at a cafe . <EOS>
Out: Rank-4: <SOS> a man in a suit sitting at a podium . <EOS>
Out: Rank-5: <SOS> a man in a is sitting at a podium . <EOS>
```
