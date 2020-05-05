# Seq2Seq Beam Search Decoding for Pytorch
This is a sample code of beam search decoding for pytorch. `run.py` trains a translation model (de -> en). 

There are two beam search implementations.
- `beam_search_decoding` decodes sentence by sentence. Although this implementation is slow, this may help your understanding for its simplicity.

- `batch_beam_search_decoding` decodes sentences as a batch and faster than `beam_search_decoding` (see the execution time in the below log). I believe that current implementation is not reasonable since there are lot of `for loop` implementations and there are much space for batch processing.

Both ouputs of two implementaions must be the same.


## How to Use?
```
// trains vanilla seq2seq model
% python run.py

// Or trains seq2seq model with attention
% python run.py --attention

// Or load pretrained vanilla model
% python run.py --model_path ./ckpts/s2s-vanilla.pt --skip_train
// Or load pretrained attention model
% python run.py --attention --skip_train --model_path ./ckpts/s2s-attn.pt
Number of training examples: 29000
Number of validation examples: 1014
Number of testing examples: 1000
Unique tokens in source (de) vocabulary: 7855
Unique tokens in target (en) vocabulary: 5893
In: <SOS> . schnee den über laufen hunde mittelgroße zwei <EOS>
for loop beam search time: 8.718
Out: Rank-1: <SOS> two medium brown dogs run across the snow . the snow . <EOS>
Out: Rank-2: <SOS> two medium brown dogs run across the snow . <EOS>
Out: Rank-3: <SOS> two medium brown dogs run across the snow . the snow . . <EOS>
Out: Rank-4: <SOS> two medium brown dogs run across the snow . . <EOS>
Out: Rank-5: <SOS> two medium brown dogs run across the snow . snow . <EOS>
Batch beam search time: 2.994
Out: Rank-1: <SOS> two medium brown dogs run across the snow . the snow . <EOS>
Out: Rank-2: <SOS> two medium brown dogs run across the snow . <EOS>
Out: Rank-3: <SOS> two medium brown dogs run across the snow . the snow . . <EOS>
Out: Rank-4: <SOS> two medium brown dogs run across the snow . . <EOS>
Out: Rank-5: <SOS> two medium brown dogs run across the snow . snow . <EOS>
In: <SOS> . <unk> mit tüten gehsteig einem auf verkauft frau eine <EOS>
for loop beam search time: 9.654
Out: Rank-1: <SOS> a woman is selling on her
Out: Rank-2: <SOS> a woman woman selling a
Out: Rank-3: <SOS> a woman is her selling
Out: Rank-4: <SOS> a woman is selling vegetables on a sidewalk
Out: Rank-5: <SOS> a woman woman selling rice
Out: Rank-6: <SOS> a woman is selling her on
Out: Rank-7: <SOS> a woman is selling watermelon on a
Out: Rank-8: <SOS> a woman is selling on the
Out: Rank-9: <SOS> a woman is sells selling
Out: Rank-10: <SOS> a woman selling selling on
Batch beam search time: 3.256
Out: Rank-1: <SOS> a woman is selling on her
Out: Rank-2: <SOS> a woman woman selling a
Out: Rank-3: <SOS> a woman is her selling
Out: Rank-4: <SOS> a woman is selling vegetables on a sidewalk
Out: Rank-5: <SOS> a woman woman selling rice
Out: Rank-6: <SOS> a woman is selling her on
Out: Rank-7: <SOS> a woman is selling watermelon on a
Out: Rank-8: <SOS> a woman is selling on the
Out: Rank-9: <SOS> a woman is sells selling
Out: Rank-10: <SOS> a woman selling selling on
In: <SOS> . bushaltestelle einer an sitzt anzug im mann ein <EOS> <pad> <pad>
for loop beam search time: 10.151
Out: Rank-1: <SOS> a man in a suit is sitting at a bus stop . <EOS>
Out: Rank-2: <SOS> a man in a suit sits at a bus stop . <EOS>
Batch beam search time: 3.383
Out: Rank-1: <SOS> a man in a suit is sitting at a bus stop . <EOS>
Out: Rank-2: <SOS> a man in a suit sits at a bus stop . <EOS>
```

## References
- [C5W3L08 Attention Model, Andrew Ng.](https://www.youtube.com/watch?v=quoGRI-1l0A&list=LLJENudvIT4mHIwNFAMlX29Q&index=2&t=0s)
- https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
