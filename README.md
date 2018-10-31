## Context-aware Beam Search for Unsupervised Word-by-Word Translation

This code implements a simple beam search where cross-lingual word embedding is combined with a language model. It is compatible with [MUSE](https://github.com/facebookresearch/MUSE) embeddings and [kenlm](https://github.com/kpu/kenlm) language models. The output translation can be further fed to a [denoising autoencoder](https://github.com/yunsukim86/sockeye-noise) for improved reordering.

If you use this code, please cite:

- Yunsu Kim, Jiahui Geng and Hermann Ney. [Improving Unsupervised Word-by-Word Translation Using Language Model and Denoising Autoencoder](https://www-i6.informatik.rwth-aachen.de/publications/download/1075/Kim-EMNLP-2018.pdf). EMNLP 2018.
- Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer and Hervé Jégou. [Word Translation Without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf). arXiv preprint.

If you are looking for the denoising autoencoder, please go to [sockeye-noise](https://github.com/yunsukim86/sockeye-noise).


### Installation

First, please install all dependencies:
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).
* [kenlm](https://github.com/kpu/kenlm) (with Python bindings)

Then clone this repository.


### Usage

Here is a simple example for translation:

```bash
> cat {input_corpus} | python translate.py --src_emb {source_embedding} \
                                           --tgt_emb {target_embedding} \
                                           --emb_dim {embedding_dimension} \
                                           --lm {language_model} > {output_translation}
```

Please refer to help message (`-h`) for other detailed options.
