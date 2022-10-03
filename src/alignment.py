from copy import deepcopy

import gensim
import numpy as np


def intersection_align_gensim(m1, m2, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.vocab.keys())
    vocab_m2 = set(m2.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.vocab[w].count + m2.vocab[w].count, reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.syn0norm = m.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(
                index=new_index, count=old_vocab_obj.count
            )
        m.vocab = new_vocab

    return (m1, m2)


def smart_procrustes_re_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes re-align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the re-alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    _base_embed = deepcopy(base_embed)
    _other_embed = deepcopy(other_embed)

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    _base_embed.init_sims(replace=True)
    _other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(
        _base_embed, _other_embed, words=words
    )

    # set anchor words
    base_vocab = list(in_base_embed.vocab.keys())
    base_vocab.sort(key=lambda w: in_base_embed.vocab[w].count, reverse=True)
    base_vocab = base_vocab[:1000]

    # get the (normalized) embedding matrices
    indices = [in_base_embed.vocab[w].index for w in base_vocab]
    base_vecs = np.array([in_base_embed.syn0norm[index] for index in indices])
    other_vecs = np.array([in_other_embed.syn0norm[index] for index in indices])

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho1 = u.dot(v)

    # set anchor words
    other_vocab = list(in_other_embed.vocab.keys())
    other_vocab.sort(key=lambda w: in_other_embed.vocab[w].count, reverse=True)
    other_vocab = other_vocab[:1000]

    # get the (normalized) embedding matrices
    indices = [in_other_embed.vocab[w].index for w in other_vocab]
    base_vecs = np.array([in_base_embed.syn0norm[index] for index in indices])
    other_vecs = np.array([in_other_embed.syn0norm[index] for index in indices])

    # just a matrix dot product with numpy
    m = base_vecs.T.dot(other_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho2 = u.dot(v)

    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    _other_embed.syn0norm = _other_embed.syn0 = (
        (_other_embed.syn0norm).dot(ortho1).dot(ortho2)
    )

    return _other_embed
