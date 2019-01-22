from __future__ import division
import torch.nn as nn

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None, ngram_input=None):
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths, ngram_input=ngram_input)

        enc_state = \
            self.decoder.init_decoder_state(lengths)

        decoder_outputs, dec_state, attns = \
                self.decoder(tgt, memory_bank,
                             enc_state if dec_state is None
                             else dec_state,
                             memory_lengths=lengths)

        return decoder_outputs, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        # usually layer_num * (beam_size * bs) * 1 * dim
        for e in self._all:
            sizes = e.size()
            if len(sizes) == 1:
                br = sizes[0]
                sent_states = e.view(beam_size, br // beam_size)[:, idx]
                sent_states.copy_(
                    sent_states.index_select(0, positions))
            elif len(sizes) == 4:
                br = sizes[1]
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]
                sent_states.data.copy_(
                    sent_states.data.index_select(1, positions))
            else:
                br = sizes[1]
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2])[:, :, idx]

                sent_states.data.copy_(
                    sent_states.data.index_select(1, positions))


