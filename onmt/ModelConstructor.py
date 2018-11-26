import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel
from onmt.modules import Embeddings, TransformerEncoder, TransformerDecoder, Generator
from onmt.Utils import use_gpu
from torch.nn.init import xavier_uniform

def load_vectors(vectors_path=None, num=50000, is_cuda=True):
    if not vectors_path:
        return None
    vectors = open(vectors_path, "r")
    next(vectors)
    vecs = []
    count = 0
    while count < num:
        v = next(vectors).split(" ", 1)[1]
        vecs.append(np.fromstring(v, " "))
        count += 1
    assert len(vecs) <= num
    embeddings = np.concatenate(vecs, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if is_cuda else embeddings
    return Variable(embeddings)

def make_embeddings(opt, word_dict, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    return Embeddings(word_vec_size=embedding_dim,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      word_vocab_size=num_word_embeddings)

def make_encoder(opt, embeddings, vecs):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    return TransformerEncoder(opt.enc_layers, opt.rnn_size, opt.dropout, embeddings, vecs)

def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return TransformerDecoder(opt.dec_layers, opt.rnn_size, opt.dropout, embeddings)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    fields = onmt.io.load_fields_from_vocab(checkpoint['vocab'])

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # Make encoder.
    src_dict = fields["src"].vocab
    src_embeddings = make_embeddings(model_opt, src_dict)

    ####### ... Load Global Data ... ######
    vecs = load_vectors(model_opt.vectors, model_opt.max_vec_num, is_cuda=True)
    encoder = make_encoder(model_opt, src_embeddings, vecs)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    tgt_embeddings = make_embeddings(model_opt, tgt_dict, for_encoder=False)
    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)

    # Make Generator.
    generator = Generator(rnn_size=model_opt.rnn_size,
                          target_vocab_len=len(fields["tgt"].vocab))

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        if "0.weight" in checkpoint['generator']:
            checkpoint['generator']["latent.weight"] = checkpoint['generator'].pop("0.weight")
            checkpoint['generator']["latent.bias"] = checkpoint['generator'].pop("0.bias")
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model