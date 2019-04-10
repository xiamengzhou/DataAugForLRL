#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts

from copy import deepcopy
import numpy as np
from collections import namedtuple

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)

# Set up the Crayon logging server.
if opt.exp_host != "":
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=opt.exp_host)

    experiments = cc.get_experiment_names()
    print(experiments)
    if opt.exp in experiments:
        cc.remove_experiment(opt.exp)
    experiment = cc.create_experiment(opt.exp)

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

progress_step = 0

def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            # Log the progress using the number of batches on the x-axis.
            report_stats.log_tensorboard(
                "progress", writer, lr, progress_step)
        report_stats = onmt.Statistics()

    return report_stats

def load_vocab(vocab):
    lines = open(vocab, "r").readlines()
    lines = [line.split() for line in lines]
    return [line[0] for line in lines]

class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, switchout=None):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.global_data = {}
        if self.is_train and switchout is not None:
            src_vocab = load_vocab(switchout.src_vocab)
            tgt_vocab = load_vocab(switchout.tgt_vocab)
            src_model = switchout.src_model
            tgt_model = switchout.tgt_model
            tmp = switchout.tmp
            di = switchout.di
            SO = namedtuple("SO", "src_vocab tgt_vocab src_model tgt_model tmp di")
            so = SO(src_vocab=src_vocab, tgt_vocab=tgt_vocab, src_model=src_model,
                    tgt_model=tgt_model, tmp=tmp, di=di)
            self.global_data["so"] = so

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None


    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False, global_data=self.global_data)

from collections import namedtuple
def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        global max_src_in_batch, max_tgt_in_batch

        def batch_size_fn(new, count, sofar):
            global max_src_in_batch, max_tgt_in_batch
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            max_src_in_batch = max(max_src_in_batch,  len(new.src) + 2)
            max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)

    device = opt.gpuid[0] if opt.gpuid else -1
    so = None
    if is_train and opt.switch_out:
        SO = namedtuple("SO", "src_vocab tgt_vocab src_model tgt_model tmp di")
        so = SO(src_vocab=opt.src_vocab, tgt_vocab=opt.tgt_vocab,
                src_model=opt.src_model, tgt_model=opt.tgt_model,
                tmp=opt.tmp, di=opt.di)
    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, so)


def make_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)

    if use_gpu(opt):
        compute.cuda()

    return compute


def validate_while_training(fields, valid_pt=None):
    valid_iter = make_dataset_iter(lazily_load_dataset("valid", valid_pt=valid_pt),
                                   fields, opt, is_train=False)
    return valid_iter

def load_swap_dict(src_field, opt, sep="|||"):
    if opt.swap_dict is not None:
        swap_dict = onmt.modules.SwapDict(swap_dict=opt.swap_dict,
                                          ec_weight=opt.ec_weight_file,
                                          src_field=src_field,
                                          sep=sep,
                                          lrl_prob=opt.lrl_prob,
                                          device=0,
                                          sample_num=opt.sample_num)
        return swap_dict
        # f = open(opt.swap_dict, "r").readlines()
        # a = []
        # b = []
        # for line in f:
        #     tokens = line.strip().split(sep)
        #     tokens1 = tokens[0].strip().split()
        #     tokens2 = tokens[1].strip().split()
        #     a.append([t for t in tokens1])
        #     b.append([t for t in tokens2])
        # a = src_field.process(a, device=-1, train=True)
        # b = src_field.process(b, device=-1, train=True)
        # print("Loading swap dict from {}.".format(opt.swap_dict))
        # return a, b

    else:
        return None

def train_model(model, fields, optim, model_opt, swap_dict):
    train_loss = make_loss_compute(model, fields["tgt"].vocab, opt)
    valid_loss = make_loss_compute(model, fields["tgt"].vocab, opt,
                                   train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size,
                           norm_method, grad_accum_count, opt.select_model)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)



    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = make_dataset_iter(lazily_load_dataset("train"), fields, opt)
        train_stats = trainer.train(train_iter,
                                    epoch,
                                    opt,
                                    fields,
                                    validate_while_training,
                                    writer,
                                    report_func,
                                    opt.valid_pt,
                                    swap_dict)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = make_dataset_iter(lazily_load_dataset("valid", valid_pt=opt.valid_pt),
                                       fields, opt,
                                       is_train=False)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Additional Step. Validate on BLEU.
        # translate.main(True, fields, model, model_opt)


        # 3. Log to remote server.
        # if opt.exp_host:
        #     train_stats.log("train", experiment, optim.lr)
        #     valid_stats.log("valid", experiment, optim.lr)
        # if opt.tensorboard:
        #     train_stats.log_tensorboard("train", writer, optim.lr, epoch)
        #     train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at and epoch % opt.save_interval == 0:
            trainer.drop_checkpoint(model_opt, epoch, deepcopy(fields), valid_stats, 0)



def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)
        print("model dir {} made!".format(model_dirname))


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def lazily_load_dataset(corpus_type, valid_pt=None):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "train_mono"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    if valid_pt is not None:
        yield lazy_dataset_loader(valid_pt, "valid")
    else:
        pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
        if pts:
            for pt in pts:
                yield lazy_dataset_loader(pt, corpus_type)
        else:
            # Only one onmt.io.*Dataset, simple!
            pt = opt.data + '.' + corpus_type + '.pt'
            yield lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, checkpoint, hack_vocab=False):
    if checkpoint is not None:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)

        v = checkpoint['vocab']
        v_names = [f[0] for f in v]
        if not hack_vocab:
            fields = onmt.io.load_fields_from_vocab(v,  dataset.ngram,
                                                    "src_sg" in v_names)
        else:
            v2 = torch.load(opt.data + '.vocab.pt')
            v2_names = [f[0] for f in v2]
            fields = onmt.io.load_fields_from_vocab(v2, dataset.ngram, "src_sg" in v2_names)
    else:
        v = torch.load(opt.data + '.vocab.pt')
        v_names = [f[0] for f in v]
        fields = onmt.io.load_fields_from_vocab(v,
                                                dataset.ngram,
                                                "src_sg" in v_names)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    if not hack_vocab:
        return fields
    else:
        return v[0][1], fields


def build_model(model_opt, opt, fields, checkpoint, old_vocab=None):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint, old_vocab)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
        print('Original LR: {}'.format(optim.optimizer.param_groups[0]['lr']))
        # optim.optimizer.param_groups[0]['lr'] *= opt.lr_am
        # optim.original_lr *= opt.lr_am
        # print('Current LR: {}'.format(optim.original_lr))
        # if opt.finetune:
        #     optim.last_ppl = None
        #     optim.start_decay = False
        #     optim.start_decay_at = None
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.named_parameters())
    return optim


def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
        if opt.finetune:
            model_opt.save_model = opt.save_model
            model_opt.length_penalty = opt.length_penalty
    else:
        checkpoint = None
        model_opt = opt

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train"))

    #### Attributes from data preprocessing ####
    if isinstance(first_dataset.ngram, int):
        model_opt.ngram = first_dataset.ngram
        opt.ngram = first_dataset.ngram
    else:
        first_dataset.ngram = -1
        model_opt.ngram = -1
        opt.ngram = -1

    # Load fields generated from preprocess phase.
    old_vocab = None
    if opt.hack_vocab:
        old_vocab, fields = load_fields(first_dataset, checkpoint, True)
    else:
        fields = load_fields(first_dataset, checkpoint, False)
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint, old_vocab)
    tally_parameters(model)
    check_save_model_path()

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    ###### Load ######:
    swap_dict = load_swap_dict(fields["src"], opt)

    # Do training.
    train_model(model, fields, optim, model_opt, swap_dict=swap_dict)

    # If using tensorboard for logging, close the writer after training.
    if opt.tensorboard:
        writer.close()


if __name__ == "__main__":
    main()
