from __future__ import division

import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.modules

import translate
from copy import deepcopy
from os.path import join
from collections import defaultdict

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.emb_loss = 0

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def update_emb_loss(self, emb_loss):
        self.emb_loss += emb_loss

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: %6.2f; emb_loss: %6.2f;" +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               self.emb_loss,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)

class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", grad_accum_count=1, select_model="ppl"):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.progress_step = 0

        self.select_model = select_model
        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, opt, fields,
              validate_while_training, writer, report_func=None, valid_pt=None,
              swap_dict=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        max_metric = 0
        min_metric = 100

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                emb_loss = self.pass_constraint(swap_dict=swap_dict, report_stats=report_stats,
                                                total_stats=total_stats)
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization, emb_loss)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            self.progress_step,
                            total_stats.start_time, self.optim.lr,
                            report_stats)
                    self.progress_step += 1

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

            if i % opt.bleu_freq == 0 and i > 0:
                valid_iter = validate_while_training(fields, valid_pt)
                valid_stats = self.validate(valid_iter)
                if epoch > opt.save_cutoff:
                    if self.select_model == "bleu":
                        opt.src = join(opt.src)
                        opt.tgt = join(opt.tgt)
                        opt.tm_pieces = False
                        metric = translate.main(training=True, fields=fields, model=self.model, opt=opt,
                                                writer=writer, step=self.progress_step, corpus_type="dev/normal_decoding")
                        if metric > max_metric - 0.1:
                            self.drop_checkpoint(opt, epoch, deepcopy(fields), valid_stats, self.progress_step)
                            if metric > max_metric:
                                max_metric = metric
                    elif self.select_model == "ppl":
                        valid_stats.log_tensorboard("valid", writer, self.optim.lr, self.progress_step)
                        metric = valid_stats.ppl()
                        if metric < min_metric + 0.1:
                            self.drop_checkpoint(opt, epoch, deepcopy(fields), valid_stats, self.progress_step)
                            if metric < min_metric:
                                min_metric = metric
                self.model.train()
                self.model.generator.train()


        if len(true_batchs) > 0:
            emb_loss = self.pass_constraint(swap_dict=swap_dict, report_stats=report_stats,
                                            total_stats=total_stats)
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization, emb_loss)
            true_batchs = []

        return total_stats


    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            tgt = batch.tgt
            tgt = tgt.unsqueeze(-1)
            src, src_lengths = batch.src
            if not isinstance(src, list):
                src = src.unsqueeze(-1)
                outputs, attns = \
                    self.pass_model(src, tgt, src_lengths, ngram_input=None)
            else:
                outputs, attns = \
                    self.pass_model(None, tgt, src_lengths, ngram_input=src)

            # F-prop through the model.


            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def pass_model(self, src, tgt, src_lengths, ngram_input):
        outputs, attns, _ = self.model(src, tgt, src_lengths, None, ngram_input=ngram_input)
        return outputs, attns

    def pass_constraint(self, swap_dict, report_stats, total_stats):
        if swap_dict is not None:
            s, w = swap_dict.sample()
            emb_loss = self.model.encoder.embeddings.embedding_constraint(w, s)
            report_stats.update_emb_loss(emb_loss.data[0])
            total_stats.update_emb_loss(emb_loss.data[0])
            return emb_loss
        else:
            return None

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats, step=None):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'step': step
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d_s%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch, step))
        print('%s_acc_%.2f_ppl_%.2f_e%d_s%d.pt' % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch, step), "saved!")

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats, normalization,
                               emb_loss=None):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src, src_lengths = batch.src
            ngram_input = None
            if not isinstance(src, list):
                src = src.unsqueeze(-1)
            else:
                ngram_input = src
                src = None
            report_stats.n_src_words += src_lengths.sum()

            tgt_outer = batch.tgt
            tgt_outer = tgt_outer.unsqueeze(-1)


            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                outputs, attns = \
                    self.pass_model(src, tgt, src_lengths, ngram_input)

                if emb_loss is not None:
                    emb_loss.backward(retain_graph=True)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()
