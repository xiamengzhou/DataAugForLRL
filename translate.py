#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

import torch.multiprocessing as mp

from datetime import datetime


class Time:
    def __init__(self):
        self.start_time = datetime.now()

    def timeit(self, task):
        time_elapsed = datetime.now() - self.start_time
        print('{} done. Time elapsed (hh:mm:ss.ms) {}'.format(task, time_elapsed))



def tok(s):
    return s.split()

class SimpleStatistic:
    def __init__(self):
        self.increase_ratio = 0
        self.n = 0

    def update(self, ratio):
        self.increase_ratio += ratio
        self.n += 1

    def log_tensorboard(self, writer, step):
        writer.add_scalar("translation/increase_ratio", self.increase_ratio / self.n, step)


def _report_score(name, score_total, words_total, writer, step, corpus_type):
    avg_score = score_total / words_total
    ppl = math.exp(-score_total / words_total)
    if writer is not None:
        writer.add_scalar("translation/{}/{}/avg_score".format(name, corpus_type), avg_score, step)
        writer.add_scalar("translation/{}/{}/ppl".format(name, corpus_type), ppl, step)
    print("%s %s AVG SCORE: %.4f, %s PPL: %.4f" % (name, corpus_type, avg_score, name, ppl))



def _report_single_source_bleu(opt, output, writer, step, corpus_type):
    import subprocess
    path = os.path.split(os.path.realpath(__file__))[0]
    print()
    res = subprocess.check_output(
        "perl %s/tools/multi-bleu.perl %s < %s"
        % (path, "{}.nonbpe".format(opt.tgt), output),
        shell=True).decode("utf-8")
    res = res.strip()
    weighted_bleu, bleus = extract_single_source_bleu(res)
    if writer is not None:
        writer.add_scalar("translation/single_source/{}/weighted_bleu".format(corpus_type), weighted_bleu, step)
        writer.add_scalar("translation/single_source/{}/bleu_1_gram".format(corpus_type), bleus[0], step)
        writer.add_scalar("translation/single_source/{}/bleu_2_gram".format(corpus_type), bleus[1], step)
        writer.add_scalar("translation/single_source/{}/bleu_3_gram".format(corpus_type), bleus[2], step)
        writer.add_scalar("translation/single_source/{}/bleu_4_gram".format(corpus_type), bleus[3], step)
    print(">> " + res.strip())
    return weighted_bleu


def extract_single_source_bleu(res):
    res = res[7:]
    end = res.find("(")
    res = res[:end].strip()
    comma = res.find(",")
    weighted_bleu = float(res[:comma])
    bleus = res[comma+1:].strip().split("/")
    bleus = [float(bleu) for bleu in bleus]
    return weighted_bleu, bleus


def _report_multi_source_bleu(output, writer, step, corpus_type):
    import subprocess
    path = os.path.split(os.path.realpath(__file__))[0]
    print()
    sgm_file = output[:-3] + "sgm"
    subprocess.check_output("python %s/tools/plain2sgm.py %s %s/tools/nist06_src.sgm %s"
                   % (path, output, path, sgm_file), shell=True)
    res = subprocess.check_output("%s/tools/mteval-v11b.pl -t %s -s "
                                  "%s/tools/nist06_src.sgm -r %s/tools/nist06_ref.sgm"
                                  % (path, sgm_file, path, path), shell=True).decode("utf-8")
    bleu = extract_multi_source_bleu(res)
    if writer is not None:
        writer.add_scalar("translation/multi_source/{}/bleu_4_gram".format(corpus_type), bleu, step)
    print("multisource_bleu_4_gram: {}".format(bleu))
    return bleu


def extract_multi_source_bleu(res):
    start = res.find("BLEU")
    end = res.find("for system")
    bleu = float(res[start:end].split()[-1])
    return bleu

def _report_rouge(opt):
    import subprocess
    path = os.path.split(os.path.realpath(__file__))[0]
    res = subprocess.check_output(
        "python3 %s/tools/test_rouge.py -r %s -c %s"
        % (path, opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())
    return res.strip()


def main(training=False, fields=None, model=None, opt=None, writer=None,  step=0, corpus_type="dev", multi_process=False):
    time = Time()
    if training:
        assert fields is not None
        assert model is not None
        assert opt is not None
        model.eval()
        model.generator.eval()
        opt.cuda = opt.gpu > -1
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)
        out_file = codecs.open("{}_{}_pred_{}.txt".format(opt.save_model,
                                                          corpus_type.replace("/", "_"),
                                                          str(step)), "w", "utf-8")
        print("Output file: ", out_file.name)
        copy_attn = opt.copy_attn
        model_opt = opt
    else:
        # Load the model.
        parser = argparse.ArgumentParser(
            description='translate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opts.add_md_help_argument(parser)
        opts.translate_opts(parser)

        opt = parser.parse_args()
        dummy_parser = argparse.ArgumentParser(description='train.py')
        opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]

        opt.cuda = opt.gpu > -1
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)

        fields, model, model_opt = \
            onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

        out_file = codecs.open(opt.output, 'w', 'utf-8')
    assert opt.tgt is None
    data = onmt.io.build_dataset(fields, opt.src, opt.tgt, use_filter_pred=False)
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".


    data_iter = onmt.io.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.translate_batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)
    output, pred_score_total, pred_words_total = \
            translate_single_process(opt, model, fields, data, data_iter, f=out_file)
    outfile_name = out_file.name

    if opt.bpe:
        import subprocess
        subprocess.check_output("sed 's/\@\@ //g' < {} > {}".format(outfile_name, outfile_name + ".nonbpe"), shell=True)
        outfile_name = outfile_name + ".nonbpe"
    if opt.new_bpe:
        generate_nonbpe(outfile_name)
        outfile_name = outfile_name + ".nonbpe"
    # if writer is not None:
    #     ratio_stats.log_tensorboard(writer, step)
    # _report_score('PRED', pred_score_total, pred_words_total, writer, step, corpus_type)
    metric = 0
    if opt.tgt:
        # _report_score('GOLD', gold_score_total, gold_words_total, writer, step, corpus_type)
        if opt.report_single_bleu:
            metric = _report_single_source_bleu(opt, outfile_name, writer, step, corpus_type)
        if opt.report_multi_bleu:
            metric = _report_multi_source_bleu(outfile_name, writer, step, corpus_type)
        if opt.report_rouge:
            metric = _report_rouge(opt)

    # if opt.dump_beam:
    #     import json
    #     json.dump(translator.beam_accum,
    #               codecs.open(opt.dump_beam, 'w', 'utf-8'))

    time.timeit(task="Translation Testing")
    return metric


def generate_nonbpe(file):
    f = open(file, "r")
    f2 = open("{}.nonbpe".format(file), "w")
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        token = ""
        tokens_ = []
        for i, t in enumerate(tokens):
            if t[0] == "▁":
                if token != "":
                    tokens_.append(token)
                token = t[1:]
            else:
                token += t
            if i == len(tokens) - 1:
                tokens_.append(token)
                break
        new_line = " ".join(tokens_)
        f2.write(new_line + "\n")


def translate_single_process(opt, model, fields, data,
                             data_iter, start_index=0, result=None, f=None):
    end_index = start_index + len(data)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             -0.,
                                             None,
                                             opt.length_penalty)
    translator = onmt.translate.Translator(
        model, fields["tgt"].vocab,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length,
        stepwise_penalty=opt.stepwise_penalty)
    builder = onmt.translate.TranslationBuilder(
        data, translator.tgt_vocab,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0


    index = iter(range(start_index, end_index))
    output = {}
    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent) + 1

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            if f is not None:
                f.write('\n'.join(n_best_preds))
                f.write('\n')
                f.flush()

            output[next(index)] = '\n'.join(n_best_preds) + '\n'

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

    if result is not None:
        result.put(output)
    return output, pred_score_total, pred_words_total


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    import time
    start = time.time()
    main()
    end = time.time()
    minute = (end - start) // 60
    second = (end - start) % 60
    print("Time taken to run the translate scripts is {} mins {} secs".format(minute, second))
