import argparse


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('-src_word_vec_size', type=int, default=500,
                       help='Word embedding size for src.')
    group.add_argument('-tgt_word_vec_size', type=int, default=500,
                       help='Word embedding size for tgt.')
    group.add_argument('-word_vec_size', type=int, default=-1,
                       help='Word embedding size for src and tgt.')

    # Encoder-Deocder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument('-layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    group.add_argument('-rnn_size', type=int, default=500,
                       help='Size of rnn hidden states')


    # Model Selection
    group = parser.add_argument_group('Model- Selection')
    group.add_argument("-select_model")
    group.add_argument("-save_cutoff", type=int, default=10)


def preprocess_opts(parser):
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('-train_src', required=True,
                       help="Path to the training source data")
    group.add_argument('-train_tgt', required=True,
                       help="Path to the training target data")
    group.add_argument('-valid_src', required=True,
                       help="Path to the validation source data")
    group.add_argument('-valid_tgt', required=True,
                       help="Path to the validation target data")
    group.add_argument('-src_vocab_file', help="source vocabulary file")
    group.add_argument('-tgt_vocab_file', help="target vocabulary file")

    group.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    group.add_argument('-max_shard_size', type=int, default=0,
                       help="""For text corpus of large volume, it will
                       be divided into shards of this size to preprocess.
                       If 0, the data will be handled as a whole. The unit
                       is in bytes. Optimal value should be multiples of
                       64 bytes.""")

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab',
                       help="Path to an existing source vocabulary")
    group.add_argument('-tgt_vocab',
                       help="Path to an existing target vocabulary")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument('-src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    group.add_argument('-src_seq_length_trunc', type=int, default=0,
                       help="Truncate source sequence length.")
    group.add_argument('-tgt_seq_length', type=int, default=50,
                       help="Maximum target sequence length to keep.")
    group.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                       help="Truncate target sequence length.")
    group.add_argument('-lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('-shuffle', type=int, default=1,
                       help="Shuffle data")
    group.add_argument('-seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")

    group.add_argument('-cover', default="standard")


def train_opts(parser):
    # Model loading/saving options
    group = parser.add_argument_group('General')
    group.add_argument('-data', default="data/news/news",
                       help="""Path prefix to the ".train.pt" and
                       ".valid.pt" and ".test.pt" file path from preprocess.py""")

    group.add_argument('-save_model', default='model',
                       help="""Model filename (the model will be saved as
                       <save_model>_epochN_PPL.pt where PPL is the
                       validation perplexity""")
    # GPU
    group.add_argument('-gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")

    group.add_argument('-seed', type=int, default=-1,
                       help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('-start_epoch', type=int, default=1,
                       help='The epoch from which to start')
    group.add_argument('-param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('-param_init_glorot', action='store_true',
                       help="""Init parameters with xavier_uniform.
                       Required for transfomer.""")

    group.add_argument('-train_from', default='', type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('-pre_word_vecs_enc',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add_argument('-pre_word_vecs_dec',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('-fix_word_vecs_enc',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('-fix_word_vecs_dec',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('-batch_size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('-batch_type', default='sents',
                       choices=["sents", "tokens"],
                       help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add_argument('-normalization', default='sents',
                       choices=["sents", "tokens"],
                       help='Normalization method of the gradient.')
    group.add_argument('-accum_count', type=int, default=1,
                       help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
    group.add_argument('-valid_batch_size', type=int, default=32,
                       help='Maximum batch size for validation')
    group.add_argument('-max_generator_batches', type=int, default=32,
                       help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
    group.add_argument('-epochs', type=int, default=13,
                       help='Number of training epochs')
    group.add_argument('-optim', default='sgd',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam',
                                'sparseadam'],
                       help="""Optimization method.""")
    group.add_argument('-max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add_argument('-dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('-truncated_decoder', type=int, default=0,
                       help="""Truncated bptt.""")
    group.add_argument('-adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument('-adam_beta2', type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add_argument('-label_smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('-learning_rate', type=float, default=1.0,
                       help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add_argument('-learning_rate_decay', type=float, default=0.5,
                       help="""If update_learning_rate, decay learning rate by
                       this much if (i) perplexity does not decrease on the
                       validation set or (ii) epoch has gone past
                       start_decay_at""")
    group.add_argument('-start_decay_at', type=int, default=8,
                       help="""Start decaying every epoch after and including this
                       epoch""")
    group.add_argument('-start_checkpoint_at', type=int, default=0,
                       help="""Start checkpointing every epoch after and including
                       this epoch""")
    group.add_argument('-decay_method', type=str, default="",
                       choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")

    # validation during trining
    group = parser.add_argument_group("Validation- Rate")
    group.add_argument("-valid_freq", type=int, default=500,
                       help="Frequency to do validation on dev set.")
    group.add_argument("-bleu_freq", type=int, default=2000,
                       help="Frequency to test bleu on dev set.")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('-exp_host', type=str, default="",
                       help="Send logs to this crayon server.")
    group.add_argument('-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add_argument('-tensorboard', action="store_true",
                       help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str,
                       default="out/runs/onmt",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)
    group.add_argument('-valid_pt', type=str, default="")


def translate_opts(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('-model', help='Path to model .pt file')

    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="Type of the source input. Options: [text|img].")
    group.add_argument('-bpe', action='store_true')
    group.add_argument('-new_bpe', action='store_true')
    group.add_argument('-tr_dir', default="",
                       help="When the translation script is invoked from training process,"
                            "we use tr_dir to access all files manually.")
    group.add_argument('-src', default="",
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-tgt', help='True target sequence (optional)')
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_single_bleu', action='store_true',
                       help="""Report single source bleu score after translation,
                       call tools/multi-bleu.perl on command line""")

    group = parser.add_argument_group('Beam')
    group.add_argument('-beam_size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=0,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=100,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='none',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-translate_batch_size', type=int, default=30,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=-1,
                       help="Device to run on")


def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self)\
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
