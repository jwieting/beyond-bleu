# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from torch.autograd import Variable
import torch.nn.functional as F

from fairseq import bleu, data, sequence_generator, tokenizer, utils
from .fairseq_criterion import FairseqCriterion
import torch
import numpy as np
from .sim_models import WordAveraging
from .sim_utils import Example
from ..spm import sp
import torch._utils
from sacremoses import MosesDetokenizer
from nltk.tokenize import TreebankWordTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class FairseqSequenceCriterion(FairseqCriterion):
    """Base class for sequence-level criterions."""

    def __init__(self, args, dst_dict):
        super().__init__()
        self.args = args
        self.dst_dict = dst_dict
        self.pad_idx = dst_dict.pad()
        self.eos_idx = dst_dict.eos()
        self.unk_idx = dst_dict.unk()

        self.length_penalty = args.length_penalty

        self._generator = None
        self._scorer = None

        if self.args.seq_score_alpha > 0:
            self.load_model()

    #
    # Methods to be defined in sequence-level criterions
    #

    def prepare_sample_and_hypotheses(self, model, sample, hypos):
        """Apply criterion-specific modifications to the given sample/hypotheses."""
        return sample, hypos

    def sequence_forward(self, net_output, model, sample):
        """Compute the sequence-level loss for the given hypotheses.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)

    #
    # Helper methods
    #

    def add_reference_to_hypotheses(self, sample, hypos):
        """Add the reference translation to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_reference' in sample:
            return hypos
        sample['includes_reference'] = True

        target = sample['target'].data
        for i, hypos_i in enumerate(hypos):
            # insert reference as first hypothesis
            ref = utils.lstrip_pad(target[i, :], self.pad_idx)
            hypos_i.insert(0, {
                'tokens': ref,
                'score': None,
            })
        return hypos

    def add_bleu_to_hypotheses(self, sample, hypos):
        """Add BLEU scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """
        if 'includes_bleu' in sample:
            return hypos
        sample['includes_bleu'] = True

        if self._scorer is None:
            self.create_sequence_scorer()

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.lstrip_pad(target[i, :], self.pad_idx).cpu()
            r = self.dst_dict.string(ref, bpe_symbol='@@ ', escape_unk=True)
            r = tokenizer.Tokenizer.tokenize(r, self.dst_dict, add_if_not_exist=True)
            for hypo in hypos_i:
                h = self.dst_dict.string(hypo['tokens'].int().cpu(), bpe_symbol='@@ ')
                h = tokenizer.Tokenizer.tokenize(h, self.dst_dict, add_if_not_exist=True)
                # use +1 smoothing for sentence BLEU
                hypo['bleu'] = self._scorer.score(r, h)
        return hypos

    def load_model(self):
        """Load SIM model.
        """
        model = torch.load(self.args.sim_model_file,
                               map_location='cpu')

        state_dict = model['state_dict']
        vocab_words = model['vocab_words']
        args = model['args']

        #turn off gpu
        args.gpu = -1

        self.model = WordAveraging(args, vocab_words)
        self.model.load_state_dict(state_dict, strict=True)

    def add_sim_to_hypotheses(self, sample, hypos):
        """Add SIM scores to the set of hypotheses.

        This can be called from prepare_sample_and_hypotheses.
        """

        detok = MosesDetokenizer('en')
        tok = TreebankWordTokenizer()

        if 'includes_sim' in sample:
            return hypos
        sample['includes_sim'] = True

        if(next(self.model.parameters()).is_cuda):
            self.model.cpu()

        def make_example(sentence):
            sentence = detok.detokenize(sentence.split())
            sentence = sentence.lower()
            sentence = " ".join(tok.tokenize(sentence))
            sentence = sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence))
            wp1.populate_embeddings(self.model.vocab)
            return wp1

        target = sample['target'].data.int()
        for i, hypos_i in enumerate(hypos):
            ref = utils.lstrip_pad(target[i, :], self.pad_idx).cpu()
            ref = self.dst_dict.string(ref, bpe_symbol='@@ ', escape_unk=True)
            ref_e = make_example(ref)
            xh = []
            xr = []
            for hypo in hypos_i:
                hyp = self.dst_dict.string(hypo['tokens'].int().cpu(), bpe_symbol='@@ ')
                hyp_e = make_example(hyp)
                xr.append(ref_e)
                xh.append(hyp_e)
            wx1, wl1, wm1 = self.model.torchify_batch(xr)
            wx2, wl2, wm2 = self.model.torchify_batch(xh)
            scores = self.model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)

            for j, hypo in enumerate(hypos_i):
                hyp_l = wl1[j]
                ref_l = wl2[j]
                lp = np.exp(1 - max(ref_l, hyp_l) / float(min(ref_l, hyp_l)))
                hypo['sim'] = lp ** self.length_penalty * scores.data[j]

        return hypos

    def create_sequence_scorer(self):
        if self.args.seq_scorer == "bleu":
            self._scorer = BleuScorer(self.pad_idx, self.eos_idx, self.unk_idx)
        else:
            raise Exception("Unknown sequence scorer {}".format(self.args.seq_scorer))

    def get_hypothesis_scores(self, net_output, sample, score_pad=False):
        """Return a tensor of model scores for each hypothesis.

        The returned tensor has dimensions [bsz, nhypos, hypolen]. This can be
        called from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        hypotheses = Variable(sample['hypotheses'], requires_grad=False).view(bsz, nhypos, hypolen, 1)
        scores = net_output.gather(3, hypotheses)
        if not score_pad:
            scores = scores * hypotheses.ne(self.pad_idx).float()
        return scores.squeeze(3)

    def get_hypothesis_lengths(self, net_output, sample):
        """Return a tensor of hypothesis lengths.

        The returned tensor has dimensions [bsz, nhypos]. This can be called
        from sequence_forward.
        """
        bsz, nhypos, hypolen, _ = net_output.size()
        lengths = sample['hypotheses'].view(bsz, nhypos, hypolen).ne(self.pad_idx).sum(2).float()
        return Variable(lengths, requires_grad=False)

    #
    # Methods required by FairseqCriterion
    #

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_state = model.training
        if model_state:
            model.train(self.args.seq_hypos_dropout)

        # generate hypotheses
        hypos = self._generate_hypotheses(model, sample)

        model.train(model_state)

        # apply any criterion-specific modifications to the sample/hypotheses
        sample, hypos = self.prepare_sample_and_hypotheses(model, sample, hypos)

        # create a new sample out of the hypotheses
        sample = self._update_sample_with_hypos(sample, hypos)


        # run forward and get sequence-level loss
        net_output = self.get_net_output(model, sample)
        loss, sample_size, logging_output = self.sequence_forward(net_output, model, sample)

        return loss, sample_size, logging_output

    def get_net_output(self, model, sample):
        """Return model outputs as log probabilities."""
        net_output = model(**sample['net_input'])
        return F.log_softmax(net_output, dim=1).view(
            sample['bsz'], sample['num_hypos_per_batch'], -1, net_output.size(1))


    def _generate_hypotheses(self, model, sample):
        args = self.args

        # initialize generator
        if self._generator is None:
            self._generator = sequence_generator.SequenceGenerator(
                [model], self.dst_dict, unk_penalty=args.seq_unkpen, sampling=args.seq_sampling)
            self._generator.cuda()

        # generate hypotheses
        input = sample['net_input']
        srclen = input['src_tokens'].size(1)
        hypos = self._generator.generate(
            input['src_tokens'], input['src_positions'],
            maxlen=int(args.seq_max_len_a*srclen + args.seq_max_len_b),
            beam_size=args.seq_beam)

        # add reference to the set of hypotheses
        if self.args.seq_keep_reference:
            hypos = self.add_reference_to_hypotheses(sample, hypos)

        return hypos

    def _update_sample_with_hypos(self, sample, hypos):
        num_hypos_per_batch = len(hypos[0])
        assert all(len(h) == num_hypos_per_batch for h in hypos)

        def repeat_num_hypos_times(t):
            return t.repeat(1, num_hypos_per_batch).view(num_hypos_per_batch*t.size(0), t.size(1))

        input = sample['net_input']
        bsz = input['src_tokens'].size(0)
        input['src_tokens'].data = repeat_num_hypos_times(input['src_tokens'].data)
        input['src_positions'].data = repeat_num_hypos_times(input['src_positions'].data)

        input_hypos = [h['tokens'] for hypos_i in hypos for h in hypos_i]
        sample['hypotheses'] = data.LanguagePairDataset.collate_tokens(
            input_hypos, self.pad_idx, self.eos_idx, left_pad=True, move_eos_to_beginning=False)
        input['input_tokens'].data = data.LanguagePairDataset.collate_tokens(
            input_hypos, self.pad_idx, self.eos_idx, left_pad=True, move_eos_to_beginning=True)
        input['input_positions'].data = data.LanguagePairDataset.collate_positions(
            input_hypos, self.pad_idx, left_pad=True)

        sample['target'].data = repeat_num_hypos_times(sample['target'].data)
        sample['ntokens'] = sample['target'].data.ne(self.pad_idx).sum()
        sample['bsz'] = bsz
        sample['num_hypos_per_batch'] = num_hypos_per_batch
        return sample


class BleuScorer(object):

    def __init__(self, pad, eos, unk):
        self._scorer = bleu.Scorer(pad, eos, unk)

    def score(self, ref, hypo):
        self._scorer.reset(one_init=True)
        self._scorer.add(ref, hypo)
        return self._scorer.score()

