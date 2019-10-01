import os
import torch
import sentencepiece as spm
import argparse
import numpy as np
from fairseq.criterions.sim_utils import Example
from fairseq.criterions.sim_models import WordAveraging
from sacremoses import MosesDetokenizer
from nltk.tokenize import TreebankWordTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--sim-model-file', default="data_and_models/sim/sim.pt",
                   help='Model file for SIM.')
parser.add_argument('--length-penalty', type=float, default=0.25, metavar='D',
                   help='Weight of length penalty on SIM term.')
parser.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                   help='source language')
parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                   help='target language')
parser.add_argument('--data', metavar='DIR',
                   help='path to data directory')
args = parser.parse_args()

def make_example(sentence, detok, tok, sp):
    sentence = detok.detokenize(sentence.split())
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    return " ".join(sentence)

def score_output(args, fname):
    sp = spm.SentencePieceProcessor()
    sp.Load('data_and_models/sim/sim.sp.30k.model')

    detok = MosesDetokenizer('en')
    tok = TreebankWordTokenizer()

    f = open(fname,'r')
    lines = f.readlines()

    pairs = []
    pairs_bleu = []
    src = None

    for i in lines:
        if i[0] == "T":
            target = i.split()[1:]
            target = " ".join(target).replace("@@ ","")
            target_bleu = target
            target_sim = make_example(target, detok, tok, sp)
        elif i[0] == "H":
            hyp = i.split()[2:]
            hyp = " ".join(hyp).replace("@@ ","")
            hyp_bleu = hyp
            hyp_sim = make_example(hyp, detok, tok, sp)
        elif i[0] == "S":
            if src is not None:
                pairs.append((target_sim, hyp_sim, src_sim))
                pairs_bleu.append((target_bleu, hyp_bleu, src_bleu))
            src = i.split()[1:]
            src = " ".join(src).replace("@@ ","")
            src_bleu = src
            src_sim = make_example(src, detok, tok, sp)

    pairs.append((target_sim, hyp_sim, src_sim))
    pairs_bleu.append((target_bleu, hyp_bleu, src_bleu))

    model = torch.load(args.sim_model_file,
                           map_location='cpu')

    state_dict = model['state_dict']
    vocab_words = model['vocab_words']
    sim_args = model['args']
    model = WordAveraging(sim_args, vocab_words)
    model.load_state_dict(state_dict, strict=True)

    scores = []
    scores_simile = []
    for i in pairs:
        wp1 = Example(i[0])
        wp1.populate_embeddings(model.vocab)
        wp2 = Example(i[1])
        wp2.populate_embeddings(model.vocab)
        wx1, wl1, wm1 = model.torchify_batch([wp1])
        wx2, wl2, wm2 = model.torchify_batch([wp2])
        score = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        ref_l = len(i[0])
        hyp_l = len(i[1])
        lp = np.exp(1 - max(ref_l, hyp_l) / float(min(ref_l, hyp_l)))
        simile = lp ** args.length_penalty * score.data[0]
        scores_simile.append(simile)
        scores.append(score.data[0])

    print("SIM: {0}".format(np.mean(scores)))
    print("SimiLe: {0}".format(np.mean(scores_simile)))

    fout = open(fname + ".target.out", "w")
    for i in pairs_bleu:
        fout.write(i[0].strip()+"\n")
    fout.close()

    fout = open(fname + ".hyp.out", "w")
    for i in pairs_bleu:
        fout.write(i[1].strip()+"\n")
    fout.close()

    fout = open(fname + ".src.out", "w")
    for i in pairs_bleu:
        fout.write(i[2].strip()+"\n")
    fout.close()

    cmd = "perl multi-bleu.perl {0} < {1}".format(fname + ".target.out", fname + ".hyp.out")
    os.system(cmd)

cmd = "python -u generate.py {0} -s {1} -t {2} --path {3}/checkpoint_best.pt  " \
      "--gen-subset valid > {3}/dev_out.txt".format(args.data, args.source_lang, args.target_lang, args.save_dir)
os.system(cmd)
score_output(args, args.save_dir + "/dev_out.txt")

cmd = "python -u generate.py {0} -s {1} -t {2} --path {3}/checkpoint_best.pt " \
      "--gen-subset test > {3}/test_out.txt".format(args.data, args.source_lang, args.target_lang, args.save_dir)
os.system(cmd)
score_output(args, args.save_dir + "/test_out.txt")
