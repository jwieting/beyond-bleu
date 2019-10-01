import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('data_and_models/sim/sim.sp.30k.model')
