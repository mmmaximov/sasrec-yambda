# sasrec-yambda
SASRec на PyTorch, обученный с нуля на Yandex Yambda (likes, 50M-сабсет). Реализация в духе nanoGPT: causal self-attention руками, без nn.MultiheadAttention. На тесте обыгрывает popularity-baseline (NDCG@10 0.424 vs 0.398). Тренировка и оценка укладываются в бесплатный Colab T4.
