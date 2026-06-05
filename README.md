# sasrec-yambda

SASRec на PyTorch с нуля, в духе nanoGPT — causal self-attention руками, без `nn.MultiheadAttention`. Обучен на Yandex Yambda (likes, 50M-сабсет).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmmaximov/sasrec-yambda/blob/main/SASRec.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Результаты

Sampled evaluation: 1 настоящий следующий трек против 100 случайных негативов, leave-last-out split, ничьи в ранге делятся честно.

| Модель                | NDCG@10    | HitRate@10 |
|-----------------------|------------|------------|
| `last_item` baseline  | 0.0145     | 0.0145     |
| `popularity` baseline | 0.3975     | 0.6119     |
| **SASRec**            | **0.4237** | **0.6349** |

SASRec обыгрывает popularity-baseline — модель выучила последовательные паттерны, а не просто советует популярное.

## Запуск

`Runtime → Change runtime type → T4 GPU → Run all`. Полный прогон ≈ 2 минуты. Чекпойнты на Google Drive после каждой эпохи — при обрыве Colab повторный `Run all` подхватит последний.

## Архитектура

2 transformer-блока × 2 головы, d_model=64, max_len=200, ~11.7M параметров (~99% — в item-эмбеддингах). AdamW `lr=1e-3`, BCE с одним случайным негативом на позицию, weight tying.

## v1: что не сделано

- Не используются content-фичи (`artist_id`, аудио-эмбеддинги).
- Не обучено на полных `listens` (46.5M событий против 880K в `likes`).
- Sampled evaluation — оптимистичный протокол; на полном каталоге метрики были бы скромнее.

## Ссылки

- Kang & McAuley (2018), *Self-Attentive Sequential Recommendation* — [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
- Yandex Yambda — [huggingface.co/datasets/yandex/yambda](https://huggingface.co/datasets/yandex/yambda)
- Andrej Karpathy, *Let's build GPT* — [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
