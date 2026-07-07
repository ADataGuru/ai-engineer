---
title: "Decoder seq2seq simple"
tags: [seq2seq, decoder, rnn]
niveau: intermédiaire
source: "Notes utilisateur + PyTorch seq2seq tutorial"
last_review: 2026-07-06
---

# Decoder seq2seq simple

## Idée en 1 phrase
Le decoder génère une séquence de sortie mot par mot à partir du contexte donné par l’encoder.

## Prérequis
- Encoder seq2seq
- Hidden state
- Token spécial `<SOS>`

## À quoi ça sert ?
- Générer une traduction.
- Produire une séquence de sortie étape par étape.

## Explication simple
L’encoder donne un résumé de la phrase source.
Le decoder utilise ce résumé comme mémoire de départ, puis prédit les mots de sortie un par un.

## Explication technique
Dans un seq2seq simple, le dernier hidden state de l’encoder initialise le hidden state du decoder.
À chaque étape, le decoder reçoit le token précédent et son hidden state courant.

## Mini exemple

```python
decoder_h = encoder_context
decoder_input = "<SOS>"

for _ in range(max_len):
    x = embedding(decoder_input)
    decoder_h = rnn_cell(x, decoder_h)
    decoder_input = predict_next_token(decoder_h)
```

## Questions pour me tester
1. Avec quoi initialise-t-on le hidden state du decoder ?
2. Pourquoi utilise-t-on `<SOS>` au début ?
3. Pourquoi le decoder génère-t-il les mots un par un ?

## Erreurs fréquentes
- Croire que le decoder reçoit toute la phrase source directement.
- Oublier que sa prédiction précédente peut devenir l’entrée suivante.

## Liens liés
- [[rnn-encoder-seq2seq]]
- [[attention-seq2seq]]
