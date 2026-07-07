---
title: "Attention dans seq2seq"
tags: [attention, seq2seq, rnn]
niveau: intermédiaire
source: "Notes utilisateur + Jalammar seq2seq attention + PyTorch seq2seq tutorial"
last_review: 2026-07-06
---

# Attention dans seq2seq

## Idée en 1 phrase
L’attention permet au decoder de regarder plusieurs hidden states de l’encoder au lieu d’utiliser seulement le dernier.

## Prérequis
- Encoder seq2seq
- Decoder seq2seq
- Hidden states

## À quoi ça sert ?
- Réduire le goulot d’étranglement du dernier hidden state.
- Aider le decoder à se concentrer sur les bons mots source.

## Explication simple
Sans attention, le decoder reçoit surtout un seul résumé final.
Avec attention, il peut consulter les états produits à chaque mot de l’input.

## Explication technique
L’encoder produit plusieurs hidden states :

```txt
[h1, h2, h3, h4]
```

À chaque étape, le decoder calcule des poids d’attention sur ces états, puis construit un contexte adapté.

## Mini exemple

```python
encoder_outputs = [h1, h2, h3, h4]
weights = attention(decoder_h, encoder_outputs)
context = weighted_sum(weights, encoder_outputs)
```

## Questions pour me tester
1. Pourquoi le dernier hidden state peut-il être insuffisant ?
2. Que regarde le decoder avec l’attention ?
3. Pourquoi l’attention est-elle utile pour les longues phrases ?

## Erreurs fréquentes
- Croire que l’attention remplace forcément le RNN.
- Croire que l’attention utilise seulement le dernier hidden state.

## Liens liés
- [[rnn-encoder-seq2seq]]
- [[outputs-vs-hidden-pytorch-rnn]]
