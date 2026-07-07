---
title: "Cellule RNN"
tags: [rnn, sequence, deep-learning]
niveau: base
source: "Notes utilisateur + PyTorch nn.RNN"
last_review: 2026-07-06
---

# Cellule RNN

## Idée en 1 phrase
Une cellule RNN lit une entrée actuelle et une mémoire précédente pour produire une nouvelle mémoire.

## Prérequis
- Vecteurs
- Multiplication matricielle
- Fonction d’activation

## À quoi ça sert ?
- Traiter une séquence élément par élément.
- Réutiliser l’information des étapes précédentes.

## Explication simple
Un RNN ne regarde pas seulement l’élément actuel.
Il garde aussi une mémoire de ce qu’il a déjà lu.

## Explication technique
À l’étape `t`, la cellule calcule :

```txt
h_t = f(x_t, h_{t-1})
```

`x_t` est l’entrée actuelle. `h_{t-1}` est l’ancienne mémoire. `h_t` est la nouvelle mémoire.

## Mini exemple

```python
h = 0  # mémoire initiale

for x in sequence:
    h = rnn_cell(x, h)  # nouvelle mémoire
```

## Questions pour me tester
1. Pourquoi un RNN a-t-il besoin d’une mémoire ?
2. Que représente `h_{t-1}` ?
3. Qu’est-ce qui est réutilisé à chaque étape ?

## Erreurs fréquentes
- Croire que le RNN traite toute la séquence d’un coup.
- Confondre l’entrée actuelle `x_t` avec la mémoire `h_t`.

## Liens liés
- [[hidden-state-rnn]]
- [[rnn-encoder-seq2seq]]
