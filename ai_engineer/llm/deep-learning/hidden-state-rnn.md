---
title: "Hidden state dans un RNN"
tags: [rnn, hidden-state, sequence]
niveau: base
source: "Notes utilisateur + PyTorch nn.RNN"
last_review: 2026-07-06
---

# Hidden state dans un RNN

## Idée en 1 phrase
Le hidden state est la mémoire numérique que le RNN transporte d’une étape à la suivante.

## Prérequis
- Cellule RNN
- Vecteurs
- Séquences

## À quoi ça sert ?
- Résumer les éléments déjà lus.
- Influencer le traitement de l’élément suivant.

## Explication simple
Quand le RNN lit une phrase, il met à jour sa mémoire après chaque mot.
Cette mémoire n’est pas du texte lisible : c’est un vecteur de nombres.

## Explication technique
À chaque étape, le hidden state est recalculé avec l’entrée actuelle et l’ancien hidden state.

```txt
ancien hidden + entrée actuelle -> nouveau hidden
```

Le même mécanisme se répète sur toute la séquence.

## Mini exemple

```python
old_h = h
new_h = rnn_cell(x_t, old_h)
h = new_h
```

## Questions pour me tester
1. Le hidden state est-il un mot, une phrase ou un vecteur ?
2. Pourquoi le hidden state change-t-il à chaque étape ?
3. Quelle différence entre `old_h` et `new_h` ?

## Erreurs fréquentes
- Croire que le hidden state contient des mots explicites.
- Oublier que le hidden state dépend de tout ce qui a été lu avant.

## Liens liés
- [[cellule-rnn]]
- [[outputs-vs-hidden-pytorch-rnn]]
