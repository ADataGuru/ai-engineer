---
title: "RNN et hidden state"
tags: [deep-learning, rnn, hidden-state]
niveau: base
source: "notes utilisateur"
last_review: 2026-07-06
---

# RNN et hidden state

## Idée en 1 phrase
Un RNN traite une séquence étape par étape en gardant une mémoire appelée hidden state.

## Prérequis
- Vecteurs
- Réseaux de neurones
- Séquences

## À quoi ça sert ?
- Traiter du texte, du son ou des séries temporelles.
- Garder une information sur les éléments précédents.

## Explication simple
Un RNN lit les tokens un par un.  
À chaque étape, il combine le token actuel avec sa mémoire du passé.

## Explication technique
À l’étape `t`, le RNN prend deux entrées :  
- le vecteur du token actuel `x_t`
- le hidden state précédent `h_{t-1}`

Il produit :
- un output `y_t`
- un nouveau hidden state `h_t`

Le même bloc RNN est réutilisé à chaque étape de la séquence.

## Mini exemple

```python
h_t = RNNCell(x_t, h_prev)
```

## Questions pour me tester

1. Quelles sont les deux entrées d’un RNN ?
2. À quoi sert le hidden state ?
3. Pourquoi un RNN est adapté aux séquences ?

## Erreurs fréquentes

- Croire que le hidden state est le texte original.
- Oublier que le même RNN est réutilisé à chaque étape.
- Confondre hidden state et output.

## Liens liés

- [[modele-sequence-a-sequence]]
- [[context-vector-seq2seq]]
