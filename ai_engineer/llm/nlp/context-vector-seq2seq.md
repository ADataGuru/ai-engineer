---
title: "Context vector dans un seq2seq classique"
tags: [seq2seq, rnn, context-vector]
niveau: intermédiaire
source: "notes utilisateur"
last_review: 2026-07-06
---

# Context vector dans un seq2seq classique

## Idée en 1 phrase
Dans un seq2seq RNN classique, le context vector est souvent le dernier hidden state de l’encoder.

## Prérequis
- [[rnn-et-hidden-state]]
- [[word-embedding]]

## À quoi ça sert ?
- Résumer toute la séquence d’entrée.
- Donner au decoder l’information nécessaire pour générer la sortie.

## Explication simple
L’encoder lit toute la phrase.  
Son dernier hidden state est censé contenir un résumé de ce qu’il a lu.  
Le decoder utilise ce résumé pour produire la sortie.

## Explication technique
À chaque token, l’encoder produit un hidden state.  
Dans le modèle seq2seq classique, seul le dernier hidden state est transmis au decoder.  
Sa dimension est fixée par la configuration du modèle : par exemple 256, 512 ou 1024.

Cette compression en un seul vecteur peut devenir un goulot d’étranglement pour les longues séquences.

## Mini exemple

```python
encoder_hidden_states = ["h1", "h2", "h3", "h4"]

context_vector = encoder_hidden_states[-1]  # h4
```

## Questions pour me tester

1. Qu’est-ce que le context vector ?
2. Pourquoi le dernier hidden state peut devenir un goulot d’étranglement ?
3. Que signifie une dimension de hidden state de 512 ?

## Erreurs fréquentes

- Croire que tous les hidden states sont utilisés dans le seq2seq classique.
- Oublier que la taille du context vector est fixée par la configuration du modèle.
- Penser que le context vector contient parfaitement toute la phrase.

## Liens liés

- [[modele-sequence-a-sequence]]
- [[attention-dans-seq2seq]]
