---
title: "Attention dans un modèle seq2seq"
tags: [attention, seq2seq, nlp]
niveau: intermédiaire
source: "notes utilisateur"
last_review: 2026-07-06
---

# Attention dans un modèle seq2seq

## Idée en 1 phrase
L’attention permet au decoder de regarder tous les hidden states de l’encoder au lieu d’utiliser seulement le dernier.

## Prérequis
- [[rnn-et-hidden-state]]
- [[context-vector-seq2seq]]
- Softmax

## À quoi ça sert ?
- Réduire le goulot d’étranglement du context vector unique.
- Mieux gérer les longues séquences.
- Donner au decoder une information différente à chaque étape.

## Explication simple
Au lieu de résumer toute la phrase en un seul vecteur, le decoder choisit à chaque étape quelles parties de l’entrée sont les plus utiles.

## Explication technique
À une étape de génération, le decoder produit un hidden state.  
Ce hidden state est comparé aux hidden states de l’encoder avec une fonction de score.  
Les scores passent dans une softmax pour obtenir des poids d’attention.  
On fait ensuite une somme pondérée des hidden states de l’encoder pour créer un nouveau context vector.

Ce context vector est souvent concaténé avec le hidden state du decoder, puis envoyé à un réseau feed-forward pour prédire le prochain token.

Important pour les LLM : l’attention est centrale pour l’inférence, la mémoire GPU et l’optimisation, notamment via le KV cache dans les Transformers.

## Mini exemple

```python
import torch

scores = torch.tensor([1.0, 3.0, 0.5])
weights = torch.softmax(scores, dim=0)

encoder_states = torch.randn(3, 512)
context = weights @ encoder_states

print(context.shape)  # torch.Size([512])
```

## Questions pour me tester

1. Pourquoi l’attention utilise-t-elle tous les hidden states de l’encoder ?
2. À quoi sert la softmax dans l’attention ?
3. Comment construit-on le nouveau context vector ?

## Erreurs fréquentes

- Croire que l’attention remplace forcément le decoder.
- Confondre score d’attention et probabilité du prochain token.
- Oublier que les Transformers généralisent l’attention sans RNN.

## Liens liés

- [[context-vector-seq2seq]]
- [[modele-sequence-a-sequence]]
- [[kv-cache]]
