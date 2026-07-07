---
title: "Word embedding"
tags: [nlp, embedding, vecteurs]
niveau: base
source: "notes utilisateur"
last_review: 2026-07-06
---

# Word embedding

## Idée en 1 phrase
Un word embedding transforme un mot ou token en vecteur numérique exploitable par un modèle.

## Prérequis
- Vecteurs
- Tokens
- Réseaux de neurones

## À quoi ça sert ?
- Donner une représentation numérique du texte.
- Permettre au modèle de calculer sur les mots.
- Rapprocher certains tokens selon leur usage statistique.

## Explication simple
Un modèle ne comprend pas directement les mots.  
Il manipule des nombres.  
L’embedding remplace donc chaque token par un vecteur.

## Explication technique
Une table d’embeddings associe chaque token ID à un vecteur dense.  
Ces vecteurs sont appris pendant l’entraînement.  
La dimension du vecteur est un hyperparamètre, par exemple 256, 512 ou 1024.

## Mini exemple

```python
import torch

embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=512)

token_ids = torch.tensor([12, 45, 87])
vectors = embedding(token_ids)

print(vectors.shape)  # torch.Size([3, 512])
```

## Questions pour me tester

1. Pourquoi faut-il transformer les mots en vecteurs ?
2. Que contient une table d’embeddings ?
3. Que signifie `embedding_dim=512` ?

## Erreurs fréquentes

- Croire qu’un embedding est une définition du mot.
- Confondre token ID et vecteur embedding.
- Oublier que les embeddings sont appris.

## Liens liés

- [[rnn-et-hidden-state]]
- [[modele-sequence-a-sequence]]
