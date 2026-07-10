---
title: "Self-attention"
tags: [self-attention, transformer, llm]
niveau: intermédiaire
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Self-attention

## Idée en 1 phrase
La self-attention permet à chaque token d’une séquence de se mettre à jour avec l’information des autres tokens.

## Prérequis
- Q, K, V
- Softmax
- Embedding

## À quoi ça sert ?
- Donner du contexte à chaque token.
- Capturer les relations entre mots éloignés.

## Explication simple
Chaque mot regarde les autres mots de la phrase et décide lesquels sont importants pour comprendre son sens.

## Explication technique
La formule classique est :

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k))V
```

Les scores `QKᵀ` donnent les relations.  
La softmax donne les poids.  
Le résultat est une somme pondérée des valeurs `V`.

Important pour le serving LLM : l’attention coûte cher en mémoire car elle compare les tokens entre eux.

## Mini exemple

```python
import torch

weights = torch.softmax(Q @ K.T / (Q.shape[-1] ** 0.5), dim=-1)
out = weights @ V
```

## Questions pour me tester

1. Pourquoi parle-t-on de self-attention ?
2. Sur quelle matrice applique-t-on la softmax ?
3. Pourquoi l’attention peut coûter cher quand la séquence est longue ?

## Erreurs fréquentes

- Penser qu’un token ne regarde que ses voisins directs.
- Oublier que l’attention produit de nouveaux embeddings contextualisés.

## Liens liés

- [[roles-de-q-k-v]]
- [[multi-head-attention]]
