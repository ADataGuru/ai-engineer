---
title: "Scaled Dot-Product Attention"
tags: [attention, transformer, similarite]
niveau: intermédiaire
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Scaled Dot-Product Attention

## Idée en 1 phrase
Le scaled dot-product attention mesure quels tokens doivent influencer un token donné.

## Prérequis
- Produit scalaire
- Matrice
- Softmax

## À quoi ça sert ?
- Calculer les relations entre tous les tokens.
- Donner plus de poids aux tokens importants.

## Explication simple
Pour savoir si deux mots sont liés, on compare leurs vecteurs.  
Plus le score est grand, plus le lien est fort.

## Explication technique
L’attention calcule les scores avec :

```text
scores = QKᵀ / sqrt(d_k)
```

On divise par `sqrt(d_k)` pour éviter des scores trop grands, qui rendraient la softmax trop extrême.

## Mini exemple

```python
import torch

Q = torch.randn(4, 8)  # 4 tokens, dimension 8
K = torch.randn(4, 8)

scores = Q @ K.T / (8 ** 0.5)
```

## Questions pour me tester

1. Que mesure le produit scalaire entre deux vecteurs ?
2. Pourquoi divise-t-on par `sqrt(d_k)` ?
3. Quelle est la forme de la matrice de scores pour 4 tokens ?

## Erreurs fréquentes

- Dire qu’on divise pour réduire la taille des vecteurs.
- Oublier que `QKᵀ` compare tous les tokens entre eux.

## Liens liés

- [[softmax-dans-attention]]
- [[roles-de-q-k-v]]
