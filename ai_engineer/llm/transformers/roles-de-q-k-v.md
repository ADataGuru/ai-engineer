---
title: "Rôles de Q, K et V"
tags: [attention, qkv, transformer]
niveau: intermédiaire
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Rôles de Q, K et V

## Idée en 1 phrase
Q, K et V sont trois projections apprises qui servent à chercher, comparer et récupérer l’information utile.

## Prérequis
- Matrice de poids
- Transformation linéaire
- Embedding

## À quoi ça sert ?
- Apprendre quels tokens sont liés.
- Produire une représentation contextualisée de chaque token.

## Explication simple
On peut voir Q comme une question, K comme une étiquette, et V comme le contenu à récupérer.  
Q compare avec K, puis les poids obtenus mélangent les V.

## Explication technique
À partir d’un embedding `X`, le modèle apprend :

```text
Q = XWq
K = XWk
V = XWv
```

`Q` et `K` servent à calculer les scores d’attention.  
`V` contient l’information qui sera effectivement mélangée.

## Mini exemple

```python
import torch

X = torch.randn(5, 16)
Wq = torch.randn(16, 8)
Wk = torch.randn(16, 8)
Wv = torch.randn(16, 8)

Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

## Questions pour me tester

1. Quel est le rôle de Q ?
2. Quel est le rôle de K ?
3. Pourquoi applique-t-on les poids d’attention sur V ?

## Erreurs fréquentes

- Dire que V sert directement à prédire le prochain mot.
- Croire que Q, K et V sont fixes : ce sont des poids appris.

## Liens liés

- [[scaled-dot-product-attention]]
- [[self-attention]]
