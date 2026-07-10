---
title: "Softmax dans l’attention"
tags: [softmax, attention, probabilites]
niveau: base
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Softmax dans l’attention

## Idée en 1 phrase
La softmax transforme les scores d’attention en poids positifs qui somment à 1.

## Prérequis
- Exponentielle
- Normalisation
- Score d’attention

## À quoi ça sert ?
- Convertir des scores bruts en coefficients utilisables.
- Forcer le modèle à répartir son attention entre les tokens.

## Explication simple
La softmax transforme une liste de scores en pourcentages.  
Les gros scores deviennent dominants, mais tous les poids restent positifs.

## Explication technique
Pour chaque token, on applique :

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Cela évite les poids négatifs et donne une somme totale égale à 1.

## Mini exemple

```python
import torch

scores = torch.tensor([1.0, 0.75, 0.0, 0.0])
weights = torch.softmax(scores, dim=0)
print(weights)
```

## Questions pour me tester

1. Pourquoi utilise-t-on `exp(x)` dans la softmax ?
2. Pourquoi les poids doivent-ils sommer à 1 ?
3. Que se passe-t-il si un score est beaucoup plus grand que les autres ?

## Erreurs fréquentes

- Confondre softmax et simple division par la somme.
- Oublier que la softmax amplifie les écarts entre scores.

## Liens liés

- [[scaled-dot-product-attention]]
- [[self-attention]]
