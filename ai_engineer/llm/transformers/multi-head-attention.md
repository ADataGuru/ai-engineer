---
title: "Multi-head attention"
tags: [multi-head-attention, transformer, llm]
niveau: intermédiaire
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Multi-head attention

## Idée en 1 phrase
La multi-head attention utilise plusieurs attentions en parallèle pour capturer différents types de relations.

## Prérequis
- Self-attention
- Projection linéaire
- Concaténation

## À quoi ça sert ?
- Repérer plusieurs relations en même temps.
- Enrichir la représentation finale des tokens.

## Explication simple
Une seule tête peut regarder un type de lien.  
Plusieurs têtes peuvent regarder plusieurs choses : grammaire, sujet, objet, contexte sémantique, etc.

## Explication technique
Chaque head possède ses propres matrices `Wq`, `Wk`, `Wv`.  
Chaque head produit une sortie.  
On concatène ensuite les sorties, puis on applique une projection linéaire finale `Wo`.

Cette projection ne choisit pas simplement “la meilleure tête”.  
Elle apprend à combiner les informations utiles de toutes les têtes.

## Mini exemple

```python
# Vision simplifiée : une head = une attention avec ses propres Wq, Wk, Wv
heads_outputs = [head(x) for head in heads]
out = linear_projection(concat(heads_outputs))
```

## Questions pour me tester

1. Pourquoi utiliser plusieurs heads ?
2. Que fait la projection linéaire finale ?
3. Pourquoi une head peut-elle apprendre une relation différente d’une autre ?

## Erreurs fréquentes

- Croire qu’une head est forcément meilleure que les autres.
- Penser que la projection finale fait seulement une réduction de dimension.

## Liens liés

- [[self-attention]]
- [[bloc-transformer]]
