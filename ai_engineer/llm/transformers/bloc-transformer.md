---
title: "Bloc Transformer"
tags: [transformer, llm, architecture]
niveau: intermédiaire
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Bloc Transformer

## Idée en 1 phrase
Un bloc Transformer combine attention, réseau feed-forward et normalisation pour construire des représentations de tokens de plus en plus utiles.

## Prérequis
- Tokenisation
- Embedding
- Self-attention
- Réseau de neurones

## À quoi ça sert ?
- Comprendre le contexte d’une séquence.
- Produire des représentations utiles pour prédire le token suivant.

## Explication simple
Le Transformer transforme d’abord les tokens en vecteurs.  
Puis il ajoute l’information de position.  
Ensuite, plusieurs blocs attention + feed-forward enrichissent ces vecteurs.

## Explication technique
Pipeline simplifié :

```text
texte → tokens → embeddings → position → attention → feed-forward → logits
```

L’attention mélange l’information entre tokens.  
Le feed-forward transforme chaque token indépendamment.  
Les couches répétées permettent d’apprendre des représentations complexes.

Important pour l’inférence LLM : les embeddings, l’attention et le KV cache influencent fortement la mémoire GPU.

## Mini exemple

```python
# Vision simplifiée
x = token_embedding(tokens) + positional_encoding
x = attention(x)
x = feed_forward(x)
```

## Questions pour me tester

1. Pourquoi faut-il une information de position ?
2. Quelle différence entre attention et feed-forward ?
3. Pourquoi les Transformers sont importants pour les LLM ?

## Erreurs fréquentes

- Croire que l’attention remplace entièrement les réseaux de neurones.
- Oublier que sans position, l’ordre des tokens est mal représenté.

## Liens liés

- [[self-attention]]
- [[multi-head-attention]]
