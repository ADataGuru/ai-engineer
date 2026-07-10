---
title: "Embedding contextuel et similarité"
tags: [embedding, attention, nlp]
niveau: base
source: "Notes utilisateur - attention et Transformer"
last_review: 2026-07-09
---

# Embedding contextuel et similarité

## Idée en 1 phrase
Un embedding contextuel représente un token en tenant compte des autres tokens autour de lui.

## Prérequis
- Token
- Embedding
- Vecteur

## À quoi ça sert ?
- Comprendre le sens réel d’un mot dans une phrase.
- Rapprocher les mots liés dans un contexte donné.

## Explication simple
Un même mot peut avoir plusieurs sens.  
Le contexte aide le modèle à choisir le bon sens.

Exemple :  
Dans “achète une pomme et une orange”, “pomme” est proche de “orange”, donc plutôt un fruit.

## Explication technique
Au départ, chaque token reçoit un embedding.  
L’attention modifie ensuite cet embedding en mélangeant l’information des autres tokens.  
Le nouveau vecteur devient plus adapté au contexte de la phrase.

## Mini exemple

```python
tokens = ["achète", "une", "pomme", "et", "une", "orange"]
# Après attention, le vecteur de "pomme" peut se rapprocher de "orange"
```

## Questions pour me tester

1. Pourquoi un embedding fixe peut être insuffisant ?
2. Comment le contexte aide-t-il à désambiguïser un mot ?
3. Pourquoi “pomme” peut se rapprocher de “orange” dans une phrase ?

## Erreurs fréquentes

- Croire qu’un mot garde toujours le même vecteur final.
- Confondre similarité sémantique globale et similarité dans un contexte précis.

## Liens liés

- [[scaled-dot-product-attention]]
- [[self-attention]]
