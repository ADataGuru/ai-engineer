---
title: "Modèle séquence à séquence"
tags: [deep-learning, seq2seq, nlp]
niveau: base
source: "notes utilisateur"
last_review: 2026-07-06
---

# Modèle séquence à séquence

## Idée en 1 phrase
Un modèle séquence à séquence transforme une séquence d’entrée en une autre séquence de sortie.

## Prérequis
- Réseaux de neurones
- Vecteurs
- [[word-embedding]]

## À quoi ça sert ?
- Traduire une phrase d’une langue vers une autre.
- Résumer, transcrire ou générer du texte.

## Explication simple
Le modèle lit une phrase, en extrait une représentation interne, puis génère une nouvelle phrase.

## Explication technique
Un seq2seq classique contient deux parties : un **encoder** et un **decoder**.  
L’encoder lit la séquence d’entrée et produit des états cachés.  
Le decoder utilise ces états pour générer la séquence de sortie, token par token.

## Mini exemple

```python
source = ["I", "love", "AI"]
target = ["J'", "aime", "l'IA"]

# Le modèle apprend : source -> target
```

## Questions pour me tester

1. Que transforme un modèle seq2seq ?
2. Pourquoi utilise-t-on un encoder et un decoder ?
3. Donne un cas d’usage concret du seq2seq.

## Erreurs fréquentes

- Croire que le modèle produit toute la sortie d’un coup.
- Confondre encoder et decoder.

## Liens liés

- [[rnn-et-hidden-state]]
- [[context-vector-seq2seq]]
- [[attention-dans-seq2seq]]
