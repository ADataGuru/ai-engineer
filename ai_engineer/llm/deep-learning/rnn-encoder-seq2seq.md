---
title: "RNN encoder dans un modèle seq2seq"
tags: [seq2seq, encoder, rnn]
niveau: intermédiaire
source: "Notes utilisateur + PyTorch seq2seq tutorial"
last_review: 2026-07-06
---

# RNN encoder dans un modèle seq2seq

## Idée en 1 phrase
Un RNN encoder lit une phrase mot par mot et produit une mémoire finale utilisée par le decoder.

## Prérequis
- RNN
- Hidden state
- Embedding

## À quoi ça sert ?
- Transformer une phrase en représentation vectorielle.
- Fournir un contexte initial au decoder.

## Explication simple
Pour la phrase `je mange une pomme`, l’encoder fait :

```txt
je     -> h1
mange  -> h2
une    -> h3
pomme  -> h4
```

`h4` est le résumé final de la phrase.

## Explication technique
Chaque mot est transformé en embedding, puis donné à la cellule RNN avec le hidden state précédent.
Dans un seq2seq simple, le dernier hidden state devient le `context vector`.

## Mini exemple

```python
h = 0

for word in ["je", "mange", "une", "pomme"]:
    x = embedding(word)
    h = rnn_cell(x, h)

context = h
```

```python
sentence = ["je", "mange", "une", "pomme"]

vocab = {
    "je": 0,
    "mange": 1,
    "une": 2,
    "pomme": 3
}

embedding_size = 5
hidden_size = 4
vocab_size = len(vocab)

# Table d'embeddings : chaque mot id → vecteur
embedding_table = np.random.randn(vocab_size, embedding_size) * 0.1

# Poids du RNN encoder
W_xh = np.random.randn(hidden_size, embedding_size) * 0.1
W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
b_h = np.zeros(hidden_size)

h = np.zeros(hidden_size)
encoder_hidden_states = []

for word in sentence:
    word_id = vocab[word]

    # 1. mot → embedding
    x_t = embedding_table[word_id]

    # 2. embedding + ancien hidden state → nouveau hidden state
    h = np.tanh(W_xh @ x_t + W_hh @ h + b_h)

    # 3. on garde le hidden state de ce mot
    encoder_hidden_states.append(h)

    print("mot:", word)
    print("hidden state:", h)
    print()

context_vector = h
```

## Questions pour me tester
1. Pourquoi l’encoder lit-il les mots un par un ?
2. Que représente `context` ?
3. Pourquoi faut-il transformer un mot en embedding ?

## Erreurs fréquentes
- Croire que l’encoder génère directement la traduction.
- Croire que le context vector est une phrase lisible.

## Liens liés
- [[cellule-rnn]]
- [[decoder-seq2seq-simple]]

