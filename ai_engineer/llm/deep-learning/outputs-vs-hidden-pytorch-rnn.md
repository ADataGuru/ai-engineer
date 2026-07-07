---
title: "outputs vs hidden dans PyTorch RNN"
tags: [pytorch, rnn, tensor]
niveau: intermédiaire
source: "Notes utilisateur + PyTorch nn.RNN"
last_review: 2026-07-06
---

# outputs vs hidden dans PyTorch RNN

## Idée en 1 phrase
Dans PyTorch, `outputs` contient les états de chaque étape, tandis que `hidden` contient le dernier état final.

## Prérequis
- PyTorch
- RNN
- Tensor shape

## À quoi ça sert ?
- Récupérer tous les hidden states.
- Utiliser le dernier hidden state comme résumé.

## Explication simple
Pour une phrase de 4 mots :

```txt
outputs = [h1, h2, h3, h4]
hidden  = h4
```

## Explication technique
Avec `batch_first=True` :

```txt
outputs.shape = (batch_size, seq_len, hidden_size)
hidden.shape  = (num_layers, batch_size, hidden_size)
```

`outputs` sert souvent à l’attention. `hidden` sert souvent de résumé final.

## Mini exemple

```python
import torch.nn as nn

embedding = nn.Embedding(vocab_size, emb_size)
rnn = nn.RNN(emb_size, hidden_size, batch_first=True)

x = embedding(input_ids)
outputs, hidden = rnn(x)
```

## Questions pour me tester
1. Que contient `outputs[:, 0, :]` ?
2. Que contient `hidden` ?
3. Pourquoi l’attention utilise-t-elle souvent `outputs` ?

## Erreurs fréquentes
- Confondre `outputs` et `hidden`.
- Oublier que `outputs` garde les états de chaque mot.

## Liens liés
- [[hidden-state-rnn]]
- [[attention-seq2seq]]
