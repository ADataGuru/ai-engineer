# Parcours nanoGPT — TP Jupyter

Ce dossier contient 4 TP au format Jupyter Notebook, chacun en deux versions :

- `*_enonce.ipynb` : énoncé sans solutions, avec questions, prérequis, sources exactes et indices.
- `*_solutions.ipynb` : corrigé détaillé, avec réponses pédagogiques, code exécutable et explications.

## Ordre recommandé

1. `tp1_enonce.ipynb` puis `tp1_solutions.ipynb`
2. `tp2_enonce.ipynb` puis `tp2_solutions.ipynb`
3. `tp3_enonce.ipynb` puis `tp3_solutions.ipynb`
4. `tp4_enonce.ipynb` puis `tp4_solutions.ipynb`

## Focus pédagogique

Le TP 2 est volontairement plus détaillé que les autres. Il sert à comprendre `model.py` de nanoGPT en profondeur :
- shapes `(B,T,C)` ;
- embeddings ;
- attention causale ;
- Q/K/V ;
- masque causal ;
- MLP ;
- logits ;
- cross-entropy loss.

## Préparation

Dans Jupyter, ouvre les notebooks dans l'ordre. Le TP 1 clone nanoGPT et prépare les données Shakespeare. Le TP 3 lance un entraînement court CPU par défaut. Si tu as un GPU CUDA, tu pourras adapter `--device=cuda`.

## Sources principales

- nanoGPT : https://github.com/karpathy/nanoGPT
- `model.py` : https://github.com/karpathy/nanoGPT/blob/master/model.py
- `train.py` : https://github.com/karpathy/nanoGPT/blob/master/train.py
- `sample.py` : https://github.com/karpathy/nanoGPT/blob/master/sample.py
- PyTorch docs : https://docs.pytorch.org/docs/stable/index.html
