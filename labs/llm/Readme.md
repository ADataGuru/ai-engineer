# nn.Embedding - Mémo

## C'est quoi ?
Une table de correspondance : INDEX → VECTEUR

## La table (matrice W)
- Forme : (num_embeddings, embedding_dim)
- Chaque ligne = le vecteur d'un mot

           dim_0    dim_1    dim_2
index 0  [ -0.38    -0.83    -1.68 ]
index 1  [  1.30    -0.68     0.88 ]
index 2  [ -0.53    -0.62    -0.90 ]
index 3  [  1.09     0.90    -0.34 ]

## Le calcul : juste une sélection de ligne !
input = [1]  →  on prend la ligne 1  →  [ 1.30  -0.68   0.88 ]
Pas de multiplication, pas de calcul → juste un coup d'index !

## Les dimensions de l'output
input shape  : (batch, nb_mots)
output shape : (batch, nb_mots, embedding_dim)
                  │        │          │
                  │        │          └──► taille du vecteur
                  │        └─────────────► nb de mots par séquence
                  └──────────────────────► nb de séquences

## Exemple
input  = [[1, 3],   # séquence_0 : mot index 1 puis mot index 3
           [0, 2]]  # séquence_1 : mot index 0 puis mot index 2

output = [
    [[1.30, -0.68,  0.88],   # séquence_0 : vecteur index 1
     [1.09,  0.90, -0.34]],  # séquence_0 : vecteur index 3

    [[-0.38, -0.83, -1.68],  # séquence_1 : vecteur index 0
     [-0.53, -0.62, -0.90]]  # séquence_1 : vecteur index 2
]

## Points clés
- Les poids sont ALÉATOIRES au départ
- Les vecteurs sont APPRIS pendant l'entraînement
- Des mots similaires → des vecteurs proches dans l'espace




# view - Mémo

## C'est quoi ?
Réorganise un tenseur dans une nouvelle forme
SANS changer les données elles-mêmes !

## La règle d'or
Total éléments AVANT = Total éléments APRÈS
2 × 3 × 4 = 24  →  6 × 4 = 24  ✅
2 × 3 × 4 = 24  →  5 × 4 = 20  ❌

## Visualisation
AVANT : shape (2, 3, 4)          APRÈS : shape (6, 4)
         B  T  C                          B*T  C

┌──────────────────┐             ┌──────────────────┐
│ seq_0            │             │ [ 1,  2,  3,  4] │
│ [ 1,  2,  3,  4] │             │ [ 5,  6,  7,  8] │
│ [ 5,  6,  7,  8] │  ────────►  │ [ 9, 10, 11, 12] │
│ [ 9, 10, 11, 12] │             │ [13, 14, 15, 16] │
├──────────────────┤             │ [17, 18, 19, 20] │
│ seq_1            │             │ [21, 22, 23, 24] │
│ [13, 14, 15, 16] │             └──────────────────┘
│ [17, 18, 19, 20] │
│ [21, 22, 23, 24] │
└──────────────────┘

## Usage typique en Deep Learning
# Certaines couches (ex: nn.Linear) attendent du 2D
# On aplatit le batch et le temps ensemble

logits.shape        # (B, T, C) → 3D
logits.view(B*T, C) # (B*T, C) → 2D  ✅ prêt pour nn.Linear

## Dimensions
B = batch    → nb de séquences
T = time     → nb de mots par séquence
C = channels → taille du vecteur

## Points clés
- Les données ne changent PAS
- Juste une nouvelle façon de les "voir"
- view(-1, C) : le -1 calcule automatiquement la dimension manquante





# F.cross_entropy - Mémo

## C'est quoi ?
Mesure à quel point le modèle se trompe
entre ses prédictions (logits) et les vraies réponses (targets)
Plus la loss est proche de 0, mieux le modèle prédit !

## Les ingrédients
F.cross_entropy(logits, targets)

logits  → scores bruts du modèle  shape : (N, C)
targets → vraies réponses         shape : (N,)

N = nb d'exemples
C = nb de classes (taille du vocabulaire)

## Les 3 étapes internes

ÉTAPE 1 : Softmax → scores en probabilités
logits = [ 1.2   0.5   -0.3   2.1 ]
                  │
                  ▼ softmax
probs  = [ 0.21  0.10   0.05  0.64 ]  (somme = 1)

ÉTAPE 2 : Log → probabilités en log-probabilités
probs     = [ 0.21   0.10   0.05   0.64 ]
                  │
                  ▼ log
log_probs = [-1.56  -2.30  -2.99  -0.44 ]

ÉTAPE 3 : On garde uniquement la classe cible
target = 3  →  log_probs[3] = -0.44
loss   = -(-0.44) = 0.44  ✅

## Visualisation loss
Modèle CORRECT                    Modèle FAUX
──────────────────────────        ──────────────────────────
probs  = [0.05, 0.05, 0.85, 0.05] probs  = [0.85, 0.05, 0.05, 0.05]
target = 2                        target = 2

loss = -log(0.85) = 0.16 ✅       loss = -log(0.05) = 2.99 ❌
       petite loss                        grande loss

## Points clés
- Softmax est inclus → pas besoin de le faire avant !
- Loss faible  → le modèle prédit bien
- Loss élevée  → le modèle se trompe
- Objectif     → minimiser cette valeur pendant l'entraînement





# Ligne J : `logits[:, -1, :]` - Le dernier token

## 🎯 C'est quoi cette syntaxe ?

```python
logits[:, -1, :]
```

```
logits[ : ,  -1 ,  : ]
        │     │    │
        │     │    └──► tous les channels (C)
        │     └───────► dernier élément de T (temps)
        └─────────────► tous les batches (B)
```

---

## 📊 Visualisation du tenseur logits

```
logits shape : (B, T, C)

Imaginons :
B = 2  → 2 séquences
T = 4  → 4 tokens par séquence
C = 5  → vocabulaire de 5 mots

┌─────────────────────────────────────────────────────┐
│                  logits (2, 4, 5)                   │
│                                                     │
│  batch_0                                            │
│  ┌─────────────────────────────────────────────┐    │
│  │ token_0  [ 0.1,  0.5, -0.3,  0.8,  0.2 ]   │    │
│  │ token_1  [ 0.9, -0.1,  0.4,  0.2,  0.7 ]   │    │
│  │ token_2  [ 0.3,  0.8,  0.1, -0.5,  0.4 ]   │    │
│  │ token_3  [ 0.6,  0.2,  0.9,  0.1, -0.3 ] ◄─┼──── on prend ça !
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  batch_1                                            │
│  ┌─────────────────────────────────────────────┐    │
│  │ token_0  [ 0.4,  0.1,  0.7, -0.2,  0.5 ]   │    │
│  │ token_1  [ 0.2,  0.6, -0.1,  0.9,  0.3 ]   │    │
│  │ token_2  [ 0.8, -0.3,  0.5,  0.4,  0.1 ]   │    │
│  │ token_3  [ 0.1,  0.4,  0.3,  0.7,  0.6 ] ◄─┼──── et ça !
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## ✂️ Après le slicing `logits[:, -1, :]`

```
AVANT                          APRÈS
shape (2, 4, 5)                shape (2, 5)

batch_0                        batch_0
[ 0.1,  0.5, -0.3,  0.8,  0.2 ]
[ 0.9, -0.1,  0.4,  0.2,  0.7 ]    [ 0.6,  0.2,  0.9,  0.1, -0.3 ]
[ 0.3,  0.8,  0.1, -0.5,  0.4 ]
[ 0.6,  0.2,  0.9,  0.1, -0.3 ] ──►
                                    [ 0.1,  0.4,  0.3,  0.7,  0.6 ]
batch_1                        batch_1
[ 0.4,  0.1,  0.7, -0.2,  0.5 ]
[ 0.2,  0.6, -0.1,  0.9,  0.3 ]
[ 0.8, -0.3,  0.5,  0.4,  0.1 ]
[ 0.1,  0.4,  0.3,  0.7,  0.6 ]
```

---

## 🧠 Mais POURQUOI le dernier token ?

### Le contexte : un modèle Bigram

```
Input séquence : "je" "aime" "les" "chats"
                  │      │     │      │
                  t0     t1    t2     t3
```

```
┌─────────────────────────────────────────────────────┐
│         CE QUE PRÉDIT CHAQUE TOKEN                  │
│                                                     │
│  token_0 "je"    → prédit le mot après "je"         │
│  token_1 "aime"  → prédit le mot après "aime"       │
│  token_2 "les"   → prédit le mot après "les"        │
│  token_3 "chats" → prédit le mot après "chats" ✅   │
│                              │                      │
│                              └──► C'EST ÇA          │
│                                   qu'on veut !      │
│                                   le PROCHAIN mot   │
│                                   de la séquence    │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Le processus de génération complet

```
ÉTAPE 1 : Séquence initiale
─────────────────────────────────────────────
idx = [ "je", "aime", "les", "chats" ]
         t0     t1     t2      t3
                                ▲
                          dernier token connu

ÉTAPE 2 : Forward pass → logits (B, T, C)
─────────────────────────────────────────────
logits =
  t0 → [ scores pour le mot après "je"    ]
  t1 → [ scores pour le mot après "aime"  ]
  t2 → [ scores pour le mot après "les"   ]
  t3 → [ scores pour le mot après "chats" ]  ← on veut ça !

ÉTAPE 3 : logits[:, -1, :] → (B, C)
─────────────────────────────────────────────
on garde SEULEMENT :
  → [ scores pour le mot après "chats" ]

ÉTAPE 4 : softmax → probs
─────────────────────────────────────────────
probs = [ 0.05, 0.60, 0.10, 0.20, 0.05 ]
             │     │     │     │     │
            "je" "mange" "les" "et" "aime"
                   ▲
              plus probable !

ÉTAPE 5 : multinomial → idx_next
─────────────────────────────────────────────
idx_next = "mange"

ÉTAPE 6 : cat → nouvelle séquence
─────────────────────────────────────────────
idx = [ "je", "aime", "les", "chats", "mange" ]
                                        ▲



----------------------------------------------------------------------------



# Accès Coalescé vs Non-Coalescé

## 🎯 L'idée de base

> Un GPU exécute **32 threads simultanément** (= 1 Warp)
> La question est : **comment ces 32 threads accèdent-ils à la mémoire ?**

---

## ✅ Accès COALESCÉ

```
32 threads accèdent à des adresses CONSÉCUTIVES en mémoire

Thread_0  → adresse 0
Thread_1  → adresse 1
Thread_2  → adresse 2
...
Thread_31 → adresse 31

MÉMOIRE VRAM
┌────┬────┬────┬────┬────┬────┬─────┬─────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ ... │ 31  │
└────┴────┴────┴────┴────┴────┴─────┴─────┘
  ▲    ▲    ▲    ▲    ▲    ▲           ▲
  │    │    │    │    │    │           │
 t0   t1   t2   t3   t4   t5  ...    t31

→ GPU regroupe tout en 1 SEULE transaction mémoire ✅
→ très rapide !
```

---

## ❌ Accès NON COALESCÉ

```
32 threads accèdent à des adresses ÉPARPILLÉES en mémoire

Thread_0  → adresse 847
Thread_1  → adresse 12
Thread_2  → adresse 3041
Thread_3  → adresse 5
...
Thread_31 → adresse 2234

MÉMOIRE VRAM
┌────┬────┬────┬─────┬──────┬─────┬──────┬──────┐
│ 5  │ 12 │ ...│ 847 │ 2234 │ ... │ 3041 │ ...  │
└────┴────┴────┴─────┴──────┴─────┴──────┴──────┘
  ▲    ▲              ▲       ▲      ▲
  │    │              │       │      │
 t3   t1             t0     t31     t2

→ GPU doit faire 32 transactions séparées ❌
→ très lent !
```

---

## 🍕 Analogie Simple

```
COALESCÉ
─────────────────────────────────────────
Livreur de pizza avec 1 adresse :
"32 pizzas à livrer au même endroit"
→ 1 seul voyage !  ✅


NON COALESCÉ
─────────────────────────────────────────
Livreur de pizza avec 32 adresses :
"1 pizza à livrer dans 32 endroits différents"
→ 32 voyages séparés ! ❌
```

---

## 📊 Dans le cas de nn.Embedding

```
POURQUOI c'est non coalescé ?

Table W en VRAM :
┌──────────────────────────────────────────┐
│ mot_0  [ 1.2,  0.5, -0.3,  2.1,  0.8]  │  adresse 0
│ mot_1  [ 0.1,  2.3,  0.4, -0.5,  1.2]  │  adresse 5
│ mot_2  [-0.2,  0.3,  1.8,  0.1, -0.9]  │  adresse 10
│ mot_3  [ 0.9, -0.1,  0.2,  1.5,  0.3]  │  adresse 15
│ mot_4  [ 1.1,  0.7, -0.8,  0.4,  2.1]  │  adresse 20
│  ...                                    │
│ mot_N  [ ...]                           │  adresse N*5
└──────────────────────────────────────────┘

Input = [847, 12, 3041, 5, ...]
         │     │    │    │
         │     │    │    └──► thread_3 → adresse 25
         │     │    └───────► thread_2 → adresse 15205
         │     └────────────► thread_1 → adresse 60
         └──────────────────► thread_0 → adresse 4235

→ chaque thread saute à une adresse complètement différente
→ accès éparpillés = NON COALESCÉ ❌
```

---

## 📉 Impact sur les performances

```
                    Bande passante VRAM utilisée
                    ────────────────────────────
Accès coalescé   :  ████████████████████  ~900 GB/s  ✅
Accès nn.Embedding: ████░░░░░░░░░░░░░░░░  ~100 GB/s  ❌
                         ▲
                    on utilise seulement
                    ~10% de la bande passante !
```

---

## 🎯 Résumé

| | Coalescé ✅ | Non-Coalescé ❌ |
|---|---|---|
| **Adresses** | Consécutives | Éparpillées |
| **Transactions** | 1 seule | 32 séparées |
| **Vitesse** | Maximale | Dégradée |
| **Exemple** | nn.Linear | nn.Embedding |

> 💡 **C'est pour ça que les Embeddings sont considérés**
> **comme une opération "memory-bound" sur GPU !**
> Le calcul est simple (juste un lookup)
> mais la mémoire est le vrai goulot d'étranglement ! 🚧






-------------------------------------------



# Memory-Bound vs Compute-Bound sur GPU

## 🎯 L'idée de base

> Un GPU a **2 ressources limitantes** :
> - Sa puissance de **calcul** (CUDA cores)
> - Sa vitesse d'accès à la **mémoire** (VRAM)
> 
> Une opération est limitée par **l'une ou l'autre** !

---

## 🏭 Analogie : L'usine

```
┌─────────────────────────────────────────────────────┐
│                    USINE GPU                        │
│                                                     │
│  Entrepôt (VRAM)          Ouvriers (CUDA Cores)     │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │              │         │  👷 👷 👷 👷 👷 👷   │  │
│  │  matières    │ ──────► │  👷 👷 👷 👷 👷 👷   │  │
│  │  premières   │         │  👷 👷 👷 👷 👷 👷   │  │
│  │              │         │  👷 👷 👷 👷 👷 👷   │  │
│  └──────────────┘         └──────────────────────┘  │
│     tapis roulant →              travail             │
│     (bande passante)             (FLOPS)             │
└─────────────────────────────────────────────────────┘
```

---

## ⚖️ Les 2 cas

### ❌ MEMORY-BOUND : le tapis roulant est trop lent

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Entrepôt              Ouvriers                     │
│  ┌──────────┐          ┌────────────────────────┐   │
│  │          │  🐌slow  │  👷 👷 👷 👷 👷 👷    │   │
│  │  données │ ───────► │                        │   │
│  │          │          │  😴 😴 😴 😴 😴 😴    │   │
│  └──────────┘          │  (ils attendent !)     │   │
│                        └────────────────────────┘   │
│                                                     │
│  → Les ouvriers sont IDLE                           │
│  → Ils attendent les données                        │
│  → Le tapis roulant est le GOULOT D'ÉTRANGLEMENT    │
└─────────────────────────────────────────────────────┘
```

### ❌ COMPUTE-BOUND : les ouvriers sont débordés

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Entrepôt              Ouvriers                     │
│  ┌──────────┐          ┌────────────────────────┐   │
│  │          │  🚀fast  │  👷 👷 👷 👷 👷 👷    │   │
│  │  données │ ───────► │  💪 💪 💪 💪 💪 💪    │   │
│  │ entassées│          │  (surchargés !)        │   │
│  └──────────┘          │                        │   │
│   (file d'attente)     └────────────────────────┘   │
│                                                     │
│  → Les données arrivent trop vite                   │
│  → Les ouvriers ne suivent pas                      │
│  → Les CUDA cores sont le GOULOT D'ÉTRANGLEMENT     │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Les chiffres réels (ex: RTX 3090)

```
┌─────────────────────────────────────────────────────┐
│                   RTX 3090                          │
│                                                     │
│  Puissance calcul   : 35.6 TFLOPS                   │
│  Bande passante VRAM: 936 GB/s                      │
│                                                     │
│  Arithmetic Intensity = FLOPS / Bytes               │
│  ─────────────────────────────────                  │
│  = 35 600 GFLOPS / 936 GB/s                         │
│  = ~38 FLOPS par byte                               │
│                                                     │
│  → Si ton opération fait MOINS de 38 FLOPS/byte     │
│    elle est MEMORY-BOUND ❌                         │
│  → Si ton opération fait PLUS de 38 FLOPS/byte      │
│    elle est COMPUTE-BOUND ❌                        │
└─────────────────────────────────────────────────────┘
```

---

## 🔢 Arithmetic Intensity : le ratio clé

```
                    nb de calculs (FLOPS)
Arithmetic  =    ─────────────────────────
Intensity         nb de bytes lus/écrits


MEMORY-BOUND        ÉQUILIBRE          COMPUTE-BOUND
────────────        ─────────          ─────────────
ratio faible        ratio ~38          ratio élevé
< 38 FLOPS/byte     FLOPS/byte         > 38 FLOPS/byte

Mémoire est         optimal !          Calcul est
le problème                            le problème
```

---

## 📊 Où se situent nos opérations ?

```
FLOPS/byte
    │
 élevé │                              ● Matmul (GEMM)
       │                           ●    (COMPUTE-BOUND)
       │                        ●
  ~38  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← seuil RTX 3090
       │             ● Softmax
       │          ● Layer Norm
       │       ● ReLU
  faible│    ● nn.Embedding
       │         (MEMORY-BOUND)
       └──────────────────────────────►
                                    opérations
```

---

## 🔍 Analyse de nos fonctions

### nn.Embedding → MEMORY-BOUND

```
Ce qui se passe :
─────────────────
Calculs    : 0 FLOPS  (juste un lookup !)
Données lues : embedding_dim × 4 bytes par index

ex : embedding_dim = 768
     → lit 768 × 4 = 3072 bytes
     → fait 0 calculs

Arithmetic Intensity = 0 FLOPS / 3072 bytes ≈ 0
→ MEMORY-BOUND extrême ! ❌
```

### F.cross_entropy → MEMORY-BOUND + COMPUTE

```
Ce qui se passe :
─────────────────
Calculs    : exp(), log(), somme    → quelques FLOPS
Données lues : tous les logits       → beaucoup de bytes

ex : vocab_size = 50 000
     → lit 50 000 × 4 = 200 000 bytes
     → fait ~50 000 FLOPS (exp pour chaque logit)

Arithmetic Intensity = 50 000 / 200 000 = 0.25 FLOPS/byte
→ MEMORY-BOUND ! ❌
```

### nn.Linear (Matmul) → COMPUTE-BOUND

```
Ce qui se passe :
─────────────────
Calculs    : N × M × K multiplications + additions
Données lues : N×K + K×M bytes de matrices

ex : N=512, M=512, K



------------

# nn.Embedding : CPU vs GPU - Le vrai débat !

## 🎯 La question centrale

```
nn.Embedding = juste un lookup (0 calculs)
             = MEMORY-BOUND extrême

→ Alors pourquoi ne pas rester sur CPU ?
```

---

## ⚖️ Comparaison CPU vs GPU pour nn.Embedding

```
┌─────────────────────────────────────────────────────┐
│                      CPU                           │
│                                                     │
│  Bande passante RAM    : ~50 GB/s                   │
│  Latence mémoire       : ~100ns                     │
│  Nb de cores           : 8-32 cores                 │
│  Parallélisme          : faible                     │
│                                                     │
│  ✅ Avantages                                       │
│  → Excellent pour accès ALÉATOIRES                  │
│  → Gère bien les cache miss                         │
│  → Pas de transfert PCIe nécessaire                 │
│                                                     │
│  ❌ Inconvénients                                   │
│  → Peu de parallélisme                              │
│  → Doit transférer le résultat vers GPU ensuite     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                      GPU                           │
│                                                     │
│  Bande passante VRAM   : ~900 GB/s                  │
│  Latence mémoire       : ~700ns                     │
│  Nb de cores           : 10 000+ cores              │
│  Parallélisme          : massif                     │
│                                                     │
│  ✅ Avantages                                       │
│  → Bande passante ENORME                            │
│  → Peut traiter tout un batch en parallèle          │
│  → Résultat déjà en VRAM pour la suite              │
│                                                     │
│  ❌ Inconvénients                                   │
│  → Latence mémoire plus élevée que CPU              │
│  → Accès non coalescés = bande passante gâchée      │
└─────────────────────────────────────────────────────┘
```

---

## 🔍 Le vrai problème : ce n'est pas l'Embedding seul !

```
PIPELINE COMPLET D'UN TRANSFORMER
───────────────────────────────────

  Input (indices)
       │
       ▼
  nn.Embedding          ← on parle de ça
       │
       ▼
  nn.Linear             ← COMPUTE-BOUND (a besoin du GPU)
       │
       ▼
  Attention             ← COMPUTE-BOUND (a besoin du GPU)
       │
       ▼
  nn.Linear             ← COMPUTE-BOUND (a besoin du GPU)
       │
       ▼
  F.cross_entropy       ← besoin des logits sur GPU
       │
       ▼
  Backward pass         ← COMPUTE-BOUND (a besoin du GPU)
```

```
⚠️ Si on fait nn.Embedding sur CPU :

CPU ──[Embedding]──► résultat
                        │
                        │ transfert PCIe ~16GB/s
                        ▼
GPU ──[Linear]──► ...

→ Ce transfert PCIe coûte PLUS CHER
  que de subir les accès non coalescés sur GPU !
```

---

## 📊 Comparaison des coûts réels

```
Exemple :
  batch_size    = 32
  seq_len       = 512
  embedding_dim = 768

  données à transférer = 32 × 512 × 768 × 4 bytes
                       = ~50 MB

┌────────────────────────────────────────────────────┐
│  OPTION 1 : Embedding sur CPU                      │
│                                                    │
│  Embedding CPU   : 50MB / 50GB/s  = ~1ms  ✅      │
│  Transfert PCIe  : 50MB / 16GB/s  = ~3ms  ❌      │
│  ─────────────────────────────────────────         │
│  TOTAL           :                  ~4ms           │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  OPTION 2 : Embedding sur GPU                      │
│                                                    │
│  Transfert indices: ~0.01MB / 16GB/s = ~0.001ms ✅ │
│  (juste les indices entiers, pas les vecteurs !)   │
│  Embedding GPU   : 50MB / 900GB/s    = ~0.05ms ✅  │
│  (même non coalescé, disons ×10 plus lent)         │
│  ─────────────────────────────────────────         │
│  TOTAL           :                   ~0.5ms        │
└────────────────────────────────────────────────────┘

  GPU GAGNE même avec accès non coalescés ! 🏆
```

---

## 💡 L'insight clé

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Sur CPU → On transfère les VECTEURS (lourds)       │
│  ┌──────────────────────────────────────────────┐   │
│  │ 32 × 512 × 768 × 4bytes = 50MB à transférer │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  Sur GPU → On transfère les INDICES (légers)        │
│  ┌──────────────────────────────────────────────┐   │
│  │ 32 × 512 × 4bytes = 0.06MB à transférer     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│         50MB  vs  0.06MB                            │
│         c'est 800× moins de données !               │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Conclusion

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  nn.Embedding est MEMORY-BOUND                      │
│  MAIS il vaut mieux l'exécuter sur GPU car :        │
│                                                     │
│  1. On transfère des INDICES (légers)               │
│     pas des VECTEURS (lourds)                       │
│                                                     │
│  2. La VRAM (900GB/s) reste plus rapide             │
│     que la RAM (50GB/s) même avec                   │
│     des accès non coalescés                         │
│                                                     │
│  3. Le résultat est DÉJÀ sur GPU                    │
│     pour les couches suivantes                      │
│     (Linear, Attention...)                          │
│                                                     │
│  → Règle générale : garder TOUT sur GPU             │
│    pour éviter les allers-retours PCIe !            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---


# Ligne K : `F.softmax(logits, dim=-1)`

## 🎯 C'est quoi Softmax ?

> Softmax transforme des scores bruts (logits)
> en **probabilités** qui somment à 1

### La formule mathématique

```
         e^(xi)
P(xi) = ────────
        Σ e^(xj)
         j

Pour chaque valeur xi :
→ on calcule e^xi
→ on divise par la somme de tous les e^xj
```

---

## 📊 Exemple Simple 1D

```
logits = [ 1.2,  0.5,  2.1, -0.3 ]

ÉTAPE 1 : Calcul des exponentielles
─────────────────────────────────────
e^1.2  =  3.32
e^0.5  =  1.65
e^2.1  =  8.17
e^-0.3 =  0.74
          ────
somme  = 13.88

ÉTAPE 2 : Division par la somme
─────────────────────────────────────
3.32  / 13.88 = 0.24  (24%)
1.65  / 13.88 = 0.12  (12%)
8.17  / 13.88 = 0.59  (59%)
0.74  / 13.88 = 0.05  ( 5%)
               ──────
somme        = 1.00  ✅

probs = [ 0.24,  0.12,  0.59,  0.05 ]
```

---

## 🎯 C'est quoi `dim=-1` ?

> `dim` indique **sur quelle dimension**
> on applique le softmax
> c'est à dire **où on fait la somme**

### Les dimensions de notre tenseur

```
logits shape : (B, C)
               │  │
               │  └──► dim=1  ou  dim=-1  (pareil !)
               └─────► dim=0
```

```
dim=-1  signifie :
→ "la dernière dimension"
→ toujours valide peu importe
   le nb de dimensions du tenseur !

(B, C)        → dim=-1 = dim=1
(B, T, C)     → dim=-1 = dim=2
(B, T, H, C)  → dim=-1 = dim=3
```

---

## 📊 Visualisation sur notre tenseur (B, C)

```
logits shape : (2, 5)
B = 2 séquences
C = 5 classes (vocabulaire)

        c0      c1      c2      c3      c4
        ──────  ──────  ──────  ──────  ──────
batch_0 [ 1.2    0.5     2.1    -0.3     0.8 ]
batch_1 [ 0.3    1.8    -0.5     0.9     0.2 ]
```

### dim=0 → softmax sur les LIGNES (mauvais ici !)

```
On comparerait batch_0 et batch_1 entre eux :

        c0              c1              c2
        ──────────      ──────────      ──────────
        e^1.2           e^0.5           e^2.1
batch_0 ──────          ──────          ──────
        e^1.2+e^0.3     e^0.5+e^1.8     e^2.1+e^-0.5

        e^0.3           e^1.8           e^-0.5
batch_1 ──────          ──────          ──────
        e^1.2+e^0.3     e^0.5+e^1.8     e^2.1+e^-0.5

❌ MAUVAIS : on compare des séquences différentes !
   ça n'a aucun sens ici
```

### dim=-1 → softmax sur les COLONNES (correct !)

```
On calcule les probs INDÉPENDAMMENT pour chaque séquence :

        c0      c1      c2      c3      c4       somme
        ──────  ──────  ──────  ──────  ──────   ─────
batch_0 [ 0.24   0.12    0.59    0.05    0.16 ]  = 1.0 ✅
batch_1 [ 0.12   0.55    0.05    0.22    0.11 ]  = 1.0 ✅
           │                      │
           │←─── dim=-1 ─────────►│
           softmax calculé
           sur cette direction

✅ BON : chaque séquence a ses propres probabilités
```


## 🔄 Dans le contexte de generate()

```
AVANT softmax
─────────────────────────────────────────────
logits[:, -1, :] shape : (B, C)

        "je"   "aime"   "les"  "chats"  "mange"
        ──────  ──────  ──────  ──────   ──────
batch_0 [ 1.2    0.5     2.1    -0.3      0.8 ]  scores bruts
batch_1 [ 0.3    1.8    -0.5     0.9      0.2 ]  scores bruts

APRÈS softmax dim=-1
─────────────────────────────────────────────
probs shape : (B, C)

        "je"   "aime"   "les"  "chats"  "mange"
        ──────  ──────  ──────  ──────   ──────
batch_0 [ 0.24   0.12    0.59    0.05     0.16 ]  somme = 1 ✅
batch_1 [ 0.12   0.55    0.05    0.22     0.11 ]  somme = 1 ✅
           │←────────────dim=-1────────────────►│
```

---

## ⚙️ CPU/GPU - Ce qui se passe dans la machine

```
┌─────────────────────────────────────────────────────┐
│                    CPU PIPELINE                     │
│                                                     │
│  FETCH   : charge l'instruction softmax             │
│  DECODE  : identifie 3 opérations                   │
│            1. exp()  → FPU (Float Point Unit)       │
│            2. sum()  → ALU + réduction              │
│            3. div()  → FPU                          │
│  EXECUTE : lance le kernel CUDA sur GPU             │
└─────────────────────────────────────────────────────┘
# Suite : ⚙️ CPU/GPU pour F.softmax

## 🖥️ GPU EXECUTION - Détail complet

```
┌─────────────────────────────────────────────────────┐
│                    GPU EXECUTION                    │
│                                                     │
│  ÉTAPE 1 : exp() → COMPUTE-BOUND                    │
│  ┌─────────────────────────────────────────────┐    │
│  │ Warp (32 threads en parallèle)              │    │
│  │                                             │    │
│  │ thread_0 → e^1.2   = 3.32                  │    │
│  │ thread_1 → e^0.5   = 1.65                  │    │
│  │ thread_2 → e^2.1   = 8.17                  │    │
│  │ thread_3 → e^-0.3  = 0.74                  │    │
│  │ ...                                         │    │
│  │                                             │    │
│  │ ✅ Tous les exp() calculés EN PARALLÈLE     │    │
│  │ ✅ Accès COALESCÉS car valeurs contiguës    │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ÉTAPE 2 : sum() → RÉDUCTION                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ Parallel Reduction Tree                     │    │
│  │                                             │    │
│  │ round 1 :                                   │    │
│  │ 3.32+1.65=4.97    8.17+0.74=8.91            │    │
│  │    │                  │                     │    │
│  │ round 2 :             │                     │    │
│  │ 4.97   +          8.91  = 13.88             │    │
│  │                    │                        │    │
│  │                  somme = 13.88              │    │
│  │                                             │    │
│  │ ✅ log2(N) étapes au lieu de N              │    │
│  │    ex: 1024 valeurs → 10 étapes seulement   │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ÉTAPE 3 : div() → COMPUTE-BOUND                    │
│  ┌─────────────────────────────────────────────┐    │
│  │ Warp (32 threads en parallèle)              │    │
│  │                                             │    │
│  │ thread_0 → 3.32  / 13.88 = 0.24            │    │
│  │ thread_1 → 1.65  / 13.88 = 0.12            │    │
│  │ thread_2 → 8.17  / 13.88 = 0.59            │    │
│  │ thread_3 → 0.74  / 13.88 = 0.05            │    │
│  │ ...                                         │    │
│  │                                             │    │
│  │ ✅ Toutes les divisions EN PARALLÈLE        │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```


## 💾 MÉMOIRE GPU - Les transferts

```
┌─────────────────────────────────────────────────────┐
│              HIÉRARCHIE MÉMOIRE GPU                 │
│                                                     │
│  Registers (~1 cycle)                               │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ valeurs exp() intermédiaires              │   │
│  │ ✅ somme partielle par thread                │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  Shared Memory (~5 cycles)                          │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ somme totale partagée entre threads       │   │
│  │    du même block                             │   │
│  │ ✅ résultats intermédiaires de réduction     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  L1/L2 Cache (~20-100 cycles)                       │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ logits en entrée                          │   │
│  │ ✅ probs en sortie                           │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  VRAM (~700 cycles)                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ tenseur logits complet (B, C)             │   │
│  │ ✅ tenseur probs complet  (B, C)             │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

FLUX MÉMOIRE :
──────────────
VRAM → L2 → L1 → Shared → Registers
  (logits)            (calculs)
                          │
                          ▼
VRAM ← L2 ← L1 ← Shared ← Registers
  (probs)             (résultats)
```

---

## ⚠️ Le problème numérique : Overflow !

```
PROBLÈME :
──────────
e^1000 = ∞  → overflow !

Si logits contient de grandes valeurs
→ exp() explose !

SOLUTION : Softmax numériquement stable
─────────────────────────────────────────

         e^(xi - max(x))
P(xi) = ─────────────────
        Σ e^(xj - max(x))
          j

EXEMPLE :
logits  = [ 1000,  1001,  1002 ]
max     = 1002

logits - max = [ -2,  -1,   0 ]

e^-2  = 0.135
e^-1  = 0.368
e^0   = 1.000
        ─────
somme = 1.503

probs = [ 0.09,  0.24,  0.67 ]  ✅ pas d'overflow !

→ PyTorch fait ça automatiquement ! 🎉
```

---

## 📊 Softmax est-il MEMORY ou COMPUTE bound ?

```
Pour logits shape (B, C) = (32, 50000) :

Données lues/écrites :
───────────────────────
logits  : 32 × 50000 × 4 bytes = 6.4 MB  (lecture)
probs   : 32 × 50000 × 4 bytes = 6.4 MB  (écriture)
total   : ~12.8 MB

Calculs effectués :
────────────────────
exp()  : 32 × 50000        = 1.6M  FLOPS
sum()  : 32 × 50000        = 1.6M  FLOPS
div()  : 32 × 50000        = 1.6M  FLOPS
total  : ~4.8M FLOPS

Arithmetic Intensity :
───────────────────────
4.8M FLOPS / 12.8MB = 0.375 FL




# `torch.multinomial` - L'échantillonnage

## 🎯 C'est quoi ?

> `torch.multinomial` tire aléatoirement un index
> en respectant les probabilités de chaque classe
> Plus une classe est probable, plus elle a de chances d'être tirée

---

## 📊 La différence fondamentale : argmax vs multinomial

```
probs = [ 0.40,  0.35,  0.15,  0.10 ]
          "le"  "chat"  "chien" "rat"

ARGMAX                        MULTINOMIAL
──────────────────────        ──────────────────────
→ prend TOUJOURS le max       → tire selon les probs

résultat = "le" (0.40)        résultat possible :
à chaque fois !               "le"    40% du temps
                              "chat"  35% du temps
❌ texte répétitif            "chien" 15% du temps
❌ pas de créativité          "rat"   10% du temps
❌ bloqué dans une
   seule séquence             ✅ texte varié
                              ✅ créatif
                              ✅ plusieurs histoires
                                 possibles
```

---

## 🎰 Comment fonctionne multinomial ?

### La Roulette des probabilités

```
probs = [ 0.40,  0.35,  0.15,  0.10 ]
          "le"  "chat"  "chien" "rat"

On construit une "roulette" :
─────────────────────────────────────────────────────
│      "le"      │    "chat"    │  "chien" │  "rat" │
│     (0.40)     │    (0.35)    │  (0.15)  │ (0.10) │
0              0.40           0.75       0.90      1.0
─────────────────────────────────────────────────────

On tire un nombre aléatoire entre 0 et 1 :
→ 0.23  tombe dans "le"    ✅
→ 0.67  tombe dans "chat"  ✅
→ 0.82  tombe dans "chien" ✅
→ 0.95  tombe dans "rat"   ✅
```

---

## 🔢 Le calcul pas à pas

### Étape 1 : Cumulative Sum (CDF)

```
probs = [ 0.40,  0.35,  0.15,  0.10 ]

CDF   = [ 0.40,  0.75,  0.90,  1.00 ]
          │      │      │      │
          │      │      │      └──► 0.40+0.35+0.15+0.10
          │      │      └─────────► 0.40+0.35+0.15
          │      └────────────────► 0.40+0.35
          └───────────────────────► 0.40

┌────────────────────────────────────────────────────┐
│  0    0.40   0.75   0.90   1.00                    │
│  │     │      │      │      │                      │
│  ├─────┼──────┼──────┼──────┤                      │
│  │ "le"│"chat"│"chien│ "rat"│                      │
│  └─────┴──────┴──────┴──────┘                      │
└────────────────────────────────────────────────────┘
```

### Étape 2 : Tirage aléatoire

```
u = random(0, 1) = 0.67   ← nombre uniforme aléatoire

CDF   = [ 0.40,  0.75,  0.90,  1.00 ]
                   ▲
                   │
          0.67 < 0.75  → index 1 → "chat" ✅
```

### Étape 3 : Recherche de l'index

```
On cherche le PREMIER index où CDF[i] > u

u = 0.67

CDF[0] = 0.40  →  0.40 < 0.67  ❌ continue
CDF[1] = 0.75  →  0.75 > 0.67  ✅ STOP → index = 1 → "chat"
```

---

## 📊 Visualisation sur notre tenseur (B, C)

```
probs shape : (2, 5)

        "je"   "aime"   "les"  "chats"  "mange"
        ──────  ──────  ──────  ──────   ──────
batch_0 [ 0.05   0.10    0.60    0.15     0.10 ]
batch_1 [ 0.20   0.40    0.05    0.25     0.10 ]

torch.multinomial(probs, num_samples=1)

batch_0 :
─────────
u = 0.54
CDF = [0.05, 0.15, 0.75, 0.90, 1.00]
                    ▲
               0.54 < 0.75 → index 2 → "les" ✅

batch_1 :
─────────
u = 0.31
CDF = [0.20, 0.60, 0.65, 0.90, 1.00]
              ▲
         0.31 < 0.60 → index 1 → "aime" ✅

OUTPUT shape : (2, 1)
┌─────────┐
│ batch_0 │ → [ 2 ]   "les"
│ batch_1 │ → [ 1 ]   "aime"
└─────────┘
```

---

## 🎯 argmax vs multinomial en pratique

```
Séquence initiale : "le chat"

ARGMAX (déterministe)          MULTINOMIAL (stochastique)
──────────────────────         ──────────────────────────

run_1 : "le chat mange         run_1 : "le chat mange
         mange mange mange"             dans le jardin"
                               
run_2 : "le chat mange         run_2 : "le chat dort
         mange mange mange"             sur le canapé"

run_3 : "le chat mange         run_3 : "le chat joue
         mange mange mange"             avec une balle"

❌ toujours pareil !           ✅ varié et créatif !
❌ boucle infinie possible     ✅ plusieurs histoires
```

---

## ⚙️ CPU/GPU - Ce qui se passe dans la machine

```
┌─────────────────────────────────────────────────────┐
│                   CPU PIPELINE                      │
│                                                     │
│  FETCH   : charge l'instruction multinomial         │
│  DECODE  : identifie 3 opérations                   │
│            1. random()  → RNG (Random Number Gen)   │
│            2. cumsum()  → réduction parallèle       │
│            3. search()  → recherche binaire         │
│  EXECUTE : lance le kernel CUDA sur GPU             │
└─────────────────────────────────────────────────────┘

# Suite : ⚙️ GPU EXECUTION pour torch.multinomial

## 🖥️ GPU EXECUTION - Détail complet

```
┌─────────────────────────────────────────────────────┐
│                   GPU EXECUTION                     │
│                                                     │
│  ÉTAPE 1 : Génération nombres aléatoires (RNG)      │
│  ┌─────────────────────────────────────────────┐    │
│  │ Philox RNG (algorithme CUDA natif)          │    │
│  │                                             │    │
│  │ thread_0 → u_0 = 0.54  (pour batch_0)      │    │
│  │ thread_1 → u_1 = 0.31  (pour batch_1)      │    │
│  │                                             │    │
│  │ ✅ généré EN PARALLÈLE pour tout le batch   │    │
│  │ ✅ reproductible si torch.manual_seed()     │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ÉTAPE 2 : Cumulative Sum (CDF)                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Parallel Prefix Sum (scan algorithm)        │    │
│  │                                             │    │
│  │ probs  = [0.05, 0.10, 0.60, 0.15, 0.10]    │    │
│  │                                             │    │
│  │ round 1 (additions par paires) :            │    │
│  │ [0.05, 0.15, 0.60, 0.75, 0.10]             │    │
│  │                                             │    │
│  │ round 2 :                                   │    │
│  │ [0.05, 0.15, 0.75, 0.90, 0.10]             │    │
│  │                                             │    │
│  │ round 3 :                                   │    │
│  │ [0.05, 0.15, 0.75, 0.90, 1.00]  ✅         │    │
│  │                                             │    │
│  │ ✅ log2(N) étapes au lieu de N              │    │
│  │    ex: 50000 classes → 16 étapes seulement  │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ÉTAPE 3 : Recherche Binaire                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ Binary Search sur la CDF                    │    │
│  │                                             │    │
│  │ CDF = [0.05, 0.15, 0.75, 0.90, 1.00]       │    │
│  │ u   = 0.54                                  │    │
│  │                                             │    │
│  │ step 1 : milieu = index 2 → 0.75           │    │
│  │          0.54 < 0.75 → cherche à gauche    │    │
│  │                                             │    │
│  │ step 2 : milieu = index 1 → 0.15           │    │
│  │          0.54 > 0.15 → cherche à droite    │    │
│  │                                             │    │
│  │ step 3 : index 2 → 0.75                    │    │
│  │          0.54 < 0.75 → TROUVÉ ! index=2    │    │
│  │                                             │    │
│  │ ✅ log2(N) comparaisons au lieu de N        │    │
│  │    ex: 50000 classes → 16 comparaisons      │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 💾 MÉMOIRE GPU

```
┌─────────────────────────────────────────────────────┐
│              HIÉRARCHIE MÉMOIRE GPU                 │
│                                                     │
│  Registers (~1 cycle)                               │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ valeur u aléatoire par thread             │   │
│  │ ✅ index courant recherche binaire           │   │
│  │ ✅ sommes partielles CDF                     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  Shared Memory (~5 cycles)                          │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ CDF partagée entre threads du même block  │   │
│  │ ✅ résultats intermédiaires prefix sum       │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  VRAM (~700 cycles)                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ ✅ tenseur probs  (B, C) en entrée           │   │
│  │ ✅ tenseur idx_next (B, 1) en sortie         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

FLUX MÉMOIRE :
──────────────
VRAM ──► Shared ──► Registers    (lecture probs)
              │
              ▼
         calculs RNG
         cumsum
         binary search
              │
              ▼
Registers ──► VRAM               (écriture idx_next)
```

---

## 📊 Memory-bound ou Compute-bound ?

```
Pour probs shape (B, C) = (32, 50000) :

Données lues/écrites :
───────────────────────
probs    : 32 × 50000 × 4 bytes = 6.4 MB  (lecture)
idx_next : 32 × 1     × 4 bytes = 0.1 KB  (écriture)
total    : ~6.4 MB

Calculs effectués :
────────────────────
RNG      : 32 × 1          =    32  FLOPS
cumsum   : 32 × 50000      = 1.6M   FLOPS
search   : 32 × log2(50000)= ~512   FLOPS
total    : ~1.6M FLOPS

Arithmetic Intensity :
───────────────────────
1.6M FLOPS / 6.4MB = 0.25 FLOPS/byte

Seuil RTX 3090 = 38 FLOPS/byte

0.25 << 38  →  MEMORY-BOUND ❌
``


------------------


# `torch.manual_seed()` - La Graine du Hasard

## 🎯 Le problème fondamental

```
Sans seed :
───────────
run_1 → multinomial → "chat"
run_2 → multinomial → "mange"
run_3 → multinomial → "dort"

→ impossible de reproduire un résultat !
→ impossible de débugger !
→ impossible de comparer des expériences !

Avec seed :
───────────
torch.manual_seed(42)
run_1 → multinomial → "chat"

torch.manual_seed(42)
run_2 → multinomial → "chat"  ← même résultat !

torch.manual_seed(42)
run_3 → multinomial → "chat"  ← toujours pareil !
```

---

## 🎰 C'est quoi un "vrai" hasard vs pseudo-hasard ?

```
┌─────────────────────────────────────────────────────┐
│              VRAI HASARD                            │
│                                                     │
│  Source physique :                                  │
│  → bruit thermique                                  │
│  → désintégration radioactive                       │
│  → turbulence atmosphérique                         │
│                                                     │
│  0.54, 0.31, 0.89, 0.12, 0.67 ...                  │
│                                                     │
│  ✅ vraiment aléatoire                              │
│  ❌ non reproductible                               │
│  ❌ lent                                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              PSEUDO-HASARD (ce que fait PyTorch)    │
│                                                     │
│  Algorithme déterministe :                          │
│  → prend une graine (seed)                          │
│  → génère une suite de nombres                      │
│  → qui SEMBLENT aléatoires                          │
│                                                     │
│  seed=42 → 0.54, 0.31, 0.89, 0.12, 0.67 ...        │
│  seed=42 → 0.54, 0.31, 0.89, 0.12, 0.67 ...        │
│  seed=42 → 0.54, 0.31, 0.89, 0.12, 0.67 ...        │
│                                                     │
│  ✅ reproductible                                   │
│  ✅ rapide                                          │
│  ✅ suffisamment "aléatoire" pour le ML             │
└─────────────────────────────────────────────────────┘
```

---

## 🔢 Comment fonctionne Philox RNG (algo de CUDA)

```
SEED = 42  (notre graine)

┌─────────────────────────────────────────────────────┐
│              PHILOX RNG                             │
│                                                     │
│  State interne :                                    │
│  ┌────────────────────────────────────────────┐     │
│  │  counter : 0        key : 42               │     │
│  └────────────────────────────────────────────┘     │
│          │                                          │
│          ▼                                          │
│  ┌────────────────────────────────────────────┐     │
│  │  opérations de mélange (bijectives)        │     │
│  │  multiply → xor → multiply → xor ...      │     │
│  └────────────────────────────────────────────┘     │
│          │                                          │
│          ▼                                          │
│  nombre_1 = 0.54                                    │
│          │                                          │
│          ▼                                          │
│  counter : 1  (on incrémente)                       │
│          │                                          │
│          ▼                                          │
│  nombre_2 = 0.31                                    │
│          │                                          │
│          ▼                                          │
│  counter : 2  (on incrémente)                       │
│  ...                                                │
└─────────────────────────────────────────────────────┘

→ même seed = même séquence de nombres ✅
→ counter différent = nombre différent ✅
```

---

## 📊 Impact sur multinomial - Exemple concret

```python
probs = torch.tensor([
    [0.40, 0.35, 0.15, 0.10],  # batch_0
    [0.20, 0.45, 0.25, 0.10]   # batch_1
])
```

### Sans seed

```
RUN 1 :
────────────────────────────────────────
u_batch0 = 0.73  → index 2 → "les"
u_batch1 = 0.15  → index 0 → "je"
résultat = [[2], [0]]

RUN 2 :
────────────────────────────────────────
u_batch0 = 0.21  → index 0 → "chat"
u_batch1 = 0.88  → index 3 → "rat"
résultat = [[0], [3]]

RUN 3 :
────────────────────────────────────────
u_batch0 = 0.56  → index 1 → "aime"
u_batch1 = 0.44  → index 1 → "aime"
résultat = [[1], [1]]

❌ résultats différents à chaque fois !
```

### Avec seed

```
torch.manual_seed(42)
RUN 1 :
────────────────────────────────────────
state = {counter:0, key:42}
u_batch0 = 0.73  → index 2 → "les"
u_batch1 = 0.15  → index 0 → "je"
résultat = [[2], [0]]  ✅

torch.manual_seed(42)
RUN 2 :
────────────────────────────────────────
state = {counter:0, key:42}  ← reset !
u_batch0 = 0.73  → index 2 → "les"
u_batch1 = 0.15  → index 0 → "je"
résultat = [[2], [0]]  ✅ pareil !

torch.manual_seed(42)
RUN 3 :
────────────────────────────────────────
state = {counter:0, key:42}  ← reset !
u_batch0 = 0.73  → index 2 → "les"
u_batch1 = 0.15  → index 0 → "je"
résultat = [[2], [0]]  ✅ toujours pareil !
```

---

## ⚠️ Les pièges du seed

### Piège 1 : CPU vs GPU

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  torch.manual_seed(42)      → seed CPU              │
│  torch.cuda.manual_seed(42) → seed GPU              │
│                                                     │
│  ⚠️ Ce sont 2 états RNG SÉPARÉS !                   │
│                                                     │
│  # Pour tout fixer d'un coup :                      │
│  torch.manual_seed(42)                              │
│  torch.cuda.manual_seed(42)                         │
│  torch.cuda.manual_seed_all(42)  # multi-GPU        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Piège 2 : L'ordre des opérations

```
torch.manual_seed(42)
a = torch

# La Seed dans PyTorch - Le Mécanisme Complet

## 🎯 Oui ! Mais voilà comment ça marche vraiment

```
PyTorch maintient un STATE RNG global
qui évolue en permanence

┌─────────────────────────────────────────────────────┐
│              STATE RNG GLOBAL PyTorch               │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  counter : 0                                 │   │
│  │  key     : ???  ← initialisé au démarrage    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  À chaque démarrage Python :                        │
│  → key est initialisé avec l'horloge système        │
│  → ou /dev/urandom (source physique)                │
│  → c'est pour ça que c'est différent à chaque fois !│
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Le cycle de vie du State RNG

```
DÉMARRAGE PYTHON
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  State initialisé ALÉATOIREMENT                     │
│  key = horloge système → ex: 1703589234             │
│  counter = 0                                        │
└─────────────────────────────────────────────────────┘
      │
      ▼
  opération_1 (ex: torch.rand)
  → utilise state courant
  → génère nombre
  → counter += 1
      │
      ▼
  opération_2 (ex: multinomial)
  → utilise state courant
  → génère nombre
  → counter += 1
      │
      ▼
  opération_3 ...
  → counter += 1
      │
      ▼
  ...à l'infini
```

---

## 📊 Sans seed vs Avec seed

```
SANS SEED
──────────────────────────────────────────────────────
Démarrage 1 :
  key = 1703589234  (horloge système)
  rand() → 0.54
  rand() → 0.31
  rand() → 0.89

Démarrage 2 :
  key = 1703589235  (1 seconde plus tard !)
  rand() → 0.21
  rand() → 0.67
  rand() → 0.43

❌ jamais pareil !

AVEC SEED
──────────────────────────────────────────────────────
torch.manual_seed(42)
  key = 42  (on FORCE la valeur)
  counter = 0  (on RESET)
  rand() → 0.54  ← toujours pareil
  rand() → 0.31  ← toujours pareil
  rand() → 0.89  ← toujours pareil

torch.manual_seed(42)
  key = 42  (on FORCE encore)
  counter = 0  (on RESET encore)
  rand() → 0.54  ← identique !
  rand() → 0.31  ← identique !
  rand() → 0.89  ← identique !

✅ toujours pareil !
```

---

## 🔍 Ce qui consomme le State RNG

```
TOUTES ces opérations font avancer le counter :

┌─────────────────────────────────────────────────────┐
│  torch.rand()          → nombres uniformes [0,1]    │
│  torch.randn()         → distribution normale       │
│  torch.randint()       → entiers aléatoires         │
│  torch.multinomial()   → sampling                   │
│  nn.Embedding()        → init aléatoire des poids   │
│  nn.Linear()           → init aléatoire des poids   │
│  nn.Dropout()          → masque aléatoire           │
│  torch.randperm()      → permutation aléatoire      │
└─────────────────────────────────────────────────────┘

→ chacune fait avancer le counter !
→ l'ordre des opérations est CRUCIAL
   pour la reproductibilité
```

---

## ⚠️ Le piège de l'ordre

```python
torch.manual_seed(42)

# CAS 1 :
a = torch.rand(3)      # counter : 0 → 3
b = torch.rand(2)      # counter : 3 → 5
c = torch.multinomial(probs, 1)  # counter : 5 → 6

# CAS 2 : (même seed mais ordre différent !)
torch.manual_seed(42)
b = torch.rand(2)      # counter : 0 → 2
a = torch.rand(3)      # counter : 2 → 5
c = torch.multinomial(probs, 1)  # counter : 5 → 6

┌─────────────────────────────────────────────────────┐
│  CAS 1 : c utilise counter=5                        │
│  CAS 2 : c utilise counter=5                        │
│                                                     │
│  ✅ même résultat ici car même nb total             │
│     d'opérations avant c                            │
│                                                     │
│  ⚠️ MAIS si on ajoute une opération entre les deux  │
│     le résultat de c change !                       │
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ CPU vs GPU : 2 States séparés !

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   CPU State RNG          GPU State RNG              │
│   ┌────────────┐         ┌────────────┐             │
│   │ counter: 0 │         │ counter: 0 │             │
│   │ key: ???   │         │ key: ???   │             │
│   └────────────┘         └────────────┘             │
│         │                      │                    │
│         │                      │                    │
│   opérations CPU          opérations GPU            │
│   torch.rand()            torch.rand().cuda()       │
│   torch.randint()         nn.Dropout() sur GPU      │
│   ...                     torch.multinomial()       │
│                            sur GPU                  │
│                                                     │
│  ⚠️ CE SONT 2 ÉTATS INDÉPENDANTS !                  │
└─────────────────────────────────────────────────────┘

# Pour tout fixer :
┌─────────────────────────────────────────────────────┐
│                                                     │
│  torch.manual_seed(42)           # CPU              │
│  torch.cuda.manual_seed(42)      # GPU actif        │
│  torch.cuda.manual_seed_all(42)  # tous les GPU     │
│                                                     │
│  # ou en une ligne :                                │
│  torch.manual_seed(42)                              │
│  → depuis PyTorch 1.9                               │
│     fixe aussi le GPU automatiquement ! ✅          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---


# `torch.cat` - La Concaténation

## 🎯 C'est quoi ?

> `torch.cat` colle plusieurs tenseurs ensemble
> le long d'une dimension choisie
> **sans créer de nouvelle dimension**

---

## 📊 Notre situation

```python
idx      shape : (B, T)    # séquence courante
idx_next shape : (B, 1)    # nouveau token généré

idx = torch.cat((idx, idx_next), dim=1)
```

```
B = 2  → 2 séquences
T = 4  → 4 tokens dans la séquence courante

idx :                          idx_next :
shape (2, 4)                   shape (2, 1)

┌─────────────────────┐        ┌─────┐
│ batch_0 [2, 5, 1, 3]│        │ [7] │  batch_0
│ batch_1 [1, 4, 6, 2]│        │ [3] │  batch_1
└─────────────────────┘        └─────┘
```

---

## 🔄 Ce qui se passe avec `dim=1`

```
dim=1 → on colle sur la dimension T (colonnes)

AVANT                          APRÈS
──────────────────────         ──────────────────────────
idx      (2, 4)                idx (2, 5)
idx_next (2, 1)

        T0  T1  T2  T3                T0  T1  T2  T3  T4
        ──  ──  ──  ──                ──  ──  ──  ──  ──
batch_0 [2,  5,  1,  3]  +  [7]  →  [2,  5,  1,  3,  7]
batch_1 [1,  4,  6,  2]  +  [3]  →  [1,  4,  6,  2,  3]

                                      ▲               ▲
                                  anciens         nouveau
                                  tokens           token
```

---

## 🔁 L'évolution sur toute la génération

```
Séquence initiale :
idx = [[2, 5]]   shape (1, 2)
       "le chat"

──────────────────────────────────────────────────────
ITÉRATION 1 :
  idx_next = [[1]]  → "mange"
  idx = cat([[2,5]], [[1]], dim=1)
  idx = [[2, 5, 1]]  shape (1, 3)
         "le chat mange"

──────────────────────────────────────────────────────
ITÉRATION 2 :
  idx_next = [[4]]  → "des"
  idx = cat([[2,5,1]], [[4]], dim=1)
  idx = [[2, 5, 1, 4]]  shape (1, 4)
         "le chat mange des"

──────────────────────────────────────────────────────
ITÉRATION 3 :
  idx_next = [[3]]  → "souris"
  idx = cat([[2,5,1,4]], [[3]], dim=1)
  idx = [[2, 5, 1, 4, 3]]  shape (1, 5)
         "le chat mange des souris"

──────────────────────────────────────────────────────
→ shape finale : (B, T + max_new_tokens)
```

---

## 🆚 dim=0 vs dim=1

```
idx      = [[2, 5, 1, 3],    shape (2, 4)
             [1, 4, 6, 2]]

idx_next = [[7],              shape (2, 1)
             [3]]

dim=0 → colle sur les LIGNES (batch)
─────────────────────────────────────
résultat shape (4, 4) ❌

[[2, 5, 1, 3],
 [1, 4, 6, 2],
 [7, ?, ?, ?],   ← ERREUR ! dimensions incompatibles
 [3, ?, ?, ?]]

dim=1 → colle sur les COLONNES (temps) ✅
──────────────────────────────────────────
résultat shape (2, 5) ✅

[[2, 5, 1, 3, 7],
 [1, 4, 6, 2, 3]]
```

---

## ⚙️ CPU/GPU - Ce qui se passe dans la machine

```
┌─────────────────────────────────────────────────────┐
│                   CPU PIPELINE                      │
│                                                     │
│  FETCH   : charge l'instruction cat                 │
│  DECODE  : identifie l'opération                    │
│            → alloue nouveau buffer mémoire          │
│            → calcule la nouvelle shape (2, T+1)     │
│  EXECUTE : lance la copie mémoire sur GPU           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   GPU EXECUTION                     │
│                                                     │
│  ÉTAPE 1 : Allocation nouveau tenseur               │
│  ┌─────────────────────────────────────────────┐    │
│  │ VRAM : réserve un bloc contigu              │    │
│  │ shape (2, T+1) × 4 bytes                   │    │
│  │                                             │    │
│  │ ⚠️ NOUVEAU BUFFER à chaque itération !      │    │
│  │    coût mémoire croissant !                 │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ÉTAPE 2 : Copie des données                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ thread_0 → copie batch_0 de idx             │    │
│  │ thread_1 → copie batch_1 de idx             │    │
│  │ thread_2 → copie batch_0 de idx_next        │    │
│  │ thread_3 → copie batch_1 de idx_next        │    │
│  │                                             │    │
│  │ ✅ copies EN PARALLÈLE                      │    │
│  │ ✅ accès COALESCÉS car données contiguës    │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 💾 MÉMOIRE GPU - Le coût croissant

```
VRAM utilisée à chaque itération :

iter_0 : idx shape (2, 4)   → 2×4×4   bytes = 32  bytes
iter_1 : idx shape (2, 5)   → 2×5×4   bytes = 40  bytes
iter_2 : idx shape (2, 6)   → 2×6×4   bytes = 48  bytes
...
iter_N : idx shape (2, 4+N) → 2×(4+N)×4 bytes

⚠️ À CHAQUE ITÉRATION :
──────────────────────────────────────────────────────
ancien idx  →  libéré de la VRAM  (garbage collector)
nouveau idx →  alloué  en  VRAM  (plus grand)
# Suite : 💾 MÉMOIRE GPU - Le coût croissant


```
VRAM
┌────────────────────────────────────────────────────┐
│                                                    │
│  iter_0 : [■■■■□□□□□□□□□□□□□□□□]  32  bytes       │
│  iter_1 : [■■■■■□□□□□□□□□□□□□□□]  40  bytes       │
│  iter_2 : [■■■■■■□□□□□□□□□□□□□□]  48  bytes       │
│  iter_3 : [■■■■■■■□□□□□□□□□□□□□]  56  bytes       │
│  ...                                               │
│  iter_N : [■■■■■■■■■■■■■■■■■■■■]  croissant !     │
│                                                    │
│  ⚠️ max_new_tokens = 1000 ?                        │
│  idx shape finale : (2, 1004)                      │
│  → 2 × 1004 × 4 bytes = ~8KB  (petit ici)         │
│                                                    │
│  ⚠️ En vrai avec GPT-3 :                           │
│  B=32, T=4096, embed=12288                         │
│  → beaucoup plus lourd !                           │
└────────────────────────────────────────────────────┘
```

---

## ⚠️ Le vrai problème de cat en boucle

```
for _ in range(max_new_tokens):
    ...
    idx = torch.cat((idx, idx_next), dim=1)  ⚠️


PROBLÈME : à chaque itération
─────────────────────────────────────────────────────

itération 1 :
  VRAM : alloue (2, 5)   ← nouveau buffer
         copie  (2, 4)   ← ancien idx
         copie  (2, 1)   ← idx_next
         libère (2, 4)   ← ancien idx supprimé

itération 2 :
  VRAM : alloue (2, 6)   ← nouveau buffer
         copie  (2, 5)   ← ancien idx
         copie  (2, 1)   ← idx_next
         libère (2, 5)   ← ancien idx supprimé

itération 3 :
  VRAM : alloue (2, 7)   ← nouveau buffer
         copie  (2, 6)   ← ancien idx
         copie  (2, 1)   ← idx_next
         libère (2, 6)   ← ancien idx supprimé

→ À chaque itération :
  ✅ 1 allocation
  ✅ 2 copies
  ✅ 1 libération

→ Sur 1000 itérations :
  ❌ 1000 allocations
  ❌ 2000 copies
  ❌ 1000 libérations
  = très coûteux en mémoire et en temps !
```

---

## 💡 La solution pro : pré-allouer

```python
# ❌ VERSION NAIVE (ce qu'on fait avec cat)
for _ in range(max_new_tokens):
    idx = torch.cat((idx, idx_next), dim=1)
    # nouvelle allocation à chaque fois !

# ✅ VERSION OPTIMISÉE (pré-allocation)
# on réserve tout l'espace dès le début !
output = torch.zeros(B, T + max_new_tokens,
                     dtype=torch.long)

# on copie le contexte initial
output[:, :T] = idx

# on remplit au fur et à mesure
for i in range(max_new_tokens):
    ...
    output[:, T+i] = idx_next[:, 0]
    # pas de nouvelle allocation ! ✅
```

```
VRAM avec pré-allocation :
──────────────────────────────────────────────────────
début   : [■■■■□□□□□□□□□□□□□□□□]  alloué UNE FOIS

iter_0  : [■■■■■□□□□□□□□□□□□□□□]  on remplit
iter_1  : [■■■■■■□□□□□□□□□□□□□□]  on remplit
iter_2  : [■■■■■■■□□□□□□□□□□□□□]  on remplit
...
iter_N  : [■■■■■■■■■■■■■■■■■■■■]  complet !

✅ 1 seule allocation !
✅ pas de copie inutile !
✅ beaucoup plus rapide !
```

---

## 🎯 Résumé Complet

```
┌─────────────────────────────────────────────────────┐
│              torch.cat - Points clés                │
│                                                     │
│  CE QUE ÇA FAIT :                                   │
│  → colle des tenseurs le long d'une dimension       │
│  → pas de nouvelle dimension créée                  │
│  → dim=1 → on colle sur T (les tokens)              │
│                                                     │
│  DANS NOTRE CONTEXTE :                              │
│  → idx      (B, T)   séquence courante              │
│  → idx_next (B, 1)   nouveau token                  │
│  → résultat (B, T+1) séquence étendue               │
│                                                     │
│  SUR LE GPU :                                       │
│  → nouvelle allocation VRAM à chaque fois           │
│  → copie des données en parallèle                   │
│  → accès coalescés ✅                               │
│  → coût croissant sur N itérations ⚠️               │
│                                                     │
│  BONNE PRATIQUE :                                   │
│  → pré-allouer le buffer final                      │
│  → évite 1000 allocations inutiles                  │
│  → bien plus efficace en VRAM                       │
└─────────────────────────────────────────────────────┘
```