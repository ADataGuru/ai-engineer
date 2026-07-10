

# Qu'est-ce que `ioctl` ?

## La définition simple

`ioctl` = **Input/Output ConTroL**

C'est un appel système **générique** pour tout ce qui ne rentre pas dans le modèle classique `read`/`write`.

```
Appels système classiques :
├── open()   → ouvrir un fichier
├── read()   → lire des données
├── write()  → écrire des données
└── close()  → fermer

Mais pour un GPU, un périphérique réseau, etc... :
└── ioctl()  → "tout le reste" (configurer, commander, interroger...)
```

---

## La signature

```c
int ioctl(int fd, unsigned long request, void *argp);
//         ^        ^                     ^
//         |        |                     |
//    file           numéro de            pointeur vers
//    descriptor     commande             données (input ou output)
//    (le device)    (quoi faire ?)       selon la commande
```

---

## Analogie concrète

```
Imagine un ascenseur :

read()  = lire quel étage on est
write() = impossible, ça n'a pas de sens

ioctl() = tout le reste :
  ├── "monte au 3ème étage"
  ├── "ouvre les portes"
  ├── "quel est ton état ?"
  ├── "mode urgence"
  └── "calibre les capteurs"

Chaque commande a un numéro, et des données associées
```

---

## Pourquoi ça existe ?

```
Le kernel Linux modélise TOUT comme des fichiers :
┌─────────────────────────────────────────┐
│  /dev/nvidia0    → ton GPU              │
│  /dev/sda        → ton disque           │
│  /dev/tty0       → ton terminal         │
│  /dev/net/tun    → interface réseau     │
│  /dev/input/mice → ta souris            │
└─────────────────────────────────────────┘

read/write = ok pour fichiers normaux
Mais un GPU ça ne se "lit" pas comme un fichier texte !
→ ioctl() est la "trappe d'évacuation" pour commandes custom
```

---

## Exemple simple : terminal

```c
#include <sys/ioctl.h>
#include <stdio.h>

int main() {
    struct winsize w;
    
    // "Hey terminal (fd=0), donne moi ta taille"
    ioctl(0, TIOCGWINSZ, &w);
    //    ^   ^
    //    │   └── TIOCGWINSZ = "Terminal IO Get WINdow SiZe"
    //    └────── stdout = le terminal
    
    printf("Colonnes: %d, Lignes: %d\n", w.ws_col, w.ws_row);
}
```

---

## Décomposer un ioctl NVIDIA

```
ioctl(8, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), ptr)
         └──────────────────────────────────────────┘
                    request = un entier encodé

_IOC(direction,    type,   numéro, taille)
     ─────────     ────    ──────  ─────
     READ|WRITE    0x46    0x2a    0x20 = 32 bytes
     ↓             ↓       ↓
     on lit ET      'F'    commande #42
     on écrit       =       = ALLOC_MEMORY
                   NVIDIA
```

```
En binaire, le numéro request est encodé sur 32 bits :

 31      30-29    28-16      15-8     7-0
┌──────┬────────┬─────────┬────────┬────────┐
│ mode │  sens  │ taille  │  type  │  cmd   │
│      │R/W/RW  │données  │ 0x46   │ 0x2a   │
└──────┴────────┴─────────┴────────┴────────┘
```

---

## Le flow complet pour ton GPU

```
Python : torch.matmul(x, y)
    │
    ▼
PyTorch C++ (ATen)
    │
    ▼
libcuda.so / libcudart.so
    │  "Lance ce kernel avec ces paramètres"
    ▼
ioctl(fd, NV_ESC_RM_CONTROL, {
    cmd: LAUNCH_KERNEL,
    data: {
        gridDim: (32, 1, 1),
        blockDim: (256, 1, 1),
        kernel_ptr: 0x...,
        args: [...]
    }
})
    │
    ▼
Kernel Linux (driver nvidia.ko)
    │  décode la commande
    ▼
Hardware GPU
```

---

## Pourquoi pas juste write() ?

```
# Théoriquement on pourrait faire :
write(fd_gpu, kernel_data, size)

# Mais ioctl est mieux parce que :

1. BIDIRECTIONNEL en un seul appel
   write() → envoie données
   read()  → reçoit données
   ioctl() → envoie ET reçoit en même temps
             (ex: "lance kernel" + "retourne handle")

2. TYPÉ / STRUCTURÉ
   write() = juste des bytes bruts
   ioctl() = commande numérotée + struct typée

3. ATOMIQUE
   une seule opération noyau = pas de race condition
   entre l'envoi de la commande et la réponse

4. CONVENTION UNIX
   "tout est fichier" mais avec des opérations custom
```

---

## Résumé en une phrase

> `ioctl` c'est comme un **appel de fonction à distance vers le kernel** : tu donnes un numéro de commande + des données, le driver fait son travail dans le ring 0, et te répond.

```
userspace          kernel space
─────────          ────────────
ioctl()     ──→    driver reçoit
                   décode commande
                   parle au hardware
           ←──     retourne résultat
```