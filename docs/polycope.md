# Polycopié de Cours : Flash Attention 2 et Programmation CUDA

## Page de Titre

**Calcul Haute Performance sur GPU : De la Théorie CUDA à Flash Attention 2**

Polycopié de cours exhaustif couvrant les fondations théoriques et architecturales pour comprendre et implémenter des algorithmes d'attention optimisés sur processeurs graphiques NVIDIA.

**Public :** Débutants en programmation GPU ayant des notions de C++ et d'algèbre linéaire.

**Contenu :** 10 chapitres théoriques couvrant l'architecture GPU, le modèle de programmation CUDA, la précision numérique, le mécanisme d'attention, et les optimisations de Flash Attention 2.

**Niveau :** Baccalauréat + 2 années (L3/Master 1)

**Prérequis :** C++, programmation concurrente, algèbre linéaire matricielle, notions de transformers.

**Date :** Février 2026

---

## Table des matières

**PARTIE 1 : Fondations Théoriques (Chapitres 1-10)**

1. [Du CPU au GPU : Paradigme fondamental](#ch1)
2. [Architecture GPU NVIDIA : Le cœur du calcul massivement parallèle](#ch2)
3. [Le modèle de programmation CUDA : Grilles, blocs et threads](#ch3)
4. [Précision numérique et formats virgule flottante](#ch4)
5. [Le mécanisme d'attention : Le cœur des transformers](#ch5)
6. [Softmax : Stabilité numérique et algorithmes online](#ch6)
7. [Le Roofline Model : Prédire les performances](#ch7)
8. [Flash Attention : L'algorithme révolutionnaire](#ch8)
9. [Mémoire partagée, conflits bancaires et primitives warp](#ch9)
10. [Triton : Abstraire le CUDA pour le rendre accessible](#ch10)

---

## Chapitre 1 : Du CPU au GPU — Paradigme fondamental {#ch1}

### 1.1 Anatomie du processeur CPU moderne

Un **processeur central (CPU)** est un composant électronique qui exécute des instructions. Pour comprendre pourquoi nous avons besoin des GPU, il faut d'abord comprendre comment les CPU modernes sont conçus et quelles sont leurs limitations fondamentales.

Un CPU moderne typique fonctionne selon un modèle appelé **pipeline d'instructions**. Imaginez une chaîne de montage dans une usine automobile : chaque instruction (comme l'addition de deux nombres) passe par plusieurs étapes successives. D'abord, l'instruction est **récupérée** depuis la mémoire (fetch). Ensuite, elle est **décodée** pour comprendre quel calcul faire. Puis, les opérandes (les nombres à additionner) sont **rassemblés**. Après cela, l'opération est **exécutée**. Enfin, le résultat est **rangé** en mémoire ou dans un registre. Chaque étape peut être effectuée par un composant différent du processeur, qui travaillent tous en parallèle sur des instructions différentes.

Les CPU modernes (comme les Intel Core ou AMD Ryzen) possèdent entre 2 et 16 cœurs physiques. Chaque cœur peut exécuter un thread d'exécution indépendant. Grâce à la technique appelée **hyper-threading** (chez Intel) ou **simultaneous multithreading** (SMT chez AMD), un seul cœur physique peut exécuter plusieurs threads logiques. Cependant, même avec cette technologie, nous ne parlons que de quelques dizaines de threads en parallèle maximum — pas des milliers.

Les CPU modernes possèdent une **hiérarchie de caches** très sophistiquée. Les registres sont les zones de stockage les plus rapides, intégrées directement dans le cœur du processeur. Ils peuvent transférer des données en 1 cycle d'horloge. Ensuite vient le **cache L1**, minuscule (32 KB) mais très rapide (3-4 cycles). Le **cache L2** est plus grand (256 KB) mais un peu plus lent (12 cycles). Enfin, le **cache L3** peut être partagé entre plusieurs cœurs, il est plus grand (8-20 MB) et plus lent (40 cycles). Au-delà de cette hiérarchie, la **mémoire vive (RAM)** est énorme mais lente — accéder à la RAM prend environ 100 cycles d'horloge. Le CPU passe une énorme quantité de temps à attendre les données : environ 90% du temps d'exécution est souvent dépensé en attentes de mémoire plutôt qu'en calcul réel.

Pour masquer cette latence de mémoire, les CPU modernes utilisent des mécanismes avancés. La **prédiction de branchement** essaie de deviner quelle instruction va être exécutée ensuite (par exemple, dans une boucle if-else, le CPU parie que la branche prise précédemment sera prise à nouveau). L'**exécution spéculative** commence à exécuter les instructions suivantes avant même que la condition soit confirmée, en espérant que la prédiction était correcte. Le **prefetching** essaie de charger les données en cache avant que le programme ne les demande réellement. Ces techniques sont essentielles parce que le CPU ne peut rien faire sans données — il doit attendre.

### 1.2 Différence fondamentale : Latence optimisée vs Débit optimisé

La conception des CPU modernes peut être résumée par une phrase : **optimiser pour minimiser la latence**. La **latence** est le temps total qu'il faut pour exécuter une instruction unique du début à la fin. Un CPU est conçu pour exécuter quelques threads très rapidement. Chaque instruction progresse aussi vite que possible à travers le pipeline, et les mécanismes sophistiqués que nous venons de décrire (caches, prédiction de branchement, etc.) sont tous là pour réduire l'attente entre les instructions.

À l'inverse, les GPU sont conçus pour optimiser le **débit** — la quantité de travail qui peut être accomplissé par unité de temps. Un GPU sacrifie la vitesse d'une instruction individuelle (une instruction peut prendre plus de cycles) pour pouvoir exécuter des milliers d'instructions en parallèle. C'est une hypothèse radicalement différente sur ce que constitue une "bonne" architecture.

Pensez à cette analogie : un CPU c'est comme un restaurant gastronomique avec un seul chef très rapide et très intelligent qui prépare un repas à la fois. Chaque plat individuel est complété en un temps record, mais le restaurant ne peut servir qu'une table à la fois. Un GPU, c'est comme une chaîne de montage d'assemblage automobile : il y a peut-être 10 000 ouvriers simples qui font chacun une tâche basique (boulonner une roue, par exemple), et même si chaque ouvrier individuel est lent par rapport au chef, ensemble ils peuvent assembler des milliers de voitures en parallèle.

### 1.3 Architecture GPU : Des milliers de cœurs simples

Un **processeur graphique (GPU)** est un coprocesseur conçu initialement pour le rendu graphique, mais qui s'est avéré être extraordinairement efficace pour le calcul massivement parallèle. Contrairement à un CPU qui possède quelques cœurs complexes avec beaucoup de logique de contrôle, un GPU possède **des milliers de petits cœurs simples** organisés en groupes.

Les cœurs d'un GPU ne possèdent pas les mécanismes sophistiqués d'un CPU. Pas de prédiction de branchement élaborée (les branches doivent être gérées manuellement par le programmeur). Pas de caches L1/L2 énormes (les caches GPU sont minuscules comparés aux CPU). Pas d'exécution spéculative (ce qui serait trop compliqué à gérer pour des milliers de cœurs). La philosophie est simple : chaque cœur est très basique, mais il y en a beaucoup.

Un GPU NVIDIA possède une **bande passante mémoire** énorme. Tandis qu'un CPU moderne peut transférer environ 100 GB/s de la RAM vers les registres, un GPU haut de gamme peut transférer plusieurs terabytes par seconde (TB/s). C'est une différence de facteur 10 à 20. Cette bande passante énorme est cruciale : elle compense le manque d'sophistication des caches en s'assurant que même si les données ne sont pas en cache, elles peuvent être chargées extrêmement rapidement.

### 1.4 Tableau comparatif : CPU vs GPU

| Caractéristique | CPU (Intel/AMD) | GPU (NVIDIA A100) |
|---|---|---|
| **Nombre de cœurs** | 8-16 | 6912 (108 SM × 64 cœurs par SM) |
| **Fréquence d'horloge** | 3-4 GHz | 1.4 GHz |
| **Mémoire cache (total)** | 8-20 MB | 192 KB shared × 108 + 40 MB L2 |
| **Latence d'accès RAM** | ~100 cycles (200 ns) | ~400 cycles (300 ns) en valeur absolue, mais cachée par le parallelisme |
| **Bande passante mémoire** | ~100 GB/s | ~2000 GB/s (2 TB/s) |
| **Nombre de threads en parallèle** | 16-32 | 6912 × 32 = 221 184 threads potentiels |
| **Peak FP32 FLOPS** | ~1 TFLOP/s | ~20 TFLOP/s |
| **Peak FP16 FLOPS** | ~4 TFLOP/s | ~160 TFLOP/s |
| **Puissance consommée** | 65-150 W | 400 W (mais pour ~10× plus de performance) |

Les chiffres de ce tableau révèlent des choix d'ingénierie radicalement différents. Le CPU a une fréquence 3× plus haute, mais beaucoup moins de cœurs. Le GPU a une bande passante 20× supérieure. Le CPU peut stocker plus de cache par cœur, mais le GPU peut exécuter des milliers de threads. C'est pourquoi les GPU dominent les tâches massivement parallèles comme le deep learning — ils sont conçus pour exactement ce type de charge de travail.

### 1.5 Quand utiliser GPU vs CPU ?

Cette question se résume à un concept fondamental : l'**intensité arithmétique**, que nous approfondirons au Chapitre 7. L'intuition est simple :

- **Tâches I/O bound (limitées par l'entrée/sortie) :** Si votre programme passe plus de temps à attendre des données du disque ou du réseau qu'à les traiter, le CPU avec ses caches sophistiqués est meilleur. Il peut démarrer le travail suivant immédiatement après que la première instruction attende les données.

- **Tâches compute-bound (limitées par le calcul) :** Si votre programme fait beaucoup de calculs sur les données qu'il a chargées, le GPU est meilleur. Vous pouvez cacher la latence de mémoire en exécutant d'autres threads pendant que certains attendent les données.

- **Tâches massivement parallèles :** Si votre travail peut être divisé en des millions de petites tâches indépendantes, le GPU peut les exécuter en parallèle tandis que le CPU en exécute quelques unes à la fois.

L'entraînement de réseaux de neurones profonds est une tâche compute-bound et massivement parallèle : la même opération (comme ajouter deux matrices ou calculer un produit scalaire) doit être répétée des milliards de fois. C'est le cas d'usage parfait pour un GPU.

---

## Chapitre 2 : Architecture GPU NVIDIA — Le cœur du calcul massivement parallèle {#ch2}

### 2.1 Hierarchie organisationnelle : Du GPU aux warps

Un GPU NVIDIA n'est pas simplement une collection aléatoire de cœurs. Il possède une hiérarchie d'organisation bien définie, comme une structure militaire où chaque niveau a des responsabilités claires.

Au niveau le plus bas se trouvent les **cœurs de calcul (CUDA cores)**. Un cœur CUDA est une unité de calcul très simple capable d'effectuer une seule opération (addition, multiplication, etc.) par cycle. Chaque cœur a ses propres registres (64 bits chacun) — ces minuscules zones de stockage dont nous avons parlé plus tôt. Avec 6912 cœurs, un GPU A100 peut faire 6912 additions simultanées.

Ces 6912 cœurs ne sont pas dispersés aléatoirement. Ils sont groupés en **warps** de 32 cœurs. Un **warp** est l'unité d'exécution élémentaire sur un GPU NVIDIA. Les 32 cœurs d'un warp exécutent **exactement la même instruction sur des données différentes**. C'est ce qu'on appelle le modèle **SIMT (Single Instruction, Multiple Thread)** — une instruction, plusieurs threads.

Pourquoi 32 ? C'est un choix historique et d'ingénierie. 32 threads par warp offre un bon équilibre entre la complexité du matériel et la bande passante. Avec 32 threads, vous pouvez charger efficacement une ligne de cache (qui fait généralement 128 bytes). De plus, 32 est une puissance de 2, ce qui facilite les opérations comme les réductions (nous verrons cela au Chapitre 9).

L'implication clé du modèle SIMT est que **tous les 32 threads d'un warp doivent exécuter la même instruction**. Si vous avez un if-else dans votre code CUDA, et que certains threads du warp prennent la branche if tandis que d'autres prennent la branche else, c'est un problème. Les threads qui prennent la branche if vont exécuter cette branche tandis que les autres sont "arrêtés" ou "masqués" en attente. Ensuite, les autres threads exécutent leur branche. Cela s'appelle une **divergence de warp** et c'est très coûteux en performance.

### 2.2 Streaming Multiprocessors (SM)

Les warps ne flottent pas librement dans le GPU. Ils sont organisés en **Streaming Multiprocessors (SM)**. Un SM est un petit processeur complet contenant un certain nombre de cœurs, de la mémoire locale, et de la logique de contrôle.

Un GPU A100 possède 108 SM. Chaque SM sur un A100 contient 64 cœurs CUDA (ce qui donne 108 × 64 = 6912 cœurs au total). Ces 64 cœurs peuvent exécuter 2 warps de 32 threads simultanément.

Chaque SM possède également une **mémoire partagée (shared memory)** très rapide. Sur un A100, chaque SM dispose de 192 KB de mémoire partagée. C'est une mémoire très rapide — à peu près aussi rapide que les registres — mais partagée entre tous les threads du SM. Nous approfondirons cela au Chapitre 9 car c'est crucial pour l'optimisation.

Chaque SM possède aussi des **registres**. Contrairement aux registres d'un CPU qui sont une ressource extrêmement limitée (peut-être 16 en total), un SM possède 256 KB de registres. Avec 65 536 registres possibles (256 KB / 4 bytes par registre), chaque thread peut théoriquement recevoir 256 KB / 6912 cœurs ≈ 38 registres. En pratique, ce nombre varie selon le nombre de threads que vous lancez.

Chaque SM possède aussi un **cache L1 de 192 KB** (sur A100) et un **cache L2 de 40 MB** est partagé entre tous les SM. Remarquez que contrairement à un CPU, le GPU n'investit pas énormément dans la cache. La philosophie est que la latence est masquée par le parallelisme plutôt que par la cache.

```
┌─────────────────────────────────────────────────────┐
│              GPU A100 (Vue simplifiée)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐        │
│  │ SM 0             │  │ SM 1             │  ...   │
│  ├──────────────────┤  ├──────────────────┤        │
│  │ 64 CUDA cores    │  │ 64 CUDA cores    │        │
│  │ (2 warps × 32)   │  │ (2 warps × 32)   │        │
│  ├──────────────────┤  ├──────────────────┤        │
│  │ Shared Mem       │  │ Shared Mem       │        │
│  │ 192 KB           │  │ 192 KB           │        │
│  │ (32 banks)       │  │ (32 banks)       │        │
│  ├──────────────────┤  ├──────────────────┤        │
│  │ Registers        │  │ Registers        │        │
│  │ 256 KB           │  │ 256 KB           │        │
│  ├──────────────────┤  ├──────────────────┤        │
│  │ L1 Cache 192 KB  │  │ L1 Cache 192 KB  │        │
│  └──────────────────┘  └──────────────────┘        │
│         (× 108 SM)                                  │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ L2 Cache (partagé) : 40 MB                  │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ HBM (High Bandwidth Memory) : 80 GB         │   │
│  │ Bande passante : 2 TB/s                     │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.3 Hiérarchie complète de mémoire

Comme promis, voici la hiérarchie complète de mémoire d'un GPU A100, du plus rapide au plus lent :

| Niveau | Taille | Latence | Bande passante | Scope | Notes |
|---|---|---|---|---|---|
| **Registres** | 256 KB par SM | 0 cycles (accès direct) | Illimitée | 1 thread | Chaque thread a ses propres registres |
| **Shared Memory** | 192 KB par SM | 1-3 cycles | ~10 TB/s | 1 bloc (32 threads × 32 = 1024 max) | Très rapide mais petite et partagée |
| **L1 Cache** | 192 KB par SM | 4 cycles | ~20 TB/s | 1 SM | Invisible au programmeur, automatique |
| **L2 Cache** | 40 MB (global) | 30 cycles | ~2 TB/s | Tout le GPU | Automatique, pour tous les SM |
| **HBM** | 80 GB (A100) | ~400 cycles | ~2 TB/s | Tout le GPU | Mémoire principale, très grande mais lente |

Les registres sont les plus rapides car il n'y a pas même un délai — c'est juste du stockage électronique dans le cœur du processeur. La shared memory est très rapide (quelques nanosecondes) mais vous devez y stocker les données explicitement dans votre code CUDA. Le cache L1/L2 fonctionne automatiquement comme sur un CPU, mais avec une approche moins agressive. La HBM (High Bandwidth Memory) est le goulet d'étranglement pour presque tous les noyaux GPU : c'est là que vivent vos tenseurs de millions de paramètres, et atteindre la bande passante de 2 TB/s est un objectif majeur.

### 2.4 Le concept d'occupancy (occupation mémoire)

L'**occupancy** ou **occupation** est un concept crucial pour la performance GPU. C'est le pourcentage de warps maximum qui sont actifs sur un SM donné, divisé par le nombre maximum de warps que le SM peut techniquement exécuter.

Un SM peut accueillir un maximum de 2048 threads actifs à tout moment (c'est 64 warps × 32 threads par warp). Si votre kernel utilise beaucoup de registres, vous avez moins d'espace pour les threads. Si votre kernel utilise beaucoup de mémoire partagée, vous avez moins d'espace pour les threads.

Voici un exemple concret. Supposons que votre kernel utilise 50 registres par thread. Chaque SM a 256 KB = 262 144 registres. Avec 50 registres par thread, un SM peut supporter 262 144 / 50 = 5 242 threads, soit 5 242 / 32 = 163 warps. Puisque un SM supporte un maximum de 64 warps, vous êtes limité à 64 warps × 32 threads = 2048 threads, ce qui donne une occupancy de 64 / 64 = 100%.

Mais si votre kernel utilise 200 registres par thread, alors un SM ne peut supporter que 262 144 / 200 = 1 310 threads, soit 1 310 / 32 = 40 warps. Cela donne une occupancy de 40 / 64 = 62,5%. Maintenant, quand un warp attend en mémoire (attendre un accès HBM), le SM peut immédiatement basculer vers un autre warp actif. Avec seulement 40 warps, il y a moins de warps en attente pour basculer, donc plus de temps le SM reste inactif.

La règle générale est qu'une occupancy d'au moins 50% est acceptable pour la plupart des kernels, et au-delà de 80%, les bénéfices sont marginaux. Le sweet spot est généralement autour de 75%.

### 2.5 Nombre concrets pour l'A100

Pour que vous ayez des chiffres concrets en tête :

- **6912 cœurs CUDA** : 108 SM × 64 cœurs par SM. Cela signifie que potentiellement 6912 opérations arithmétiques simples peuvent être effectuées chaque cycle d'horloge.

- **Fréquence horloge : 1.4 GHz (1.4 × 10^9 Hz)**. Un cycle prend 1 / 1.4 GHz ≈ 0.71 nanosecondes.

- **Peak FP32 performance (32-bit floats) : 6912 cœurs × 2 opérations par cycle (FMA = fused multiply-add) × 1.4 GHz = 19.3 TFLOP/s** (19,3 trillions de floating-point operations par seconde).

- **Peak FP16 performance : Avec Tensor Cores (que nous verrons au Chapitre 4), ~160 TFLOP/s.**

- **Bande passante HBM : 80 GB / (1 / 2 TB/s) ≈ 2 TB/s = 2000 GB/s**. Pour atteindre le peak de 19.3 TFLOP/s, vous devriez faire 19.3 TFLOP/s / (2000 GB/s) ≈ 9.65 FLOP/byte. C'est possible mais exigeant.

---

## Chapitre 3 : Le modèle de programmation CUDA — Grilles, blocs et threads {#ch3}

### 3.1 Concept de kernel CUDA

Un **kernel CUDA** est une fonction qui s'exécute sur le GPU. Contrairement aux fonctions CPU normales, un kernel s'exécute dans un contexte massivement parallèle : la même fonction est exécutée par des milliers de threads en parallèle sur des données différentes.

Conceptuellement, voici comment cela fonctionne :

```c
// Déclaration du kernel (noté __global__)
__global__ void add_kernel(float *a, float *b, float *c, int N) {
    // Chaque thread exécute ce code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Dans le code CPU main
int main() {
    // Allouer mémoire GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copier les données
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Lancer le kernel avec une grille de 32 blocs, 256 threads par bloc
    add_kernel<<<32, 256>>>(d_a, d_b, d_c, N);

    // Copier le résultat
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
}
```

L'important à comprendre est que `add_kernel` s'exécute des milliers de fois — une fois pour chaque thread. Chaque thread reçoit les mêmes arguments (`a`, `b`, `c`, `N`), mais des variables différentes comme `threadIdx.x` leur permettent de traiter différentes parties des données.

### 3.2 Hiérarchie Grid-Block-Thread

CUDA organise les threads en une hiérarchie à trois niveaux : **grille (grid)**, **blocs (blocks)**, et **threads**. Comprendre cette hiérarchie est essentiel pour programmer efficacement sur GPU.

**Grid (Grille) :** Une grille est la totalité de tous les threads lancés par un kernel. Si vous lancez `add_kernel<<<32, 256>>>`, vous lancez une grille contenant 32 × 256 = 8192 threads.

**Block (Bloc) :** Une grille est divisée en blocs. Un bloc est un groupe de threads qui partagent la mémoire partagée et peuvent être synchronisés ensemble. Dans `<<<32, 256>>>`, le premier nombre (32) est le nombre de blocs, et chaque bloc contient 256 threads.

**Thread :** Un thread est la plus petite unité d'exécution. C'est le code à l'intérieur de `add_kernel` qui s'exécute une fois pour chaque thread.

Voici un diagramme pour visualiser cela :

```
┌────────────────────────────────────────────────────────┐
│                  GRID (Grille)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Block 0 │  │  Block 1 │  │ Block 31 │  ...      │
│  ├──────────┤  ├──────────┤  ├──────────┤            │
│  │ Thread 0 │  │ Thread 0 │  │ Thread 0 │            │
│  │ Thread 1 │  │ Thread 1 │  │ Thread 1 │            │
│  │ ...      │  │ ...      │  │ ...      │            │
│  │ Thread   │  │ Thread   │  │ Thread   │            │
│  │ 255      │  │ 255      │  │ 255      │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                        │
│  Total : 32 blocks × 256 threads = 8192 threads      │
└────────────────────────────────────────────────────────┘
```

Les blocs sont importants car ils définissent le **scope** de la mémoire partagée et de la synchronisation. Tous les threads d'un bloc peuvent être synchronisés avec `__syncthreads()`, mais les threads de différents blocs ne peuvent pas se synchroniser directement.

### 3.3 Indices pour localiser un thread

Chaque thread a besoin de savoir qui il est. Dans notre exemple `add_kernel`, nous avions besoin de savoir quel élément du tableau traiter. CUDA fournit deux variables de construction : **blockIdx** et **threadIdx**.

**threadIdx** est un vecteur 3D contenant l'indice du thread dans son bloc. Sur A100, les blocs peuvent avoir un maximum de 1024 threads (par exemple, 32 × 32, ou 256, ou 512 × 2, etc.). Le threadIdx.x, threadIdx.y, threadIdx.z donnent les coordonnées du thread dans ce bloc.

**blockIdx** est un vecteur 3D contenant l'indice du bloc dans la grille. Cela vous permet de savoir quel bloc vous êtes.

**blockDim** est un vecteur 3D donnant les dimensions du bloc. Par exemple, si vous lancez `kernel<<<gridDim, (256, 1, 1)>>>`, alors blockDim.x = 256, blockDim.y = 1, blockDim.z = 1.

**gridDim** donne les dimensions de la grille. Si vous lancez `kernel<<<(32, 1, 1), 256>>>`, alors gridDim.x = 32, gridDim.y = 1, gridDim.z = 1.

Pour convertir ces indices 3D en un indice linéaire 1D (ce qu'on fait généralement), on utilise :

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Pour une grille et un bloc 1D
```

Pour une configuration 2D plus complexe :

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * num_cols + col;  // Accès ligne-majeur
```

### 3.4 Synchronisation avec __syncthreads()

Un problème commun en programmation parallèle est la **race condition** : deux threads écrivent ou lisent les mêmes données sans coordination, conduisant à des résultats imprévisibles.

CUDA fournit `__syncthreads()` pour synchroniser les threads d'un même bloc. Quand un thread rencontre `__syncthreads()`, il s'arrête et attend que tous les autres threads du bloc atteignent le même point.

Voici un exemple d'une **réduction somme** où nous devons faire attention à la synchronisation :

```c
__global__ void sum_reduction(float *input, float *output, int N) {
    // Allocation dynamique de shared memory (nous verrons cela au Chapitre 9)
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Chaque thread charge une valeur en shared memory
    if (idx < N) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();  // S'assurer que tous les loads sont terminés

    // Réduction en arbre
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // S'assurer que tous les additions sont terminées
    }

    // Écrire le résultat
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

Sans ces `__syncthreads()`, nous ne saurions pas si le thread 0 a lu sdata[1] avant que le thread 1 ait écrit sa valeur. Avec `__syncthreads()`, nous avons la garantie que tous les threads ont terminé leur travail avant de passer à l'étape suivante.

### 3.5 Mémoire partagée dynamique

Jusqu'à présent, nous avons supposé que la taille de la mémoire partagée était connue à la compilation. Mais souvent, elle dépend du lancement du kernel. CUDA permet la **mémoire partagée dynamique** en utilisant `extern __shared__`.

```c
__global__ void kernel_with_dynamic_smem(float *input, float *output) {
    extern __shared__ float sdata[];  // Déclaration sans taille

    int tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();

    // Utiliser sdata
    output[tid] = sdata[tid] * 2.0f;
}

// Lancement :
// La taille en bytes est passée comme 3e paramètre
kernel_with_dynamic_smem<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(d_input, d_output);
```

Le troisième paramètre dans `<<<gridDim, blockDim, smem_size>>>` est la taille en bytes de la mémoire partagée dynamique. Cela permet une flexibilité maximale.

### 3.6 Configuration de lancement : <<<grid, block, smem, stream>>>

Maintenant que nous comprenons les composants, voici la syntaxe complète de lancement d'un kernel :

```c
kernel_name<<<gridDim, blockDim, dynamic_smem_bytes, stream>>>(args);
```

**gridDim :** Dimensions de la grille. Peut être dim3(x) pour 1D ou dim3(x, y, z) pour 3D. Maximum de 2^31 - 1 dans chaque dimension.

**blockDim :** Dimensions du bloc. Maximum de 1024 threads par bloc total (donc 32 × 32 est OK, mais 1024 × 2 n'est pas possible). Peut être 1D, 2D ou 3D.

**dynamic_smem_bytes :** (Optional, par défaut 0) Bytes de mémoire partagée dynamique à allouer. On ne peut pas dépasser la limite du SM (192 KB pour A100).

**stream :** (Optional, par défaut le stream 0) Les streams permettent le recouvrement d'exécution. Plusieurs kernels peuvent s'exécuter en parallèle s'ils sont sur des streams différents.

Exemple complet :

```c
dim3 gridDim(32, 16);    // 32 × 16 blocs = 512 blocs
dim3 blockDim(16, 16);   // 16 × 16 threads = 256 threads par bloc
size_t smem_size = 16 * 16 * sizeof(float);  // 256 floats = 1024 bytes

kernel_2d<<<gridDim, blockDim, smem_size>>>(d_data);
```

Cette configuration crée une grille de 512 blocs contenant 256 threads chacun, soit 131 072 threads au total.

### 3.7 Types de mémoire et leur durée de vie

CUDA expose plusieurs types de mémoire avec des sémantiques différentes :

| Type | Scope | Durée de vie | Accès | Notes |
|---|---|---|---|---|
| **Registres** | 1 thread | Durée du kernel | 1 thread seulement | Rapide mais limité |
| **Shared Memory** | 1 bloc | Durée du kernel | Tous les threads du bloc | Synchronisation requise |
| **Global Memory (HBM)** | GPU entier | Jusqu'à libération | N'importe quel thread | Lent mais énorme |
| **Constant Memory** | GPU entier | Jusqu'à libération | Lecture seulement | Cached agressivement |
| **Local Memory** | 1 thread | Durée du kernel | 1 thread seulement | Déborde en HBM si trop grand |

La plupart du temps, vous travaillez avec la **global memory** (HBM) et la **shared memory**. Les registres sont gérés automatiquement par le compilateur.

---

## Chapitre 4 : Précision numérique et formats virgule flottante {#ch4}

### 4.1 Standard IEEE 754

Pour comprendre les optimisations de précision dans Flash Attention 2, nous avons besoin de bien comprendre comment les nombres à virgule flottante fonctionnent en informatique.

Le standard **IEEE 754** définit comment représenter les nombres à virgule flottante. Un nombre à virgule flottante est représenté par trois composants : le **signe**, l'**exposant**, et la **mantisse** (ou **fraction**).

Pour un nombre à virgule flottante 32-bit (FP32) :
- **1 bit** pour le signe (0 pour positif, 1 pour négatif)
- **8 bits** pour l'exposant (valeurs de -126 à 127)
- **23 bits** pour la mantisse (la partie fractionnaire)

La valeur réelle est calculée comme : (-1)^signe × 1.mantisse × 2^(exposant - 127)

Par exemple, le nombre 5.5 en FP32 :
- 5.5 = 101.1 en binaire
- = 1.011 × 2^2 en notation scientifique normalisée
- Signe = 0 (positif)
- Exposant = 2 + 127 = 129 = 10000001 en binaire
- Mantisse = 011 (les bits après le point, avec des zéros à droite)

Cela donne une représentation de 32 bits.

### 4.2 FP32 vs FP16 vs BF16

La précision full est FP32 (32 bits). Pour l'entraînement de réseaux de neurones profonds sur GPU, une précision réduite suffit souvent.

**FP32 (Float32)** : La précision complète.
- Exposant : 8 bits (range énorme : 10^-38 à 10^38)
- Mantisse : 23 bits (environ 7 chiffres décimaux de précision)
- Utilisé pour : Accumulation, certains calculs critiques

**FP16 (Half precision)** : Réduit de moitié.
- Exposant : 5 bits (range : 10^-4.5 à 10^4.5)
- Mantisse : 10 bits (environ 3 chiffres décimaux de précision)
- Utilisé pour : Calculs de forward pass, économie de bande passante
- Problème : Peut déborder/underflow facilement (par exemple, multiplier deux petits nombres FP16 peut donner 0 par underflow)

**BF16 (Brain Float16)** : Une variante de FP16 développée par Google.
- Exposant : 8 bits (même range que FP32 ! Range : 10^-38 à 10^38)
- Mantisse : 7 bits (moins de précision que FP16, seulement ~2 chiffres décimaux)
- Utilisé pour : Calculs de forward pass avec meilleure stabilité numérique que FP16
- Avantage : Même exposant que FP32, donc moins de risques de débordement/underflow

| Paramètre | FP32 | FP16 | BF16 |
|---|---|---|---|
| **Bits totaux** | 32 | 16 | 16 |
| **Exposant** | 8 bits | 5 bits | 8 bits |
| **Mantisse** | 23 bits | 10 bits | 7 bits |
| **Précision décimale** | ~7 chiffres | ~4 chiffres | ~2 chiffres |
| **Range** | 10^-38 à 10^38 | 10^-4.5 à 10^4.5 | 10^-38 à 10^38 |
| **Bande passante** | 1× (baseline) | 2× | 2× |
| **Stabilité numérique** | Excellente | Risquée | Bonne |

Pour le deep learning, l'intuition clé est que la plupart des calculs ne nécessitent pas 7 chiffres de précision décimale. Les gradients descendent plus ou moins régulièrement en toute façon. BF16 offre le meilleur des deux mondes : la stabilité de FP32 (range large) avec la bande passante de FP16.

### 4.3 CUDA half type et conversions

En CUDA, les half-precision floats sont représentés par le type `__half`. Vous pouvez les utiliser, mais vous devez être prudent.

```c
#include <cuda_fp16.h>

__global__ void half_example(float *input, __half *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Conversion implicite de float à __half
        __half h = __float2half(input[idx]);

        // Arithmétique FP16
        __half h2 = h + h;  // Addition FP16

        // Conversion retour
        output[idx] = h2;
    }
}
```

Les conversions entre FP32 et FP16 sont relativement bon marché (quelques cycles) mais pas gratuites. C'est pourquoi les GPU modernes possèdent des **Tensor Cores** — des unités spécialisées qui peuvent faire des multiplications FP16 ou BF16 très rapidement.

### 4.4 Tensor Cores et opérations masquées (WMMA)

Un **Tensor Core** sur un GPU A100 est une unité spécialisée capable de faire une **opération matrice-matrice** en quelques cycles. Spécifiquement, un Tensor Core sur A100 peut faire une opération 16×16×16 : multiplier une matrice FP16 16×16 par une matrice FP16 16×16, accumuler le résultat en FP32, en quelques cycles.

Cela semble spécifique, mais c'est exactement ce dont vous avez besoin pour les opérations de deep learning courantes. Une **multiplication matrice-matrice (GEMM, General Matrix Multiply)** peut être décomposée en milliers de ces opérations 16×16×16 exécutées en parallèle.

Pour accéder aux Tensor Cores depuis CUDA, vous utilisez les **WMMA APIs (Warp Matrix Multiply-Accumulate)** :

```c
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensor_core_example(half *a, half *b, float *c) {
    // Déclaration de fragments de matrice
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialiser l'accumulateur
    fill_fragment(c_frag, 0.0f);

    // Charger les fragments (depuis global memory)
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);

    // Multiplication FP16 × FP16 → FP32 accumulation
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Stocker le résultat
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}
```

Les Tensor Cores peuvent faire environ 10× plus d'opérations par seconde que les cœurs CUDA standards si vous utilisez FP16 ou BF16. C'est la raison principale pour laquelle le deep learning s'est tourné vers des précisions réduites.

### 4.5 Pourquoi le deep learning tolère FP16

C'est une question légitime : pourquoi pouvez-vous réduire la précision de 32 bits à 16 bits et obtenir toujours un modèle qui fonctionne ?

La raison est que l'entraînement de réseaux de neurones est **intrinsèquement bruyant et robuste**. Chaque gradient est une somme de contributions de milliers d'exemples d'entraînement, donc une petite erreur numérique dans un gradient particulier ne change pas beaucoup le grand tableau. De plus, les mises à jour d'optimiseur comme l'Adam contiennent déjà une certaine quantité de bruit randomisé.

En pratique, on entraîne souvent avec des poids d'accumulation (accumulateurs) en FP32 et des poids stockés en FP16 :

```c
// FP16 forward pass (rapide grâce aux Tensor Cores)
__half *weights_fp16;     // Stocké en FP16
__half activation = matmul_fp16(input, weights_fp16);

// FP32 accumulation pour les gradients (stable)
float weight_grad = 0.0f;  // FP32 accumulation
for (int i = 0; i < batch_size; i++) {
    float grad_contrib = gradient_fp16[i];  // Charger depuis FP16
    weight_grad += grad_contrib;  // Accumuler en FP32
}

// Mise à jour FP32
float new_weight = (float)weights_fp16 - learning_rate * weight_grad;
weights_fp16 = __float2half(new_weight);  // Convertir retour en FP16
```

Cette approche mixte (calcul en FP16, accumulation en FP32) offre la meilleure stabilité numérique.

---

## Chapitre 5 : Le mécanisme d'attention — Le cœur des transformers {#ch5}

### 5.1 Contexte historique : Transformers (Vaswani et al., 2017)

En 2017, Ashish Vaswani et ses collègues ont publié l'article révolutionnaire **"Attention Is All You Need"** qui introduisait l'architecture Transformer. Pour la première fois, une architecture sans convolutions ni récurrence (LSTM) pouvait égaler les architectures précédentes sur les tâches de traitement du langage naturel.

L'idée centrale du Transformer est le **mécanisme d'attention** : plutôt que de forcer le modèle à compresser toute l'information contextuelle en un vecteur de taille fixe (comme les RNN), l'attention permet au modèle de regarder sélectivement les positions passées et futures pertinentes.

Pensez-y comme ceci : si vous lisez une phrase, vous ne traitez pas chaque mot de manière isolée. Vous établissez des connexions entre les mots. Quand vous lisez le mot "pronoms" à la fin d'une phrase, vous vous souvenez du sujet mentionné au début. C'est ce que fait l'attention — elle établit des connexions à long terme entre les mots.

### 5.2 Décomposition de l'attention : Q, K, V

Le mécanisme d'attention utilise trois projections de l'entrée : **Q (Query)**, **K (Key)**, et **V (Value)**. Voici l'intuition :

**Query (Q) :** "Qui demande des informations ?" Si nous traitons le mot "pronoms", sa Query demande : "Quels mots dans la phrase sont pertinents pour "pronoms" ?"

**Key (K) :** "Qui suis-je pour la recherche ?" Chaque mot produit une Key. La Key du mot "chat" dit essentiellement : "Je suis un mot pour un animal."

**Value (V) :** "Quelle information dois-je donner ?" La Value contient la représentation réelle de ce mot qu'il faut transmettre.

L'analogie avec une base de données est utile : imaginez que vous avez une table avec des colonnes Key et Value. Pour récupérer des informations, vous envoyez une Query. La base de données la compare à toutes les Keys, trouve les plus pertinentes (celles avec le score de similarité le plus élevé), et retourne les Values correspondantes.

Mathématiquement, si l'entrée est une séquence de vecteurs x_1, x_2, ..., x_N, on projette :
- **Q = X × W_Q** (requêtes)
- **K = X × W_K** (clés)
- **V = X × W_V** (valeurs)

Où W_Q, W_K, W_V sont des matrices de poids apprenables.

### 5.3 Score, softmax et output

L'attention calcule un **score de similarité** entre chaque Query et chaque Key :

**Score = Q × K^T**

Ce score est une matrice N×N où l'élément (i,j) donne la similarité entre la i-ème position et la j-ème position. Si Q a les dimensions [batch, seq_len, head_dim], K a les dimensions [batch, seq_len, head_dim], alors Score = Q @ K.transpose() a les dimensions [batch, seq_len, seq_len].

Pour convertir ces scores en probabilités (poids), on utilise la **softmax** :

**P = softmax(Score)**

La softmax convertit un vecteur de scores en un vecteur de probabilités qui somment à 1 : P[i,j] = exp(Score[i,j]) / sum_j(exp(Score[i,j]))

Enfin, on multiplie ces poids par les Values :

**Output = P × V**

Si V a les dimensions [batch, seq_len, head_dim], Output a les mêmes dimensions.

### 5.4 Facteur d'échelle 1/√d

En pratique, les scores peuvent devenir très grands (imagine 64 ou 128 dimensions), ce qui rend la softmax instable (les gradients deviennent minuscules). Pour stabiliser, on divise par √d où d est la dimension des heads :

**Score = (Q × K^T) / √d**

Pourquoi √d ? L'intuition est que si chaque élément de Q et K est de moyenne 0 et variance 1, alors un produit scalaire de d dimensions a une variance d (car on somme d variables de variance 1). Diviser par √d normalise la variance, donc la softmax reçoit des entrées "raisonnables".

### 5.5 Multi-Head Attention

Un seul mécanisme d'attention peut sembler limitant. Que faire si vous voulez qu'un mot prête attention à différents aspects ? Par exemple, pour "le chat", vous pourriez vouloir qu'une head prête attention à "Le" (l'article) et une autre à "chat" (le sujet principal).

La **Multi-Head Attention** divise les head_dim en plusieurs têtes :

```
Input X (shape: [batch, seq_len, d_model])
  ↓ Split into h heads
  ├─→ Head 0 : Q_0 = X @ W_Q_0 → Attention → Output_0
  ├─→ Head 1 : Q_1 = X @ W_Q_1 → Attention → Output_1
  ...
  └─→ Head h-1 : Q_{h-1} = X @ W_Q_{h-1} → Attention → Output_{h-1}
  ↓ Concatenate
Concatenate([Output_0, Output_1, ..., Output_{h-1}])
  ↓ Project
Final Output = Concat @ W_O
```

Pour un modèle typique avec d_model = 512 et 8 heads, chaque head traite une dimension d'attention de 512 / 8 = 64.

Le calcul pour une head unique avec des dimensions [B, H, N, D] où :
- B = batch size
- H = nombre de heads
- N = longueur de la séquence
- D = dimension par head

L'opération entière est : Attention(Q, K, V) = softmax(Q K^T / √D) V

où Q, K, V ont tous les dimensions [B, H, N, D].

### 5.6 Masquage causal (pour la génération)

Dans la génération autoréagressive (comme GPT), un token ne peut pas prêter attention aux tokens futurs (puisqu'ils n'ont pas encore été générés à l'inférence). Pour empêcher cela, on utilise le **masquage causal**.

Avant softmax, on remplace les positions (i, j) où i < j (token i regardant un token futur j) par -∞ :

```
Score = (Q K^T) / √D
Score[i, j] = -∞ for all i < j  // Masquer les tokens futurs
P = softmax(Score)  // softmax(-∞) = 0
```

Après softmax, P[i, j] = 0 pour les positions futures, ce qui signifie que le token i reçoit une attention de 0 du token j à l'avenir.

### 5.7 Layouts mémoire et complexité

Stockage sur GPU : un tenseur d'attention complète [B, H, N, D] où B=32, H=12, N=4096, D=64 :
- Nombre d'éléments : 32 × 12 × 4096 × 64 = 100,663,296 éléments
- Taille en FP32 : ~402 MB
- Taille en FP16 : ~201 MB

Pour un seul Transformer forward pass, la bande passante déplacée est énorme. Une attention standard déplace :
- **Lecture :** Q, K, V de global memory
- **Écriture :** Output vers global memory
- **Opérations :** Lire et écrire la matrice Score N×N complète en mémoire intermédiaire

Pour N=4096 et d'autres paramètres ci-dessus, la matrice Score N×N seule fait [32, 12, 4096, 4096] = 6,442,450,944 éléments ≈ 25 GB en FP32. C'est énorme et c'est l'un des goulots d'étranglement majeurs du Transformer.

---

## Chapitre 6 : Softmax — Stabilité numérique et algorithmes online {#ch6}

### 6.1 Le problème d'overflow naïf

La softmax semble simple en théorie :

P[i] = exp(Score[i]) / sum_j(exp(Score[j]))

Mais en pratique, c'est piégé numériquement. Considérez un Score avec la valeur max de 1000. Alors exp(1000) ≈ 10^434, ce qui dépasse largement le range de FP32 (max ≈ 10^38). Vous obtenez inf (overflow).

Même avec des valeurs plus petites, disons Score max = 100 : exp(100) ≈ 2.7 × 10^43, ce qui dépasse FP32. Overflow.

Ce qui est plus traître, c'est que si vous avez un Score avec des valeurs négatives comme [-5, -4, 1000], alors :
- exp(-5) ≈ 0 (underflow en FP32)
- exp(-4) ≈ 0 (underflow)
- exp(1000) ≈ inf (overflow)

Vous obtenez P = [0, 0, 1] (techniquement NaN à cause de inf / inf), ce qui est faux. Le vrai softmax devrait donner [0, 0, 1] mais par une chance, pas par du débordement numérique.

### 6.2 Le max-shift trick

La solution classique est le **max-shift trick** (ou **log-sum-exp trick**). On réécrit :

P[i] = exp(Score[i]) / sum_j(exp(Score[j]))
     = exp(Score[i] - max(Score)) / sum_j(exp(Score[j] - max(Score)))

En soustrayant le maximum avant d'exponentier, on s'assure que tous les exponents sont ≤ 0, donc tous les exponentielles sont ≤ 1, donc pas de overflow. Et puisque le max devient 0 après soustraction, exp(0) = 1 ne peut pas underflow.

Exemple concret :
- Score = [-5, -4, 1000]
- max(Score) = 1000
- Score - max = [-1005, -1004, 0]
- exp(Score - max) ≈ [0, 0, 1] (les deux premiers underflowent à 0 in practice, ce qui est correct pour softmax)
- sum = 1 + ε (où ε est minuscule)
- P ≈ [0, 0, 1] ✓ (correct, sans débordement intermédiaire)

### 6.3 Softmax online et statistiques courantes (running stats)

Pour Flash Attention, on a besoin de calculer le softmax sur des blocs de N éléments. Si N = 4096, on ne peut pas charger tous les 4096 éléments en mémoire partagée à la fois. Flash Attention utilise une approche appelée **softmax online** ou **softmax incrémental** développée par Milakov & Gimelshein (2018).

L'idée est de maintenir deux statistiques à mesure que vous lisez les éléments :
- **m :** Le maximum courant
- **l :** La somme courante des exponentielles décalées

À chaque nouvel élément x :
1. Si x > m_old, mettre à jour m := x
2. Recalculer l en tenant compte du nouveau m (les anciens exponentiels doivent être décalés pour correspondre au nouveau m)

Voici l'algorithme en pseudocode annoté :

```python
# Initialisation
m = -inf          # Running max
d = 0.0           # Running sum of exp (d pour "denominator")
result = []       # Accumule les probas

# Traiter les éléments du score un par un (ou par bloc)
for x in Score:
    m_prev = m
    m = max(m, x)

    # Rescaler l'ancien d pour correspondre au nouveau m
    # Avant : d = sum_i(exp(score_i - m_prev))
    # Après : d = sum_i(exp(score_i - m_new))
    # Différence : exp(score_i - m_new) = exp(score_i - m_prev) * exp(m_prev - m_new)
    d = d * exp(m_prev - m) + exp(x - m)

    result.append(exp(x - m) / d)  # Approximation courante de softmax

# Résultat final après traiter tous les scores
P = result  # Les valeurs finales

# Mais il y a un problème ! À la fin, d = sum_i(exp(score_i - m))
# Alors le vrai softmax est P[i] = exp(score_i - m) / sum_j(exp(score_j - m)) = result[i] * d / d = result[i]
# En réalité, ce n'est pas correct tant que le dernier calcul de d est correct.
```

L'algorithme exact est un peu plus délicat. On stocke :
- **m_block :** Le maximum d'un bloc
- **l_block :** La somme des exponentielles du bloc avec le max décalé : sum_i(exp(score_i - m_block))

Quand on lit un nouveau bloc avec max m_new_block, on rescale le bloc précédent :

```python
alpha = exp(m_block_old - m_block_new)  # Facteur de rescaling
l_new = alpha * l_block_old + l_block_new  # Nouvelle somme cumulée

# Les probabilités précédentes doivent être rescalées :
P_old = P_old * alpha  # Cela s'appelle "rescale des poids précédents"
```

### 6.4 Problème clé : accumulation stable pour Softmax

Pour Flash Attention 2, la clé est que **on ne stocke jamais la matrice Score N×N complet**. Au lieu de cela :

1. On traite Score par blocs (carrelage, ou "tiling")
2. Pour chaque bloc, on calcule le softmax
3. On accumule les résultats finaux

Mais il y a une subtilité : si vous calculez softmax par bloc indépendamment, vous n'obtenez pas le vrai softmax global. Flash Attention gère cela en retenant :
- **L (logsumexp) :** log(sum_j(exp(score_j - m))) — stocké après traiter chaque bloc pour comparaison future
- **m (max) :** Le maximum global jusqu'à présent
- **Facteur alpha :** exp(m_ancien - m_nouveau) pour rescaler les anciens résultats

Voici le pseudocode pour l'accumulation dans Flash Attention 2 :

```python
# Initialisation
m = -inf
l = 0.0
O = zeros([batch, heads, seq_len, d_head])
L = zeros([batch, heads, seq_len])

# Traiter les blocs de K, V (blocs colonne dans la matrice Score)
for block_j in range(num_blocks_kv):
    # Charger K_block et V_block
    K_block = K[:, :, block_j * block_size : (block_j+1) * block_size, :]  # [B, H, block_size_k, D]
    V_block = V[:, :, block_j * block_size : (block_j+1) * block_size, :]  # [B, H, block_size_k, D]

    # Score de ce bloc : Q @ K_block^T
    S_block = Q @ K_block.transpose(-1, -2)  # [B, H, seq_len, block_size_k]
    S_block = S_block / sqrt(d_head)

    # Appliquer le masquage causal si nécessaire
    if causal:
        S_block = apply_causal_mask(S_block)

    # Trouver le max de ce bloc
    m_block = max(S_block, dim=-1, keepdim=True)  # [B, H, seq_len, 1]

    # Probabilités brutes : exp(S - m_block)
    P_block = exp(S_block - m_block)  # [B, H, seq_len, block_size_k]

    # Somme des exponentielles du bloc
    l_block = sum(P_block, dim=-1, keepdim=True)  # [B, H, seq_len, 1]

    # Rescaler les résultats précédents pour correspondre au nouveau max global
    # l est la somme jusqu'à présent (pour les anciens blocs)
    # m est le max jusqu'à présent
    alpha = exp(m - m_block)  # Facteur de rescaling : peut être > 1 ou < 1
    O = O * alpha.unsqueeze(-1)  # Rescaler l'accumulateur de sortie
    l = l * alpha + l_block  # Rescaler et ajouter la nouvelle somme
    m = m_block  # Mettre à jour le max global

    # Accumuler les résultats : O += P_block @ V_block
    O = O + P_block @ V_block  # [B, H, seq_len, D]

    # Sauvegarder les statistiques
    L[:, :, block_j] = l  # Pour utiliser en backward

# Normalisation finale
Output = O / l.unsqueeze(-1)  # Diviser par les sommes des exponentielles
```

L'étape clé est la **rescaling** : quand le max global augmente, tous les poids précédents doivent être divisés par exp(m_ancien - m_nouveau), ce qui rescale toutes les probabilités correctement.

### 6.5 Pourquoi c'est important pour Flash Attention 2

Flash Attention 2 utilise cette approche online de softmax car elle permet :
1. **Pas de stockage N×N :** On n'a jamais besoin de stocker la matrice Score complète
2. **Calculer par blocs :** Chaque bloc peut être indépendamment traité
3. **Calculer avec stabilité numérique :** Le max-shift trick est appliqué automatiquement

Sans cette technique, vous seriez forcé de :
- Charger la matrice Score N×N complète en mémoire (trop grande)
- Calculer le softmax globalement (coûteux en bande passante)

Avec elle, Flash Attention divise le problème en blocs traitable qui tient en mémoire partagée et arrive toujours au même résultat mathématiquement correct.

---

## Chapitre 7 : Le Roofline Model — Prédire les performances {#ch7}

### 7.1 Intensité arithmétique

Un concept clé en programmation GPU est l'**intensité arithmétique**, souvent notée I ou AIR (Arithmetic Intensity Ratio).

L'intensité arithmétique est définie comme :

**AIR = Opérations arithmétiques / Bytes transférés en mémoire**

Mesurée en FLOP/byte ou opérations par byte. Par exemple, si vous faites 1000 FLOP (floating point operations) pour transférer 100 bytes de données, AIR = 1000 / 100 = 10 FLOP/byte.

Pourquoi c'est important ? Parce que cela détermine si votre calcul est **limitée par le calcul (compute-bound)** ou **limitée par la mémoire (memory-bound)**.

### 7.2 Roofline Diagram

Le **Roofline Model** est une visualisation développée par Samuel Williams et al. (2009) qui prédit les performances théoriques maximum d'un kernel en fonction de son intensité arithmétique.

Voici comment construire un diagramme Roofline pour un GPU A100 :

**Axe X :** Intensité arithmétique (FLOP/byte), échelle logarithmique
**Axe Y :** Performance (GFLOP/s), échelle logarithmique

On trace deux lignes :

1. **Ligne de mémoire (Memory Bound Line) :** Performance = Bande passante mémoire × Intensité arithmétique
   - Pour A100 : Performance (GFLOP/s) = 2000 GB/s × Intensité (FLOP/byte)
   - Pente = 2000
   - Exemple : À AIR = 10 FLOP/byte, Performance = 2000 × 10 = 20,000 GFLOP/s = 20 TFLOP/s

2. **Ligne de calcul (Compute Ceiling) :** Performance = Peak FP32 FLOPS
   - Pour A100 : Performance = 19.3 TFLOP/s (constant horizontal)

Ensemble, ces deux lignes forment une forme de toit. À faible intensité arithmétique, vous êtes limité par la mémoire. À haute intensité arithmétique, vous êtes limité par le calcul.

```
Performance (GFLOP/s)
│
│         ╱╱╱╱╱╱╱╱╱╱╱ Compute Ceiling = 19.3 TFLOP/s
│        ╱╱╱╱╱╱╱╱╱╱╱
│       ╱╱╱╱╱╱╱╱╱╱╱
│      ╱╱╱╱╱╱╱╱╱╱╱
│     ╱╱╱╱╱╱╱╱╱╱╱ (Point de croisement)
│    ╱╱╱╱╱╱╱╱╱╱╱
│   ╱ Memory Bound ╱ (pente = 2000 GB/s)
│  ╱╱╱╱╱╱╱╱╱╱╱
│ ╱╱╱╱╱╱╱╱╱╱╱
└────────────────────→ Intensité Arithmétique (FLOP/byte)
  0.1   1    10   100
```

Le point de croisement se trouve à : 19.3 TFLOP/s = 2000 GB/s × AIR
AIR = 19.3 TFLOP/s / 2000 GB/s ≈ 9.65 FLOP/byte (pour FP32)

### 7.3 Calcul d'intensité arithmétique pour l'attention standard

Analysons l'intensité arithmétique de l'attention standard (sans Flash Attention).

**Opérations :**
- Score = Q @ K^T : (N × D) @ (D × N) = N^2 × D opérations = N^2 × D × 2 FLOPs (compte à la fois mult et add)
- Softmax : N^2 opérations (mais peu comparé au matmul)
- Output = P @ V : (N × N) @ (N × D) = N^2 × D × 2 FLOPs

Total : ~4 N^2 D FLOPs

**Transferts mémoire :**
- Lecture Q : (N × D) = N × D
- Lecture K : (N × D) = N × D
- Lecture V : (N × D) = N × D
- Écriture Output : (N × D) = N × D
- Matrice Score intermédiaire N×N : Dépend de comment vous l'implémentez

En pratique, on doit stocker la matrice Score N×N en mémoire. Pour N = 4096, D = 64 :
- Octets transférés : 4 × (N × D) + N^2 = 4 × (4096 × 64) + 4096^2 = 1,048,576 + 16,777,216 ≈ 17.8 MB

Non, attendez. Écrivons cela plus soigneusement pour les chiffres réels.

Pour un attention layer avec B = 32, H = 12, N = 4096, D = 64 :

**Opérations (FLOPs) :**
- Q @ K^T : B × H × N × D × N × 2 = 32 × 12 × 4096 × 64 × 4096 × 2 ≈ 101 TFLOP (Tera FLOP)
- Softmax : B × H × N × N ≈ 2 TFLOP (marginal)
- P @ V : B × H × N × N × D × 2 ≈ 101 TFLOP
- Total : ~204 TFLOP

**Mémoire (Bytes) :**
- Lecture Q : 32 × 12 × 4096 × 64 × 4 = 402 MB
- Lecture K : 32 × 12 × 4096 × 64 × 4 = 402 MB
- Lecture V : 32 × 12 × 4096 × 64 × 4 = 402 MB
- Écriture Output : 32 × 12 × 4096 × 64 × 4 = 402 MB
- Matrice Score N×N : 32 × 12 × 4096 × 4096 × 4 ≈ 25 GB (ÉNORME !)
- Total : ~26.8 GB

**Intensité arithmétique :**
AIR = 204 TFLOP / 26.8 GB = 204 × 10^12 FLOP / 26.8 × 10^9 bytes ≈ 7600 FLOP/byte

Cela semble énorme ! Mais attendez — vous ne pouvez pas faire 204 TFLOP en une fois sans mémoire cache. Pour une seule GPU A100 qui peut faire 19.3 TFLOP/s, cela prend 204 / 19.3 ≈ 10.5 secondes. C'est parce que la matrice Score N×N doit être chargée en mémoire intermédiaire (global ou partagée), ce qui devient le goulot d'étranglement.

### 7.4 Pourquoi l'attention standard est memory-bound

Le problème principal est la **matrice Score N×N**. Pour N = 4096, c'est une matrice de 16 millions d'éléments. Cela ne tient pas en mémoire partagée (192 KB maximum), donc vous devez l'accumuler dans global memory, qui est lente.

Chaque actruer le transfert de cette matrice coûte un round-trip :
1. Écriture de Score en global memory après Q @ K^T
2. Lecture de Score depuis global memory pour appliquer softmax
3. Relecture de Score depuis global memory pour P @ V

Le Roofline Model pour l'attention standard :

```
Performance (GFLOP/s)
│
│         ┌─────────── Compute Ceiling ≈ 19.3 TFLOP/s
│         │
│         │  Standard Attention
│         │  AIR ≈ 0.001 FLOP/byte
│         │  Performance ≈ 2000 * 0.001 = 2 GFLOP/s (memory-bound)
│         │
│   ╱─────┘ (Point de croisement à AIR ≈ 9.65)
│  ╱
│ ╱ Memory Bound (pente = 2000)
│╱
└──────────────────────────
```

On voit que l'attention standard est **loin** du plafond de calcul. Elle est 10× ralentie par la mémoire.

### 7.5 Flash Attention et amélioration de l'intensité arithmétique

Flash Attention améliore drastiquement l'intensité arithmétique en **éliminant la matrice Score N×N complète du global memory**.

Au lieu de cela :
1. Traiter Q, K, V par **blocs de taille modest** (par exemple, 128 ou 256)
2. Calculer le score pour ce bloc **partiellement** en registres et shared memory
3. Appliquer softmax **pendant le calcul** (online softmax)
4. Accumuler immédiatement dans Output **sans stocker le score intermédiaire**

Cela réduit drastiquement les transferts mémoire. Les estimations empiriques pour Flash Attention :

**Transferts mémoire réduits :**
- Pas d'écriture/relecture de Score
- Lecture Q une fois : ~402 MB
- Lecture K une fois (mais par blocs) : ~402 MB
- Lecture V une fois (mais par blocs) : ~402 MB
- Écriture Output une fois : ~402 MB
- Total : ~1.6 GB (vs 26.8 GB avant !)

**Même FLOPs :** ~204 TFLOP (les opérations are les mêmes, juste réorganisées)

**Nouvelle intensité arithmétique :**
AIR = 204 TFLOP / 1.6 GB ≈ 127,500 FLOP/byte

Cela dépasse le crossover point de 9.65 FLOP/byte, donc vous êtes maintenant **compute-bound** et pouvez obtenir la performance du plafond (19.3 TFLOP/s).

Cela représente une **amélioration de ~10×** en performance pour les boucles d'attention à long contexte.

---

## Chapitre 8 : Flash Attention — L'algorithme révolutionnaire {#ch8}

### 8.1 Le problème N×N et la motivation

Avant Flash Attention (2022), chaque implémentation d'attention réalisait essentiellement ceci :

```python
def attention_naive(Q, K, V, scaling_factor=1.0):
    # Q, K, V sont [batch, heads, seq_len, d_head]

    # Matrice Score complète [batch, heads, seq_len, seq_len]
    scores = Q @ K.transpose(-1, -2) * scaling_factor

    # Softmax global [batch, heads, seq_len, seq_len]
    probs = softmax(scores, dim=-1)

    # Output [batch, heads, seq_len, d_head]
    output = probs @ V

    return output
```

Pour une séquence long N=4096, la matrice Score et probs occupent énormément de mémoire. Cela crée un goulot d'étranglement de bande passante car :
1. Vous écrivez Score en mémoire global
2. Vous le lisez à nouveau pour softmax
3. Vous le lisez une troisième fois pour multiplier par V

### 8.2 L'idée clé : Carrelage (Tiling) + Online Softmax

Flash Attention, introduit par Tri Dao et al. (2022), utilise deux idées brillantes :

1. **Carrelage (tiling) :** Diviser Q, K, V en blocs et traiter bloc par bloc, de sorte que le calcul intermédiaire tient en mémoire rapide (shared memory).

2. **Online Softmax :** Appliquer le softmax progressivement au fur et à mesure, sans jamais stocker la matrice Score complète.

Ces deux idées ensemble éliminent le transfert mémoire de Score et réduisent de façon drastique les transferts K, V grâce à la réutilisation locale.

### 8.3 Forward pass : Pseudocode annoté de Flash Attention 2

Voici le pseudocode complet avec annotations :

```python
def flash_attention_forward(
    Q, K, V,
    scaling_factor,
    block_size_q=128,
    block_size_kv=128,
    causal=False
):
    """
    Flash Attention Forward Pass

    Args:
        Q: [batch, heads, seq_len_q, d_head]
        K: [batch, heads, seq_len_kv, d_head]
        V: [batch, heads, seq_len_kv, d_head]
        scaling_factor: 1.0 / sqrt(d_head)
        block_size_q: Taille de bloc pour Q (nombre de tokens Q)
        block_size_kv: Taille de bloc pour K, V (nombre de tokens K)
        causal: Si True, appliquer le masquage causal (masquer tokens futurs)

    Returns:
        output: [batch, heads, seq_len_q, d_head]
        L: [batch, heads, seq_len_q] (logsumexp, stocké pour backward)
    """

    B, H, N_q, D = Q.shape
    N_kv = K.shape[2]

    # Nombre de blocs
    num_blocks_q = ceil(N_q / block_size_q)
    num_blocks_kv = ceil(N_kv / block_size_kv)

    # Initialiser les buffers de sortie
    output = zeros_like(Q)  # [B, H, N_q, D]
    L = zeros([B, H, N_q])  # [B, H, N_q] - logsumexp pour chaque position
    m = full([B, H, N_q], float('-inf'))  # [B, H, N_q] - max courant

    # === BOUCLE PRINCIPALE : Itérer sur les blocs de K, V ===
    # (On regarde tous les tokens K, V pour chaque bloc Q)
    for kv_block_idx in range(num_blocks_kv):
        # Bornes du bloc K, V
        kv_start = kv_block_idx * block_size_kv
        kv_end = min((kv_block_idx + 1) * block_size_kv, N_kv)
        kv_block_len = kv_end - kv_start

        # Charger le bloc K, V
        K_block = K[:, :, kv_start:kv_end, :]  # [B, H, block_size_kv, D]
        V_block = V[:, :, kv_start:kv_end, :]  # [B, H, block_size_kv, D]

        # === SOUS-BOUCLE : Itérer sur les blocs de Q ===
        # (Pour chaque bloc Q, calculer son attention avec le bloc K, V courant)
        for q_block_idx in range(num_blocks_q):
            # Bornes du bloc Q
            q_start = q_block_idx * block_size_q
            q_end = min((q_block_idx + 1) * block_size_q, N_q)
            q_block_len = q_end - q_start

            # Charger le bloc Q
            Q_block = Q[:, :, q_start:q_end, :]  # [B, H, block_size_q, D]

            # --- Calculs locaux pour ce bloc ---

            # 1. Calculer les scores : Q_block @ K_block^T
            #    [B, H, block_size_q, D] @ [B, H, D, block_size_kv]
            #    = [B, H, block_size_q, block_size_kv]
            S_block = torch.matmul(
                Q_block,
                K_block.transpose(-2, -1)
            ) * scaling_factor

            # 2. Appliquer le masquage causal si nécessaire
            if causal:
                # Les positions Q ne peuvent pas prêter attention aux positions K ultérieures
                # Créer une matrice de masque
                mask = torch.ones(q_block_len, kv_block_len, device=S_block.device)
                for i in range(q_block_len):
                    for j in range(kv_block_len):
                        # Position globale
                        q_global = q_start + i
                        kv_global = kv_start + j
                        if q_global < kv_global:
                            mask[i, j] = 0

                # Appliquer le masque (les positions masquées deviennent -inf)
                S_block = torch.where(
                    mask.bool().unsqueeze(0).unsqueeze(0),
                    S_block,
                    torch.tensor(float('-inf'), device=S_block.device)
                )

            # 3. Trouver le max local de ce bloc
            m_block = torch.max(S_block, dim=-1, keepdim=True)[0]  # [B, H, block_size_q, 1]

            # 4. Probabilités locales (avec décalage max pour stabilité)
            # exp(S - m_block)
            P_block = torch.exp(S_block - m_block)  # [B, H, block_size_q, block_size_kv]

            # 5. Somme locale des probabilités
            l_block = torch.sum(P_block, dim=-1, keepdim=True)  # [B, H, block_size_q, 1]

            # --- Mise à jour des statistiques courantes ---

            # Comparer le max du bloc avec le max courant
            m_old = m[:, :, q_start:q_end].clone()  # [B, H, block_size_q]
            m_new = torch.max(
                m_old.unsqueeze(-1),
                m_block.squeeze(-1),
                dim=-1
            )[0]  # [B, H, block_size_q]

            # Facteur alpha pour rescaler les anciens résultats
            alpha = torch.exp(m_old - m_new)  # [B, H, block_size_q]

            # Rescaler l'output accumulé jusqu'à présent
            output[:, :, q_start:q_end, :] = output[:, :, q_start:q_end, :] * alpha.unsqueeze(-1)

            # Rescaler et ajouter la nouvelle somme
            L[q_start:q_end] = L[:, :, q_start:q_end] * alpha + l_block.squeeze(-1)

            # Mettre à jour le max courant
            m[:, :, q_start:q_end] = m_new

            # 6. Accumuler le output partiel : P_block @ V_block
            #    [B, H, block_size_q, block_size_kv] @ [B, H, block_size_kv, D]
            #    = [B, H, block_size_q, D]
            output_contrib = torch.matmul(P_block, V_block)

            # Ajouter au résultat accumulé
            output[:, :, q_start:q_end, :] = output[:, :, q_start:q_end, :] + output_contrib

    # === Normalisation finale ===
    # Diviser par les sommes d'exponentiation
    output = output / L.unsqueeze(-1)  # [B, H, N_q, D]

    return output, L
```

Les points clés à comprendre :

1. **Boucle imbriquée :** Nous itérons sur tous les blocs KV, et pour chaque bloc KV, sur tous les blocs Q. Cela assure que toutes les paires (Q, K, V) sont traitées.

2. **Statistiques courantes :** Pour chaque bloc Q, nous maintenons m (max) et L (somme). Quand nous traitons un nouveau bloc KV, nous rescalons l'output précédent au besoin.

3. **Pas de stockage N×N :** La matrice Score n'est jamais plus grande que (block_size_q × block_size_kv), qui tient en shared memory.

4. **Online softmax :** Le softmax est appliqué progressivement bloc par bloc, pas globalement.

### 8.4 Backward pass : Gradient computation

Le backward pass est plus complexe car vous devez différencier par rapport à Q, K, V.

Les gradients clés sont :
- dQ : Contribution de l'erreur de chaque Q à ses gradients
- dK : Contribution de l'erreur de chaque K à ses gradients
- dV : Contribution de l'erreur de chaque V à ses gradients

Flash Attention 2 utilise une stratégie appelée **recomputation** : plutôt que de sauvegarder la matrice Score N×N complète (espace énorme), vous la recalculez lors du backward pass. C'est acceptable car le temps de recomputation est négligeable comparé aux économies d'espace.

Pseudocode du backward (simplifié) :

```python
def flash_attention_backward(
    dout,  # Gradient from upstream [B, H, N_q, D]
    Q, K, V,
    out,  # Output du forward [B, H, N_q, D]
    L,  # Logsumexp du forward [B, H, N_q]
    scaling_factor,
    causal=False
):
    """
    Flash Attention Backward Pass

    Returns:
        dQ, dK, dV avec les mêmes formes que Q, K, V
    """

    # Initialiser les gradients
    dQ = zeros_like(Q)
    dK = zeros_like(K)
    dV = zeros_like(V)

    # === Première boucle : Calculer dV et dK directement ===
    # Ces gradients dépendent de P (probabilités), qui dépend de scores

    # Boucler sur les blocs (similaire au forward)
    for kv_block_idx in range(num_blocks_kv):
        # ...charger K_block, V_block...

        for q_block_idx in range(num_blocks_q):
            # ...charger Q_block...

            # Recompute scores et probabilities (pour ce bloc)
            S_block = Q_block @ K_block.transpose(-2, -1) * scaling_factor
            m_block = max(S_block)
            P_block = exp(S_block - m_block)

            # Calculer les gradients partiels
            dout_block = dout[..., q_start:q_end, :]  # [B, H, block_size_q, D]

            # dV += P_block^T @ dout
            dV_block = P_block.transpose(-2, -1) @ dout_block
            dV[..., kv_start:kv_end, :] += dV_block

            # dP = dout @ V^T
            dP = dout_block @ V_block.transpose(-2, -1)  # [B, H, block_size_q, block_size_kv]

            # dS = d(softmax) wrt S = P * (dP - (P * dP).sum(dim=-1, keepdim=True))
            # C'est la dérivée de softmax
            sum_dP = (P_block * dP).sum(dim=-1, keepdim=True)
            dS = P_block * (dP - sum_dP)

            # dK += dS^T @ Q
            dK[..., kv_start:kv_end, :] += dS.transpose(-2, -1) @ Q_block

            # dQ_block = dS @ K
            dQ[..., q_start:q_end, :] += dS @ K_block

    # Appliquer le facteur d'échelle
    dQ = dQ * scaling_factor
    dK = dK * scaling_factor

    return dQ, dK, dV
```

Le backward est coûteux mais le coûts est caché par le recomputation de matrice Score au besoin.

### 8.5 Flash Attention 2 vs v1 : Améliorations

Flash Attention v1 (2022) était révolutionnaire mais avait des limitations. Flash Attention 2 (2023) apportait plusieurs améliorations :

| Aspect | Flash Attention v1 | Flash Attention 2 |
|---|---|---|
| **Ordre de boucle** | Outer loop sur Q, inner sur KV | Outer loop sur KV, inner sur Q (meilleur cache) |
| **Recomputation** | Recompute K, V | Recompute K, V + optimisation du coût |
| **Work partitioning** | Simple : 1 bloc Q = 1 bloc GPU | Sophistiqué : parallélisme flexible |
| **Warps dans un bloc** | Tous les warps partagent Q_block | Optimisation de workload balancing |
| **Performance** | ~2-3× plus rapide que attention naïve | ~3-5× plus rapide que attention naïve |

Flash Attention 2 réorganise les boucles pour une meilleure localité cache et parallélisme.

### 8.6 Choix de taille de bloc et impact sur l'occupancy

La taille de bloc affect directement l'occupancy (nous l'avons vu au Chapitre 2).

Pour Flash Attention sur A100 :
- **block_size_q = 128** : Chaque bloc Q contient 128 tokens
- **block_size_kv = 128** : Chaque bloc KV contient 128 tokens

Avec D = 64 (dimension par head) :

**Mémoire partagée requise :**
- Q_block : 128 × 64 × 2 bytes (FP16) = 16 KB
- K_block : 128 × 64 × 2 bytes = 16 KB
- V_block : 128 × 64 × 2 bytes = 16 KB
- Score intermédiaire : 128 × 128 × 2 bytes = 32 KB
- Total : ~80 KB (bien dans la limite de 192 KB)

**Registres requis :** ~50-100 registres par thread, ce qui limite l'occupancy à ~75-100%.

Ces paramètres offrent un bon équilibre entre occupancy et efficacité mémoire.

---

## Chapitre 9 : Mémoire partagée, conflits bancaires et primitives warp {#ch9}

### 9.1 Structure de la mémoire partagée : Les 32 banques

La **mémoire partagée** est une ressource précieuse sur GPU. Sur un A100, chaque SM a 192 KB de shared memory que vous pouvez diviser entre les threads d'un bloc.

Internement, la shared memory est organisée en **32 banques**. Chaque byte de shared memory appartient à l'une des 32 banques selon sa position. Pour une mémoire de 192 KB avec 32 banques, chaque banque contient 6 KB.

Les banques sont accédées en **parallèle**. Si chaque thread d'un warp accède à une banque différente, toutes les lectures/écritures peuvent se faire simultanément en 1 cycle. Mais si plusieurs threads accédent à la même banque, leurs accès sont **sérialisés**, ce qui crée un goulot d'étranglement appelé **bank conflict**.

### 9.2 Bank conflicts explication

Voici un exemple concret. Supposons vous avez un tableau de floats (4 bytes chacun) en shared memory :

```c
__shared__ float data[256];  // 1024 bytes, 32 banques de 32 bytes chacun
```

Avec 32 bytes par banque et 4 bytes par float, chaque banque contient 8 floats consécutifs.

**Accès sans conflict :**
```c
int tid = threadIdx.x;
float val = data[tid];  // Le warp lit data[0], data[1], ..., data[31]
// Bank 0 : data[0], data[32], data[64], ...
// Bank 1 : data[1], data[33], data[65], ...
// Chaque thread du warp accède à une banque différente → 1 cycle
```

**Accès avec conflict :**
```c
int tid = threadIdx.x;
float val = data[tid * 2];  // Stride de 2 !
// Thread 0 lit data[0] (bank 0)
// Thread 1 lit data[2] (bank 0, même banque !)
// Thread 2 lit data[4] (bank 0)
// ...
// Tous les 16 threads du premier demi-warp lisent la banque 0
// Les accès doivent être sérialisés → 16 cycles (catastrophe !)
```

### 9.3 Padding pour éviter les conflicts

La solution standard est le **padding**. Vous ajoutez des dummy bytes après chaque ligne de données pour décaler les banques.

Pour une matrice 2D en shared memory, l'approche classique est :

```c
// Sans padding : 64 floats per row = 256 bytes = 8 banques
// Tous les éléments d'une colonne tombent dans la même banque !
__shared__ float matrix[64][64];  // Problème : accès colonne → conflicts

// Avec padding : chaque ligne a une colonne vide
__shared__ float matrix_padded[64][65];  // 65*4 = 260 bytes, décale les colonnes
// Maintenant matrix_padded[0][0], matrix_padded[1][0], ... vont à des banques différentes
```

L'ajout d'une seule colonne (au moins pour D=64, on ajoute 8 floats ≈ 32 bytes) décale les accès colonne par colonne pour éviter les conflicts.

Pour Flash Attention avec FP16 (2 bytes par element) et D=64 :
- Ligne : 64 × 2 bytes = 128 bytes = 4 banques
- Ajout de padding : +8 half = 16 bytes = 0.5 banque supplémentaire
- Nouvelle structure : 64 floats + 8 padding demi-floats per row

```c
__shared__ __half kv_block[BLOCK_SIZE][D + 8];  // D=64, BLOCK_SIZE=128
// Maintenant les accès colonne dans kv_block ne conflit pas
```

### 9.4 Warp shuffles et réductions

Les **warp shuffles** sont des opérations spécialisées qui permettent aux threads d'un warp d'échanger des données directement sans passer par shared memory.

L'instruction clé est `__shfl_xor_sync(mask, var, laneid)`. Elle échange la valeur de `var` entre les threads dont les IDs diffèrent par `laneid` (opération XOR).

Exemple : Réduction somme d'un warp

```c
__global__ void warp_reduce_kernel(float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // ID du thread dans le warp (0-31)
    int warp = threadIdx.x / 32;  // ID du warp (0-31)

    // Somme locale
    float sum = (idx < N) ? input[idx] : 0.0f;

    // Réduction de warp : somme les 32 valeurs
    // Étape 1 : Échanger avec le thread 16 positions plus loin
    sum += __shfl_xor_sync(0xffffffff, sum, 16);

    // Étape 2 : Échanger avec le thread 8 positions plus loin
    sum += __shfl_xor_sync(0xffffffff, sum, 8);

    // Étape 3 : Échanger avec le thread 4 positions plus loin
    sum += __shfl_xor_sync(0xffffffff, sum, 4);

    // Étape 4 : Échanger avec le thread 2 positions plus loin
    sum += __shfl_xor_sync(0xffffffff, sum, 2);

    // Étape 5 : Échanger avec le thread 1 position plus loin
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    // Maintenant le thread 0 contient la somme complète
    if (lane == 0) {
        output[warp] = sum;
    }
}
```

Ce schéma butterfly est élégant et très rapide (aucune latence mémoire, tout en registres). Les 5 étapes correspondent à 2^5 = 32 threads.

```
Étape 0 : [0  1  2  3  4  5  6  7  8  9 ... 31]
            ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓ ↓↓    ↓↓
Étape 1 : [0  1  2  3  4  5  6  7  8  9 ... 31]
            (échange sur 16)
                                      ↓ échange sur 8
Étape 2 : [0  1  2  3  4  5  6  7  8  9 ... 31]
                                  ↓ échange sur 4
Étape 3 : [0  1  2  3  4  5  6  7  8  9 ... 31]
                              ↓ échange sur 2
Étape 4 : [0  1  2  3  4  5  6  7  8  9 ... 31]
                             ↓ échange sur 1
Étape 5 : Somme finale dans thread 0
```

Chaque échange XOR est un papillon du réseau de réduction.

### 9.5 Réductions inter-warp via shared memory

Pour réduire sur plusieurs warps (par exemple, réduire un bloc complet), vous devez utiliser la shared memory car les shuffles ne fonctionnent qu'intra-warp.

Schéma général :

```c
__global__ void block_reduce_kernel(float *output, float *input, int N) {
    __shared__ float sdata[32];  // Une valeur par warp maximum

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    // Somme locale
    float sum = (idx < N) ? input[idx] : 0.0f;

    // Réduction intra-warp (butterfly)
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    // Chaque warp écrit son résultat en shared memory
    if (lane == 0) {
        sdata[warp] = sum;
    }
    __syncthreads();  // S'assurer que tous les warp ont écrit

    // Dernière réduction (faite uniquement par le premier warp)
    if (tid < blockDim.x / 32) {
        sum = sdata[tid];
        sum += __shfl_xor_sync(0xffffffff, sum, 16);
        sum += __shfl_xor_sync(0xffffffff, sum, 8);
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
    }

    // Le thread 0 du bloc écrit le résultat final
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}
```

Ce pattern est fondamental pour les réductions efficaces en CUDA. Flash Attention l'utilise pour les réductions de softmax locales.

---

## Chapitre 10 : Triton — Abstraire le CUDA pour le rendre accessible {#ch10}

### 10.1 Motivation : CUDA est difficile

Écrire du CUDA efficace est difficile. Vous devez penser à :
- **Synchronisation** : `__syncthreads()`, `__shfl_sync()`, etc.
- **Mémoire partagée** : allocation, bank conflicts, padding
- **Registres** : occupancy, spilling en local memory
- **Coalescing** : les accès mémoire doivent être alignés
- **Warps et blocs** : configuration et équilibre de charge

Un petit changement peut casser la performance en moitié.

De plus, déboguer CUDA est un cauchemar. Les race conditions sont silencieuses, les deadlocks sont cryptiques, les erreurs mémoire sont difficiles à localiser.

**Triton** (développé par OpenAI et maintenant par NVIDIA/community) vise à abstraire ces détails. L'idée est simple : **pensez en blocs 2D/3D, pas en threads individuels**.

### 10.2 Le modèle de programmation Triton

Triton abstrait la programmation GPU au niveau des blocs plutôt que des threads. Voici les concepts clés :

**Block :** Une unité de calcul qui opère sur une tile de données. Triton gère automatiquement comment diviser cette tile entre les threads.

**Pointeurs :** Au lieu de gérer les indices manuellement, vous travaillez avec des pointeurs qui avancent automatiquement.

**Autotuning :** Triton peut tester automatiquement différentes tailles de blocs et configurations.

Voici un exemple simple : addition élément par élément en Triton.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    """
    Kernel Triton pour addition : output = x + y
    """
    # Calculer l'indice du bloc (chaque bloc traite BLOCK_SIZE éléments)
    pid = tl.program_id(axis=0)  # ID du bloc (0, 1, 2, ...)

    # Indice de départ pour ce bloc
    block_start = pid * BLOCK_SIZE

    # Créer un vecteur d'indices pour ce bloc
    # [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Charger les données (Triton gère automatiquement l'alignement mémoire)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements)

    # Calculer
    output = x + y

    # Stocker le résultat
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

# Appel du kernel
import torch

def add_triton(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Nombre de blocs nécessaires
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 1D grid

    # Lancer le kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```

Comparé au CUDA équivalent, c'est beaucoup plus lisible. Triton gère :
- Le mappage de chaque block ID aux indices
- L'alignement des accès mémoire
- Les masks pour les limites du tenseur
- Le coalescing automatique

### 10.3 Opérations courantes : load, store, dot

Les opérations courantes en Triton :

**tl.load(pointer, mask=None)** : Charger des données depuis la mémoire globale.
- `pointer` : Pointeur (peut être un vecteur d'indices)
- `mask` : Mask booléen pour les charges conditionnelles
- Retourne un tenseur

**tl.store(pointer, value, mask=None)** : Stocker des données en mémoire globale.
- `pointer` : Pointeur destination
- `value` : Tenseur de valeurs
- `mask` : Mask pour les stockages conditionnels

**tl.dot(a, b)** : Produit scalaire ou produit matriciel (matmul).
- Utilise automatiquement les Tensor Cores si disponibles
- Beaucoup plus rapide que d'écrire la boucle manuellement

Exemple : GEMM (General Matrix Multiply) simple en Triton

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel Triton pour C = A @ B
    A : [M, K], B : [K, N], C : [M, N]
    """
    # ID du bloc
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Indice de départ pour ce bloc
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Boucle sur K (produit scalaire)
    for k in range(0, K, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)

        # Charger A et B
        a = tl.load(
            a_ptr + m_idx[:, None] * stride_am + k_idx[None, :] * stride_ak,
            mask=(m_idx[:, None] < M) & (k_idx[None, :] < K)
        )
        b = tl.load(
            b_ptr + k_idx[:, None] * stride_bk + n_idx[None, :] * stride_bn,
            mask=(k_idx[:, None] < K) & (n_idx[None, :] < N)
        )

        # Accumulation
        acc += tl.dot(a, b)

    # Stocker le résultat
    c_idx_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_idx_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.store(
        c_ptr + c_idx_m[:, None] * stride_cm + c_idx_n[None, :] * stride_cn,
        acc,
        mask=(c_idx_m[:, None] < M) & (c_idx_n[None, :] < N)
    )
```

### 10.4 Autotuning

L'autotuning est l'une des grandes forces de Triton. Vous pouvez définir des paramètres comme `BLOCK_SIZE` et laisser Triton tester automatiquement différentes valeurs et retenir la plus rapide.

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32),
    ],
    key=["n_elements"]  # Reoptimiser basé sur n_elements
)
@triton.jit
def optimized_add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
    # même code que avant
    ...
```

Triton compile et teste chaque configuration, mesure les temps, et garde la plus rapide. C'est automatique !

### 10.5 CUDA vs Triton : Tableau comparatif

| Aspect | CUDA | Triton |
|---|---|---|
| **Niveau d'abstraction** | Threads/Warps | Blocs |
| **Synchronisation** | `__syncthreads()`, `__shfl_sync()`, etc. | Automatique (lazy) |
| **Mémoire partagée** | Allocation manuelle, bank conflicts | Abstraite, gérée automatiquement |
| **Registres** | Gérés par le compilateur | Gérés par le compilateur |
| **Coalescing** | À faire manuellement | Automatique |
| **Langage** | C++ avec extensions CUDA | Python + Python IR (Triton IR) |
| **Compilation** | Just-In-Time (JIT) | Just-In-Time (JIT) |
| **Performances** | Potentiellement maximales (experts) | ~5-20% moins rapide (optimisation manuelle difficile) |
| **Productivité** | Lente pour débuter | Rapide pour prototyper |
| **Autotuning** | Manuel, fastidieux | Automatique, simple |
| **Debugging** | Difficile, peu d'outils | Meilleur (Python-like) |
| **Communauté** | Énorme, documentation abondante | En croissance rapide |

### 10.6 Flash Attention en Triton

Flash Attention a été implémenté en Triton par Tri Dao (l'auteur original). Voici une esquisse de ce que cela ressemble (simplifié) :

```python
@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, out, L,
    batch_size, seq_len, d_head,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention forward en Triton
    Simplifié pour clarity
    """
    # ID du bloc
    block_m = tl.program_id(axis=0)
    block_batch = tl.program_id(axis=1)
    block_head = tl.program_id(axis=2)

    # Limites du bloc Q
    start_m = block_m * BLOCK_M
    end_m = tl.minimum(start_m + BLOCK_M, seq_len)

    # Initialiser les statistiques
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, d_head), dtype=tl.float32)

    # Charger Q
    q = tl.load(Q + ...)  # Simplifiée

    # Boucle sur les blocs K, V
    for block_n in range(tl.cdiv(seq_len, BLOCK_N)):
        start_n = block_n * BLOCK_N
        end_n = tl.minimum(start_n + BLOCK_N, seq_len)

        # Charger K, V
        k = tl.load(K + ...)
        v = tl.load(V + ...)

        # Calculer les scores
        s = tl.dot(q, k.transpose())  # [BLOCK_M, BLOCK_N]

        # Appliquer max-shift
        m_j = tl.max(s, axis=1)
        s = s - m_j[:, None]

        # Probabilités
        p = tl.exp(s)

        # Somme
        l_j = tl.sum(p, axis=1)

        # Rescaler les résultats précédents
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_j
        m_i = m_new

        # Accumuler
        acc = acc + tl.dot(p, v)

    # Normaliser
    out = acc / l_i[:, None]

    # Stocker
    tl.store(out + ...)
```

Ce code Triton est beaucoup plus lisible que le CUDA équivalent. Les détails d'implémentation basse niveau (bank conflicts, synchronisation, occupancy) sont gérés automatiquement.

### 10.7 Quand utiliser Triton vs CUDA

**Utiliser Triton si :**
- Vous optimisez pour la productivité et la readability
- Vous prototypez rapidement
- Les performances 5-20% moins bonnes sont acceptables
- Vous aimez Python et les notebooks Jupyter
- Vous voulez autotuning automatique

**Utiliser CUDA si :**
- Vous avez besoin de la dernière goutte de performance
- Vous optimisez pour des millions de requêtes (5-20% compte)
- Vous faites de la micro-optimization (shared memory, register usage)
- Vous avez besoin d'astuces de bas niveau (memory intrinsics, etc.)
- La communauté CUDA et les benchmarks sont essentiels

Pour Flash Attention, Triton est une excellente option car l'algorithme est déjà bien compris, et laisser Triton gérer les détails bas niveaux réduit les risques de bugs.

---

**FIN DE LA PARTIE 1 — FONDATIONS THÉORIQUES**

*Résumé : Nous avons couvert 10 chapitres d'architecture GPU, programmation CUDA, mécanique d'attention, optimisations de softmax, analyse de performance Roofline, l'algorithme révolutionnaire Flash Attention avec backward pass, optimisations mémoire partagée, et les abstractions Triton. La base théorique est maintenant solide pour aborder l'implémentation détaillée en PARTIE 2.*

---

# PARTIE 2 : Architecture et Anatomie du Projet

## 2.1 Arborescence du Projet Annotée

```
01-flash-attention/
├── README.md                          # Vue d'ensemble et quick start
├── Makefile                          # Orchestration : build, test, bench, clean
├── cuda/
│   ├── CMakeLists.txt                # Build system (cmake 3.18+, multi-arch)
│   ├── setup.py                      # PyTorch CUDAExtension build wrapper
│   ├── include/                      # Header-only utilities
│   │   ├── cuda_utils.cuh            # Macros : CUDA_CHECK, CudaTimer, cdiv
│   │   ├── smem_utils.cuh            # Shared mem : padding, tile load/store
│   │   ├── online_softmax.cuh        # Warp/block reduce, OnlineSoftmaxState
│   │   └── flash_attn.cuh            # Kernel declarations (public API)
│   ├── src/                          # Implementation kernels
│   │   ├── naive_attention.cu        # 3 kernels : QK, softmax, PV
│   │   ├── flash_attn_fwd.cu         # Forward pass (main kernel)
│   │   ├── flash_attn_bwd.cu         # Backward pass (D, main, convert)
│   │   └── torch_bindings.cpp        # PyTorch C++ extension wrapper
│   └── tests/                        # Correctness & edge cases
│       ├── test_correctness.py       # ref_attention, test_naive/flash_fwd/bwd
│       └── test_edge_cases.py        # Non-aligned seqlen, short seqs, head_dim
├── triton/                           # Pure Triton implementations
│   ├── flash_attn_triton.py         # Triton fwd/bwd with autotuning
│   ├── test_triton.py               # Correctness tests vs reference
│   └── bench_triton.py              # Triton-only benchmarks
└── benchmarks/                       # Performance evaluation suite
    ├── bench_all.py                 # 5 impls: naive, flash, SDPA, Triton, Dao
    ├── plot_results.py              # Matplotlib visualisations (latency, TFLOPS)
    └── results/                     # JSON outputs from benchmarks

```

### Fichiers clés et responsabilités

| Fichier | Lignes | Responsabilité |
|---------|--------|-----------------|
| cuda_utils.cuh | 97 | Macros CUDA_CHECK, timers, division entière, limits |
| smem_utils.cuh | 125 | Padding bancaire, load/store coopératifs de tuiles |
| online_softmax.cuh | 128 | Réductions warp/block, OnlineSoftmaxState |
| flash_attn.cuh | 56 | Déclarations C : naive_attention_fwd, flash_attn_fwd, flash_attn_bwd |
| naive_attention.cu | 185 | 3 kernels indépendants (QK, softmax, PV) + launcher |
| flash_attn_fwd.cu | 265 | Kernel flash forward + launcher |
| flash_attn_bwd.cu | 299+ | compute_D_kernel, kernel backward, convert lambda, launcher |
| torch_bindings.cpp | 130+ | Wrapper Python (CHECK_*, ptr helpers, module binding) |
| flash_attn_triton.py | 250+ | Fwd/bwd Triton @jit, autotuning, wrappers Python |
| CMakeLists.txt | 49 | Build : static lib, optional standalone, architecture auto-detect |
| setup.py | 20+ | Build PyTorch extension via CUDAExtension |

---

## 2.2 Diagramme d'Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                         APPLICATION PYTHON                         │
│  (PyTorch: forward/backward, autograd.Function, optimizer)          │
└────────────┬────────────────────────────┬─────────────────────┬────┘
             │                            │                     │
    ┌────────▼────────────────────────────▼──────┐    ┌────────▼──────┐
    │   C++ PyTorch Extension Layer              │    │  Triton       │
    │   (torch_bindings.cpp)                     │    │  (autograd)   │
    │                                            │    │               │
    │  naive_fwd(Q,K,V)→O                        │    │ _flash_attn   │
    │  flash_fwd(Q,K,V)→(O,L)                    │    │ _fwd/bwd_kernel
    │  flash_bwd(Q,K,V,O,dO,L)→(dQ,dK,dV)       │    │               │
    │                                            │    │ Python wrapper│
    └────────┬────────────────────────────┬──────┘    └────────┬──────┘
             │                            │                    │
    ┌────────▼────────────────────────────▼──────────────────┬┘
    │           CUDA Kernel Launch Layer (C++)              │
    │  (flash_attn.cuh declarations, host launchers)        │
    └────────┬─────────────────────────────┬────────────────┘
             │                             │
    ┌────────▼───────────────┐   ┌────────▼────────────────┐
    │  CUDA Kernels (GPU)    │   │  Triton Compiles       │
    │                        │   │  to PTX (GPU)          │
    │ - naive_qk_kernel      │   │                        │
    │ - naive_softmax        │   │ - _flash_attn_fwd_k    │
    │ - naive_pv_kernel      │   │ - _flash_attn_bwd_k    │
    │ - flash_attn_fwd_k     │   │ - lambda convert       │
    │ - compute_D_kernel     │   │                        │
    │ - flash_attn_bwd_k     │   │  (shared mem, warps    │
    │ - lambda convert       │   │   managed by Triton)   │
    │                        │   │                        │
    │ (explicit smem, reg    │   │                        │
    │  mgmt, warp reduce)    │   │                        │
    └────────┬───────────────┘   └────────┬───────────────┘
             │                            │
             └────────┬───────────────────┘
                      │
            ┌─────────▼──────────┐
            │   GPU Memory HBM   │
            │  Q, K, V, O, L, d* │
            │  (+ smem, regs)    │
            └────────────────────┘

```

---

## 2.3 Flux de Données : Forward Pass

**Étape 1 : Chargement des entrées (PyTorch → HBM)**
- Application PyTorch crée des tenseurs [B, H, N, D] en FP16 sur GPU
- Tensors contiguës en mémoire (row-major)

**Étape 2 : Reshape logique (B*H fusion)**
- torch_bindings.cpp reshape mentalement (B,H,N,D) → (B*H, N, D)
- Aucune copie : juste pointer offset

**Étape 3 : Lancement du kernel Forward**
```
Grid:  (cdiv(N, FWD_BR), B*H)  → (cdiv(1024, 64), 8*12) = (16, 96) blocs
Block: (128 threads)
```

**Étape 4 : À l'intérieur du bloc flash_attn_fwd_kernel**

```
Pour chaque bloc Q (64 lignes) :

  ├─ 1. Charger Q en registres (128 threads, parallèle)
  │     Q_smem n'existe pas → Q reste en registres
  │
  │     Allocation registres : q_reg[128], o_acc[128], s_row[64]
  │
  ├─ 2. Boucle sur blocs KV (0 à N/64) :
  │     │
  │     ├─ 2.1 Charger K, V dans smem (coopératif par tous threads)
  │     │       K_smem[64][68] ← K_bh[kv_start*D : D]  (padding bancaire)
  │     │       V_smem[64][68] ← idem
  │     │       __syncthreads()
  │     │
  │     ├─ 2.2 GEMM-I : S = Q @ K^T (scores)
  │     │       - Chaque thread : dot(q_reg[i], K_smem[j])
  │     │       - Réduction warp si 2 threads par ligne
  │     │       - s_row[j] = dot * scale + masque
  │     │
  │     ├─ 2.3 Online Softmax update
  │     │       - m_block = max(s_row)
  │     │       - alpha = exp(m_i - m_new)
  │     │       - Rescale o_acc *= alpha
  │     │       - s_row = exp(s_row - m_new)
  │     │
  │     ├─ 2.4 GEMM-II : O += P @ V
  │     │       - o_acc[i] += s_row[j] * V_smem[j][i]
  │     │       - Accumulation incrémentale
  │     │
  │     └─ 2.5 __syncthreads() avant next KV block
  │
  └─ 3. Écriture (O, L) en HBM
        O_bh[row][d] = o_acc[d] / l_i
        L_bh[row] = m_i + log(l_i)
```

**Étape 5 : Retour à PyTorch**
- O [B*H, N, D] FP16
- L [B*H, N] FP32 (pour backward)

---

## 2.4 Flux de Données : Backward Pass

**Entrées supplémentaires :** O, dO (grad de output), L (sauvegardé en fwd)

```
┌─ Étape 1 : compute_D_kernel
│  Pour chaque ligne q :
│    D[q] = sum_d( dO[q,d] * O[q,d] )
│  Grille: (cdiv(N, 256), B*H)
│
├─ Étape 2 : Allocation dQ_f
│  dQ_f [B*H, N, D] en float32 (pour atomicAdd)
│
├─ Étape 3 : flash_attn_bwd_kernel
│  Boucle externe : sur blocs KV (outer loop)
│  Boucle interne : sur blocs Q
│
│  Pour chaque KV row :
│    ├─ Charger K, V en registres
│    ├─ Accumulateur dk_acc, dv_acc
│    │
│    ├─ Pour chaque Q bloc :
│    │   ├─ Charger Q, dO en smem
│    │   │
│    │   ├─ Pour chaque Q row :
│    │   │   ├─ Recompute S = Q @ K^T
│    │   │   ├─ Recompute P = exp(S - L)
│    │   │   ├─ dP = dO @ V^T
│    │   │   ├─ dS = P * (dP - D_i)
│    │   │   │
│    │   │   ├─ dV += P * dO^T
│    │   │   ├─ dK += dS * Q^T * scale
│    │   │   └─ atomicAdd(&dQ_f, dS * K * scale)
│    │   │
│    │   └─ __syncthreads()
│    │
│    └─ Écrire dK, dV en HBM (1 fois par KV row)
│
└─ Étape 4 : Lambda kernel convert
   Pour chaque élément :
     dQ_half[idx] = __float2half(dQ_f[idx])
   Grille: (cdiv(B*H*N*D, 256), ...)

Retour: dQ, dK, dV tous en FP16
```

---

## 2.5 Flux de Données : Benchmark

```
┌─ Charger 5 implémentations ─────────────────────┐
│                                                  │
│ 1. naive CUDA (torch_bindings)                  │
│ 2. flash CUDA (torch_bindings)                  │
│ 3. PyTorch SDPA (torch.nn.functional)           │
│ 4. Triton custom (flash_attn_triton.py)         │
│ 5. Dao flash-attn (pip package, si disponible) │
│                                                  │
└────────────────┬─────────────────────────────────┘
                 │
        Pour chaque (B, H, N, D) :
        │
        ├─ Générer Q, K, V aléatoires
        │
        ├─ Pour chaque impl :
        │   ├─ Warmup (10 iters)
        │   ├─ Benchmark (100 iters)
        │   ├─ CUDA sync
        │   ├─ Mesurer latence (ms)
        │   ├─ Calculer TFLOPS = 4*B*H*N²*D / (latency*1e-3) / 1e12
        │   └─ Sauver résultats JSON
        │
        └─ Générer graphes (matplotlib)

```

---

## 2.6 Cycle de Vie du Projet : Build → Test → Bench → Plot

### Phase 1 : Build

```bash
$ make install
  ├─ cd cuda && pip install -e . --no-build-isolation
  │   └─ Exécute setup.py
  │       ├─ CMake trouve CUDA toolkit
  │       ├─ Compile .cu en .o (device code)
  │       ├─ Compile torch_bindings.cpp (host + device link)
  │       └─ Link tout → flash_attn_cuda.*.so
  │
  └─ Extension Python prête à importer
```

### Phase 2 : Test

```bash
$ make test
  ├─ Exécute pytest cuda/tests/test_correctness.py
  │   ├─ test_naive_fwd : compare vs ref_attention (float32)
  │   ├─ test_flash_fwd : compare O et L
  │   └─ test_flash_bwd : compare dQ, dK, dV
  │
  └─ Affiche relative error % pour chaque test

$ make test-triton
  └─ pytest triton/test_triton.py (même logique)
```

### Phase 3 : Benchmark

```bash
$ make bench
  └─ python benchmarks/bench_all.py
      ├─ Pour (B,H,N,D) ∈ {petits, moyens, grands}
      ├─ Benchmark 5 implémentations
      └─ Sauve JSON → benchmarks/results/*.json

$ make bench-triton
  └─ python triton/bench_triton.py (Triton only)
```

### Phase 4 : Visualisation

```bash
$ make plots
  └─ python benchmarks/plot_results.py
      ├─ Lit benchmarks/results/*.json
      ├─ Trace : latence vs N, TFLOPS vs D, etc.
      └─ PDF/PNG dans benchmarks/plots/
```

### Phase 5 : Nettoyage

```bash
$ make clean
  ├─ Efface build/ (CMake artefacts)
  ├─ Efface cuda/build, dist, *.egg-info
  └─ Efface .so, __pycache__
```

---

## 2.7 Tableau Comparatif : Choix d'Architecture

| Aspect | Choix | Justification |
|--------|-------|--------------|
| **Stockage Q** | Registres | Q petit (64×64), répété dans boucle KV → cache locality. Pas de conflit smem. |
| **Stockage K, V** | Shared mem | Réutilisé dans boucle interne (Q). Permet coalescing HBM→smem. |
| **Softmax** | Online algorithm | Évite O(N²) mémoire. Recompute possible en backward. |
| **Reduction Warp** | __shfl_xor_sync | Pas de smem latency. 5 itérations butterfly log2(32). |
| **Mapping Threads** | 2 threads/row pour Br=64 | Équilibre : pas trop idle, réduction simple. |
| **Backward K,V** | Registres (outer loop) | Une fois par KV block → pas de reload. Efficace. |
| **Backward dQ** | atomicAdd float32 | Accumulation multi-bloc. Conversion float16 après. |
| **Padding SMEM** | 8 halfs | Multiplie stride par 2 → 2*64 ≠ 32 banks (pas conflit). |
| **Causal Mask** | Condition + -1e20f | Appliqué après dot product (efficace). Alternative : skip KV blocs. |
| **Architecture Multi-GPU** | Batch/Head parallelization | Chaque SM indépendant. Pas sync cross-GPU (voir Megatron). |

---

## 2.8 Conventions de Nommage et Patterns Récurrents

### Nommage des Variables

- **Indices globaux :** `my_q_row_global`, `kv_start`, `q_start`
- **Indices locaux :** `my_q_row_local`, `my_kv_local`, `tid`, `lane`, `warp_id`
- **Dimensions :** `N` (seq_len), `D` (head_dim), `Br` (Q block size), `Bc` (KV block size)
- **Registres floats :** `q_reg[128]`, `o_acc[128]`, `s_row[64]` (tous float32 interne)
- **Accumus :** `m_i`, `l_i` (online softmax), `dk_acc`, `dv_acc` (backward)

### Pattern : Cooperative Tile Load

```c
// Tous threads du bloc travaillent ensemble pour charger une tuile
for (int idx = tid; idx < total; idx += blockDim.x) {
    int r = idx / cols;      // decompose linear index
    int c = idx % cols;
    // load, convert, write
}
__syncthreads();  // tout le monde a fini
```

Récurrent dans : smem_load_tile, smem_load_tile_strided, smem_store_tile.

### Pattern : Warp-level Reduction

```c
float val = ...;
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
}
```

Utilisé pour max/sum sur 32 threads → 1 résultat en 5 itérations.

### Pattern : Online Softmax Update

```c
float m_new = fmaxf(m_i, m_block);
float alpha = __expf(m_i - m_new);      // rescale old
float beta  = __expf(new_max - m_new);
l_i = alpha * l_i + beta * new_sum;
m_i = m_new;
```

Stable numériquement → aucun overflow/underflow.

### Pattern : Causal Masking

```c
if (causal && global_j > my_q_row_global)
    dot = -1e20f;  // -inf en float
```

Appliqué une fois par score. Alternative : skip KV blocs (logique, mais moins régulière).

---

## 2.9 Considérations de Performance : Goulots d'Étranglement

| Goulot | Manifestation | Mitigation |
|--------|---------------|-----------|
| **Mémoire HBM** | Bande passante limitée à ~2 TB/s | Computation/Bytes ratio. Online softmax. |
| **Conflits bancaires SMEM** | Bank serialization si stride=32 | SMEM_PAD_HALFS=8 → stride=72 (pas 64) |
| **Latence compute** | S = Q @ K^T très rapide | Masked par latency HBM si blocs petits |
| **Divergence threads** | if (causal) → branches | Masque uniform par warp → pas de réelle divergence |
| **Register pressure** | Si trop de registres → spill SMEM | q_reg[128], o_acc[128] ≤ ~256 floats OK |
| **SM occupancy** | Peu de blocs par SM → idle cores | FWD_BR=64, FWD_BC=64, 128 threads → 4 warps/bloc |

---

# PARTIE 3 : Analyse Profonde du Code

Chaque fichier reçoit : rôle, dépendances, walkthrough ligne-à-ligne, lien théorie, diagrammes, pièges.

## 3.1 cuda_utils.cuh — Utilitaires CUDA de base

### Rôle et Responsabilité

Fournit des **macros et fonctions helper** pour :
- Vérification et rapport d'erreurs CUDA
- Timing précis via GPU events
- Arithmétique entière (ceiling division)
- Introspection du device (mémoire partagée)
- Configuration dynamique de SMEM limite

C'est le **fondement** : tout autre fichier .cuh/.cu l'inclut.

### Dépendances

- **Dépend de :** `<cuda_runtime.h>`, `<cstdio>`, `<cstdlib>`
- **Dépendants :** TOUS les autres fichiers

### Walkthrough Ligne-à-Ligne

#### Bloc 1 : CUDA_CHECK macro (lignes 13-21)

```c
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
```

**Ligne 14 : `do { ... } while (0)`** — Idiome C pour wrapper macro sûr. Permet utiliser dans if sans accolades.

**Ligne 15 : `cudaError_t err = (call)`** — Exécute l'appel CUDA, capture le code d'erreur.

**Ligne 16 : Check explicite** — `err != cudaSuccess` déclenche rapport.

**Ligne 17-18 : Message diagnostic** — `__FILE__`, `__LINE__` pour localiser l'appelant. `cudaGetErrorString(err)` décode le code.

**Ligne 19 : Termination** — `exit(EXIT_FAILURE)` arrête le processus immédiatement. Brutal, mais prévient silent errors.

**Utilisation courante :**
```c
CUDA_CHECK(cudaMalloc(&d_Q, bytes));
CUDA_CHECK(cudaMemcpyAsync(d_Q, h_Q, bytes, cudaMemcpyHostToDevice, stream));
```

#### Bloc 2 : CUDA_CHECK_LAST macro (lignes 23-31)

```c
#define CUDA_CHECK_LAST()                                                  \
    do {                                                                   \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA kernel error at %s:%d — %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
```

**Différence clé :** `cudaGetLastError()` au lieu d'appel direct. Captures les erreurs de **kernel launch asynchrone**.

Après un `kernel<<<grid, block>>>(...)`, l'erreur n'est connue que quand le runtime valide l'appel (avant exécution GPU). `CUDA_CHECK_LAST()` capture cette validation.

**Utilisation :**
```c
flash_attn_fwd_kernel<<<grid, block, smem_bytes, stream>>>(Q, K, V, O, L, N, D, scale, causal);
CUDA_CHECK_LAST();  // Découvre les erreurs de launch (overflow smem, etc.)
```

#### Bloc 3 : CudaTimer struct (lignes 37-61)

```c
struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    float toc(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
```

**Concept :** GPU **events** sont des points de synchronisation légers. `cudaEvent_t` est un marqueur dans le stream.

**Constructeur (lignes 40-43):** Alloue 2 events sur le GPU. Chaque event consomme ~100 bytes mémoire device.

**Destructeur (lignes 44-47):** Libère. RAII pattern → pas de fuite.

**tic() (lignes 49-51):** Enregistre start event dans le stream. Non-bloquant : retour immédiat (CPU continue).

**toc() (lignes 53-60):**
- Ligne 54 : Enregistre stop event
- Ligne 55 : **Synchronize** — bloque CPU jusqu'à stop event complet (GPU a fini)
- Ligne 57-58 : `cudaEventElapsedTime` retourne ms écoulé entre start et stop
- Ligne 59 : Retourne latence

**Utilisation typique :**
```c
CudaTimer timer;
timer.tic();
for (int i = 0; i < 100; ++i) {
    kernel<<<grid, block>>>(args...);
}
float ms = timer.toc();
printf("Avg: %.3f ms\n", ms / 100.0f);
```

**Avantage vs CPU clock :** Mesure temps GPU pur, pas CPU overhead. Précision ~us.

#### Bloc 4 : cdiv (lignes 68-69)

```c
__host__ __device__ __forceinline__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
```

**Définition :** Ceiling division pour entiers positifs. Équivalent à `ceil(a/b)`.

**Exemple :** `cdiv(1024, 64) = (1024 + 63) / 64 = 1087 / 64 = 16`. Correct (1024=64*16).

**Attributs :**
- `__host__ __device__` : Compilé pour CPU et GPU
- `__forceinline__` : Force inlining (0 overhead d'appel fonction)
- `constexpr` : Peut être évalué compile-time si arguments constants

**Usage :** Calculer nombre de blocs → `dim3 grid(cdiv(N, FWD_BR), BH);`

#### Bloc 5 : print_smem_info (lignes 79-89)

```c
inline void print_smem_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_per_block, smem_per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block,
               cudaDevAttrMaxSharedMemoryPerBlock, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm,
               cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    printf("Device %d — smem/block: %d KB, smem/SM: %d KB\n",
           device, smem_per_block / 1024, smem_per_sm / 1024);
}
```

**Utilité :** Introspection du GPU pour tuning SMEM.

**Exemple output :** `Device 0 — smem/block: 96 KB, smem/SM: 96 KB` (A100). Vs GTX 1080: 48 KB/block.

**Appel:** Utile en début de main() pour vérifier capacités avant kernel launch.

#### Bloc 6 : set_smem_limit (lignes 92-96)

```c
template <typename Kernel>
inline void set_smem_limit(Kernel kernel, int smem_bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
}
```

**Contexte :** Flash Attention utilise **shared memory dynamique** (non constant au compile-time).

Par défaut, CUDA limite SMEM dynamique à 96 KB (héritage du compute capability). Si besoin > 96 KB (rare sur A100), il faut appeler `cudaFuncSetAttribute`.

**Appel dans flash_attn_fwd :**
```c
int smem_bytes = 2 * FWD_BC * d_padded * sizeof(half) + 32 * sizeof(float);
set_smem_limit(flash_attn_fwd_kernel, smem_bytes);
```

Configu AVANT le kernel launch.

### Pont vers Théorie (Chapitre 3 & 9)

Comme vu au **Chapitre 3, Section 3.3 (Gestion d'erreurs)**, les GPUs exécutent asynchrones. `CUDA_CHECK` synchronise et valide. `CUDA_CHECK_LAST` capture les **kernel launch errors** (grille trop grande, SMEM overflow, etc.).

Au **Chapitre 9, Section 9.4**, SMEM est la ressource contrainte. `print_smem_info()` et `set_smem_limit()` gèrent cette ressource.

### Diagramme : Flux d'Erreur

```
      Appel CUDA (e.g. cudaMalloc)
           │
           ├─ Immédiatement : validation CPU
           │   └─ Erreur → CUDA_CHECK capture
           │
           └─ Kernel launch <<<...>>>
               │
               └─ Asynchrone : GPU reçoit le command
                   │
                   ├─ Queue d'exécution : attendez GPU
                   │   └─ Erreur latente (smem overflow, etc.)
                   │
                   └─ CUDA_CHECK_LAST() appelle cudaGetLastError()
                       └─ Retourne l'erreur asynchrone
```

### Pièges et Subtilités

1. **exit(EXIT_FAILURE) brutal :** Pas de cleanup. Meilleur : throw exception C++ (pas dans ce code).
2. **CUDA_CHECK_LAST() après kernel :** Doit être appelé AVANT tout autre appel CUDA (sinon erreur antérieure écrasée).
3. **CudaTimer asynchrone :** Mesure GPU time, pas CPU. Si kernel retour avant GPU finish, `toc()` bloque.
4. **cdiv avec b=0 :** Division par zéro → crash. Pas de check (assumé caller responsable).

---

## 3.2 smem_utils.cuh — Shared Memory : Padding, Load, Store

### Rôle

Fournit des **primitives pour gérer la mémoire partagée** :
1. **Padding bancaire** — éviter conflits
2. **Cooperative tile loading** — charger matrices de HBM → SMEM
3. **Cooperative tile storing** — écrire SMEM → HBM
4. **Float ↔ Half conversion** — registre helpers

C'est le **sous-système mémoire** — chaque accès SMEM fait appel à ces fonctions.

### Dépendances

- **Dépend de :** `cuda_utils.cuh`, `<cuda_fp16.h>`
- **Dépendants :** `online_softmax.cuh`, `naive_attention.cu`, `flash_attn_fwd.cu`, `flash_attn_bwd.cu`

### Walkthrough

#### Bloc 1 : SMEM_PAD_HALFS constant & padded_stride (lignes 23-27)

```c
constexpr int SMEM_PAD_HALFS = 8;  // 8 × sizeof(half) = 16 bytes

__host__ __device__ __forceinline__
constexpr int padded_stride(int cols) { return cols + SMEM_PAD_HALFS; }
```

**Concept (Chapitre 9, Section 9.2) :** Shared memory est divisée en 32 **banques**. Chaque banque = 4 bytes. Deux threads accédant même banque → **bank conflict** (sérialisation).

**Cas pathologique :** Matrice K_smem[Bc][D] avec Bc=64, D=64 (halves).
- Stride = 64 halves = 128 bytes = 32 banques exactement
- 32 threads accédent colonne j → tous la même banque → 32-way conflict (MAUVAIS)

**Solution :** Padding → stride = 64 + 8 = 72 halves = 144 bytes = 36 banques.
- 32 threads accédent colonne j → réparti sur 4 banques max (BIEN)

**Code :** `padded_stride(D)` retourne `D + 8`.

#### Bloc 2 : smem_padded_bytes (lignes 30-33)

```c
__host__ __device__ __forceinline__
constexpr int smem_padded_bytes(int rows, int cols) {
    return rows * padded_stride(cols) * sizeof(half);
}
```

Calcule **total bytes** pour matrice [rows×cols] avec padding.

**Exemple :** K_smem [64×64] en half :
```
smem_padded_bytes(64, 64) = 64 * (64 + 8) * 2 = 64 * 72 * 2 = 9216 bytes ≈ 9 KB
```

Utilisé pour calculer total SMEM needed:
```c
int smem_bytes = 2 * FWD_BC * d_padded * sizeof(half) + 32 * sizeof(float);
//             = K_smem    + V_smem    + reduce scratch
```

#### Bloc 3 : smem_load_tile (lignes 54-71)

```c
__device__ __forceinline__
void smem_load_tile(half* __restrict__ dst,
                    const half* __restrict__ src,
                    int rows, int cols,
                    int tid, int n_threads,
                    int max_row)
{
    const int stride_dst = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        half val = __float2half(0.0f);
        if (r < max_row) {
            val = src[r * cols + c];
        }
        dst[r * stride_dst + c] = val;
    }
}
```

**Fonction :** Charge tuile [rows×cols] depuis global memory (src, stride=cols) → shared memory (dst, stride=padded_stride(cols)).

**Coopératif :** Tous threads du bloc travaillent ensemble (loop step = n_threads).

**Ligne 60 : Décomposition index linéaire**
```
idx = 0..total-1 distribuée en round-robin
r = idx / cols  → row 0..rows-1
c = idx % cols  → col 0..cols-1
```

**Ligne 61-64 : Boundary handling**
- `max_row` = nombre de rows valides en source
- Si `r >= max_row` → remplir zéro (padding pour taille uniforme)
- Évite out-of-bounds reads

**Ligne 66 : Écriture SMEM avec padding**
```
dst[r * stride_dst + c]  // stride = cols + 8
```
Espace laissé entre rows dans SMEM → diffère d'espace source.

**Avantages :**
- **Coalescing :** Multiple threads chargent séquentiellement depuis HBM → transaction efficace.
- **Padding :** Élimine conflits bancaires d'accès colonne.

#### Bloc 4 : smem_load_tile_strided (lignes 76-94)

```c
__device__ __forceinline__
void smem_load_tile_strided(half* __restrict__ dst,
                            const half* __restrict__ src,
                            int rows, int cols,
                            int src_row_stride,  // <-- NOUVEAU
                            int tid, int n_threads,
                            int max_row)
{
    const int stride_dst = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        half val = __float2half(0.0f);
        if (r < max_row) {
            val = src[r * src_row_stride + c];  // <-- STRIDE CUSTOM
        }
        dst[r * stride_dst + c] = val;
    }
}
```

**Différence :** Source a stride custom (pas cols).

**Utilité :** Si data source non-contigu (e.g., V dans [B,H,N,D] où stride = H*D, pas D).

Non utilisé dans ce code, mais present pour completude.

#### Bloc 5 : smem_store_tile (lignes 99-114)

```c
__device__ __forceinline__
void smem_store_tile(half* __restrict__ dst,
                     const half* __restrict__ src,
                     int rows, int cols,
                     int tid, int n_threads,
                     int max_row)
{
    const int stride_src = padded_stride(cols);
    const int total = rows * cols;
    for (int idx = tid; idx < total; idx += n_threads) {
        int r = idx / cols;
        int c = idx % cols;
        if (r < max_row) {
            dst[r * cols + c] = src[r * stride_src + c];
        }
    }
}
```

**Inverse de load :** SMEM (padded) → HBM (compact).

Utilisé dans flash_attn_bwd pour écrire dK, dV.

#### Bloc 6 : h2f, f2h (lignes 120-125)

```c
__device__ __forceinline__
float h2f(half v) { return __half2float(v); }

__device__ __forceinline__
half f2h(float v) { return __float2half(v); }
```

Wrappers pour intrinsics CUDA. Conversions rapides (1 instruction).

**Utilité :** Code lisible. Explicite que conversion happens.

### Pont vers Théorie

**Chapitre 9, Section 9.2 : Bank conflicts.**
- Stride = D (colonne-major) → 32-way conflict
- Padding à stride = D + 8 → réparti uniformément
- Speedup : ~32× dans ce cas (32 vs 1 thread par cycle)

**Chapitre 4, Section 4.5 : FP16 vs FP32.**
- `__half2float()` = conversion loss (rounding)
- Interne : compute en FP32, input/output FP16 (économise SMEM & HBM)

### Diagrammes

**Mémoire Partagée : Sans vs Avec Padding**

```
SANS PADDING (stride = 64 halves = 128 bytes = 32 banks):
┌────────────────────────────────────────────────┐
│ Bank 0  Bank 1  ...  Bank 31                  │
├──────────────────────────────────────────────  │
│ K[0,0]  K[0,1] ... K[0,63]  (stride = 128 B)  │
│                 ↓                              │
│         32 threads access column j            │
│         → ALL hit bank j%32 → 32-way conflict │
│         (only 1 thread per cycle)             │
└─────────────────────────────────────────────  ┘

AVEC PADDING (stride = 72 halves = 144 bytes):
┌────────────────────────────────────────────────┐
│ Bank 0  Bank 1  ...  Bank 35  (+ wraps)       │
├─────────────────────────────────────────────  │
│ K[0,0]  K[0,1] ... K[0,63] [PAD]              │
│        (stride = 144 B = 4.5 banks)           │
│                 ↓                              │
│         32 threads access column j            │
│         → Distributed across 4 banks          │
│         → ~4 threads per cycle (8× speedup!)  │
└───────────────────────────────────────────────  ┘
```

### Pièges

1. **Oublier padding dans manual allocate :** Si code alloc `float smem[Bc * D]`, pas `[Bc * (D+PAD)]` → conflict.
2. **Stride mismatch :** Si load utilise stride X mais access pattern assume stride Y → silent corruption.
3. **Boundary : max_row < rows :** Remplir zéro est OK pour compute (assume padding masked), mais peut changer résultat (attention ne l'assume pas toujours).

---

## 3.3 online_softmax.cuh — Réductions et État Online Softmax

### Rôle

Implémente :
1. **Warp-level reductions** — max/sum à travers 32 threads sans SMEM
2. **Block-level reductions** — via SMEM pour coordonner warps
3. **OnlineSoftmaxState struct** — maintains m_i, l_i during forward pass

C'est le **cœur du softmax incrémental** (Chapitre 6).

### Dépendances

- **Dépend de :** `<cfloat>`, `<cmath>`
- **Dépendants :** `flash_attn_fwd.cu`, `flash_attn_bwd.cu`

### Walkthrough

#### Bloc 1 : warp_reduce_max (lignes 17-22)

```c
__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}
```

**Concept :** Butterfly reduction pattern. 32 threads → 1 max value en log2(32)=5 itérations.

**`__shfl_xor_sync(mask, val, offset)`:**
- **mask = 0xffffffff** : tous 32 lanes participent
- **offset = 16, 8, 4, 2, 1** : XOR avec lane±offset
- Exempli : lane 0 ↔ 16, lane 1 ↔ 17, etc.

**Itération 1 (offset=16):**
```
lane 0 : val = max(val[0], val[16])
lane 1 : val = max(val[1], val[17])
...
lane 31 : val = max(val[31], val[15])  // wraps around
```

Après it 1 : pairs (0,16), (1,17), ..., (15,31) contiennent max locaux.

**Itération 2-5 :** Chaque paire fusionne avec voisin → après 5 iters, toutes lanes ont même valeur (le global max).

**#pragma unroll :** Compiler hint → dérouler la boucle (limiter latence jmp).

**Retour :** Chaque lane a la même valeur (le max global de la warp).

#### Bloc 2 : warp_reduce_sum (lignes 24-30)

Identique mais `+=` lieu de `fmaxf`. Somme cumulative dans chaque lane.

#### Bloc 3 : block_reduce_max (lignes 43-61)

```c
__device__ __forceinline__
float block_reduce_max(float val, float* smem_reduce,
                       int lane, int warp_id, int n_warps_per_row)
{
    val = warp_reduce_max(val);                          // Step 1
    if (lane == 0)
        smem_reduce[warp_id] = val;
    __syncthreads();
    if (warp_id == 0 && lane < n_warps_per_row)          // Step 2
        val = smem_reduce[lane];
    else if (warp_id == 0)
        val = -FLT_MAX;
    if (warp_id == 0)
        val = warp_reduce_max(val);                      // Step 3
    // Broadcast result back via smem.
    if (warp_id == 0 && lane == 0)
        smem_reduce[0] = val;
    __syncthreads();
    return smem_reduce[0];
}
```

**Scenario :** 128 threads = 4 warps. Réduire max à travers 4 warps.

**Étape 1 :** Chaque warp réduit en interne → 4 valeurs (1 per warp).

**Étape 2 :** Warp 0 charge ces 4 valeurs depuis SMEM.

**Étape 3 :** Warp 0 réduit 4 valeurs → 1 global max.

**Broadcast :** Store dans SMEM[0] → tous threads syncthreads → lisent.

**Pourquoi 4 warps ?** Br=64, 128 threads → 2 threads per row. Mais per **KV block**, on peut avoir jusqu'à Br/32 warps par ligne (128 threads / 32 = 4 warps max).

**Retour :** Tous threads ont smem_reduce[0] = global block max.

#### Bloc 4 : block_reduce_sum (lignes 64-82)

Identique mais avec `+` et initialiser `-FLT_MAX` → `0.0f`.

#### Bloc 5 : OnlineSoftmaxState struct (lignes 95-127)

```c
struct OnlineSoftmaxState {
    float m;   // running max
    float l;   // running sum of exp(x - m)

    __device__ __forceinline__
    OnlineSoftmaxState() : m(-FLT_MAX), l(0.0f) {}

    __device__ __forceinline__
    float update(float new_max, float new_sum) {
        float m_new   = fmaxf(m, new_max);
        float alpha   = __expf(m - m_new);
        float beta    = __expf(new_max - m_new);
        l = alpha * l + beta * new_sum;
        m = m_new;
        return alpha;
    }

    __device__ __forceinline__
    float logsumexp() const {
        return m + logf(l);
    }
};
```

**Concept (Chapitre 6, Section 6.3) :** Softmax online.

À chaque bloc KV, on reçoit :
- `new_max` = max des scores S_new
- `new_sum` = sum(exp(S_new - new_max))

On accumule :
- `m` = global max vu jusqu'ici
- `l` = sum(exp(x - m)) pour tous x vu

**Formule update :**
```
m_new = max(m, new_max)
alpha = exp(m - m_new)          // rescale old accumulation
beta  = exp(new_max - m_new)    // scale new values
l = alpha * l + beta * new_sum  // merge accumulators
```

**Pourquoi stable :**
- Exposant `m - m_new` ≤ 0 → exp() < 1 (no overflow)
- Exposant `new_max - m_new` ≤ 0 → exp() < 1 (no overflow)
- Tout reste en [0, inf) (no underflow catastrophique)

**logsumexp() :**
```
Final result = exp(m + log(l)) = exp(m) * l
                                = (sum of unnormalized probs)
Softmax = exp(x - m) / l (already normalized by update())
```

Pour backward : logsumexp = log(sum(exp(x))) = log-partition function.

### Pont vers Théorie

**Chapitre 6, Section 6.3 :** Online softmax évite stocker N×N matrice.

**Chapitre 9, Section 9.3 :** Warp shuffles = primitive synchronisation.

**Chapitre 8, Section 8.2 :** Flash Attention utilise OnlineSoftmaxState pour accumuler O incrémentalement.

### Diagramme : Butterfly Reduction

```
Offset = 16 :
Lane:   0   1   2 ... 15  16  17 ... 31
Val in: [a0, a1, a2, ... a15, a16, a17, ... a31]
        │                  │
        └─── max? ───┐     │
                     └─────┘
Val out:[max(a0,a16), max(a1,a17), ..., max(a15,a31), ...]

Offset = 8 :
Lane:   0    1    2 ... 7    8    9 ... 15 ...
Val in: [m0,  m1,  m2, ... m7, m8, m9, ... m15, ...]
        │           │
        └─ max? ────┘
Val out:[max(m0,m8), ..., max(m7,m15), ...]

...itérer jusqu'offset=1, toutes lanes = global max
```

### Pièges

1. **Lane ordering :** `__shfl_xor_sync` XOR de lane ID. Lane 0 parle à lane 16 (0 XOR 16 = 16), pas lane 1.
2. **Masquer tous les lanes :** `0xffffffff` assume 32-lane warp. Si warp < 32 (rare), masque doit refléter vraiment actives lanes.
3. **smem_reduce buffer :** Caller doit allouer suffisant (≥ n_warps_per_row floats). Pas de check.

---

## 3.4 flash_attn_fwd.cu — Forward Pass : Le Cœur

### Rôle

Implémente le **kernel principal Flash Attention forward**.

Computes : O = softmax(Q K^T / sqrt(d)) V, mais :
- **Tuile** Q,KV en blocs
- **Streaming** K,V depuis HBM
- **Online softmax** pour accumuler O sans materialiser N×N
- **Zwei GEMMs** : scores et output

**Ceci est l'algorithme révolutionnaire du Chapitre 8.**

### Dépendances

- **Dépend de :** `cuda_utils.cuh`, `smem_utils.cuh`, `online_softmax.cuh`
- **Dépendants :** `torch_bindings.cpp`, tests, benchmarks

### Walkthrough Ligne-à-Ligne

#### Bloc 1 : Configuration Constants (lignes 44-54)

```c
static constexpr int FWD_BR = 64;           // Q block size (rows)
static constexpr int FWD_BC = 64;           // KV block size (rows)
static constexpr int FWD_NUM_THREADS = 128; // threads per block

static constexpr int ROWS_PER_THREAD = FWD_BR / FWD_NUM_THREADS;
static constexpr int RPT = (ROWS_PER_THREAD > 0) ? ROWS_PER_THREAD : 1;
```

**Tuning parameters :**
- **Br=64 :** Nombre de rows Q traitées par bloc. Plus grand → moins de blocs, moins overhead. Limite : SMEM pour K_smem, V_smem.
- **Bc=64 :** Nombre de rows KV loaded en SMEM. Matching Br pour symétrie (peut varier).
- **NUM_THREADS=128 :** Threads par bloc. Choix standard : 4 warps = bonne occupancy.

**ROWS_PER_THREAD :**
```
= 64 / 128 = 0 (! → clamp à 1)
```
Donc chaque thread "owns" ≤1 row. Avec 128 threads mais 64 rows → chaque thread responsable d'1 row, donc 2 threads par row (threads 0-63 pour row 0-63, threads 64-127 aussi pour rows 0-63).

#### Bloc 2 : Kernel Signature (lignes 61-71)

```c
__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    float*      __restrict__ L,    // logsumexp
    const int N,
    const int D,
    const float scale,              // 1 / sqrt(d)
    const bool causal)
```

**Entrées :**
- Q, K, V : [B*H, N, D] en FP16
- scale : precompute 1/sqrt(D) (const pour tous)
- causal : bool pour masque causal

**Sorties :**
- O : [B*H, N, D] attention output
- L : [B*H, N] logsumexp sauvegardé pour backward

#### Bloc 3 : Thread Mapping & Global Indexing (lignes 72-100)

```c
const int bh         = blockIdx.y;          // which (B, H) pair
const int block_q    = blockIdx.x;          // which Q block (0..cdiv(N,Br)-1)
const int q_start    = block_q * FWD_BR;    // global row offset
const int tid        = threadIdx.x;         // thread 0..127

const half* Q_bh = Q + (size_t)bh * N * D;  // pointer to (B,H) slice
const half* K_bh = K + (size_t)bh * N * D;
const half* V_bh = V + (size_t)bh * N * D;
half*       O_bh = O + (size_t)bh * N * D;
float*      L_bh = L + (size_t)bh * N;

// ── Shared memory layout ──────────────────────
extern __shared__ char smem_raw[];
const int d_padded = padded_stride(D);
half* K_smem = reinterpret_cast<half*>(smem_raw);
half* V_smem = K_smem + FWD_BC * d_padded;
float* reduce_smem = reinterpret_cast<float*>(V_smem + FWD_BC * d_padded);

// ── Thread → Q row mapping ───────────────────
const int my_q_row_local = tid % FWD_BR;            // 0..63
const int my_q_row_global = q_start + my_q_row_local;
const int threads_per_row = FWD_NUM_THREADS / FWD_BR;  // 128/64 = 2
const int d_tid = tid / FWD_BR;                        // 0 or 1

const int d_per_thread = D / threads_per_row;        // D/2 elements
const int d_offset = d_tid * d_per_thread;            // start column
```

**Grid structure :**
- blockIdx.x ∈ [0, cdiv(N, 64)) : Q blocks
- blockIdx.y ∈ [0, B*H) : batch×head indices

**Pointers :** Décalés par `bh * N * D` pour isoler la paire (B,H).

**SMEM layout :**
```
[    K_smem [64×68]    ]
[    V_smem [64×68]    ]
[ reduce_smem[32]      ]
```
Total ≈ 9KB + 9KB + 128B ≈ 18.3 KB.

**Thread mapping :**
- tid=0..63 : responsible pour Q row 0..63 (each row has 2 partner threads)
- tid=64..127 : responsible pour Q row 0..63 (partner 2)
- d_tid=0 : colonnes D/2...D-1 ? NON — colonnes 0...D/2-1 (d_tid=tid/64 is 0 for tid 0..63, 1 for tid 64..127)

**Exempli :** D=64, threads_per_row=2.
```
tid=0 : Q row 0, columns 0..31
tid=1 : Q row 1, columns 0..31
...
tid=63 : Q row 63, columns 0..31
tid=64 : Q row 0, columns 32..63  ← PARTNER OF tid=0
tid=65 : Q row 1, columns 32..63  ← PARTNER OF tid=1
...
```

#### Bloc 4 : Register Allocation (lignes 107-110)

```c
float q_reg[128];   // max D = 128
float o_acc[128];   // accumulator for output
float s_row[64];    // max Bc = 64 — score row
```

**q_reg :** Stocke chunk de Q row (un thread owns une fraction de la ligne).

**o_acc :** Accumule O_row au fil des blocs KV.

**s_row :** Temporaire pour scores une bloc KV. Réutilisé à chaque KV iteration.

#### Bloc 5 : Load Q into Registers (lignes 113-119)

```c
for (int i = 0; i < d_per_thread; ++i) {
    int d_idx = d_offset + i;
    if (my_q_row_global < N && d_idx < D)
        q_reg[i] = __half2float(Q_bh[my_q_row_global * D + d_idx]);
    else
        q_reg[i] = 0.0f;
}
```

**Chaque thread :**
- Adresse : Q_bh[row * D + col]
- Conversion FP16 → FP32 inline
- Boundary check : si row ou col out of bounds → zéro

**Résultat :** q_reg[0..d_per_thread-1] = Q fragment.

#### Bloc 6 : Initialize Accumulators (lignes 122-127)

```c
for (int i = 0; i < d_per_thread; ++i)
    o_acc[i] = 0.0f;

float m_i = -FLT_MAX;
float l_i = 0.0f;
```

O accumulator initialized to zero.
OnlineSoftmaxState: m_i = -inf, l_i = 0.

#### Bloc 7 : KV Block Loop Setup (lignes 130-137)

```c
const int num_kv_blocks = cdiv(N, FWD_BC);
int kv_end = num_kv_blocks;
if (causal) {
    // Causal masking: skip KV blocks entirely to the right.
    kv_end = cdiv(q_start + FWD_BR, FWD_BC);
    if (kv_end > num_kv_blocks) kv_end = num_kv_blocks;
}
```

**num_kv_blocks :** Combien de chunks KV.

**kv_end avec causal :** Si masque causal actif, Q block [q_start, q_start+Br) ne peut consulter KV beyond column q_start+Br. Donc kv_end = cdiv(q_start + Br, Bc).

#### Bloc 8 : Main KV Loop (lignes 139-214)

```c
for (int kv_block = 0; kv_block < kv_end; ++kv_block) {
    const int kv_start = kv_block * FWD_BC;
    const int kv_valid = min(FWD_BC, N - kv_start);
```

Pour chaque bloc KV dans la séquence.

**kv_valid :** Si dernier bloc incomplet (N non-multiple de Bc), marquer valid rows. Rows beyond kv_valid sont padding.

#### Bloc 8.1 : Load K, V into SMEM (lignes 144-148)

```c
smem_load_tile(K_smem, K_bh + kv_start * D,
               FWD_BC, D, tid, FWD_NUM_THREADS, kv_valid);
smem_load_tile(V_smem, V_bh + kv_start * D,
               FWD_BC, D, tid, FWD_NUM_THREADS, kv_valid);
__syncthreads();
```

Tous threads **coopérativement** chargent [64×64] tile de K et V depuis HBM → SMEM. Puis synchronize.

#### Bloc 8.2 : GEMM-I : Score Computation (lignes 150-177)

```c
for (int j = 0; j < FWD_BC; ++j) {
    float dot = 0.0f;
    for (int i = 0; i < d_per_thread; ++i) {
        int d_idx = d_offset + i;
        dot += q_reg[i] * __half2float(K_smem[j * d_padded + d_idx]);
    }
    // Reduce across partner threads sharing this Q row.
    if (threads_per_row == 2) {
        dot += __shfl_xor_sync(0xffffffff, dot, FWD_BR);
    }
    dot *= scale;

    // Causal mask.
    if (causal) {
        int global_j = kv_start + j;
        if (global_j > my_q_row_global)
            dot = -1e20f;
    }

    // Mask padding.
    if (j >= kv_valid)
        dot = -1e20f;

    s_row[j] = dot;
}
```

**Pour chaque KV row j :**

1. Chaque thread compute partial dot product q_reg · K_smem[j][thread_cols]
2. Shuffle_xor reduce avec partner thread (if 2 per row) → full dot
3. Multiply par scale = 1/sqrt(D)
4. Appliquer masques (causal et padding)
5. Store dans s_row[j]

**Result :** s_row = [score_{row,0}, score_{row,1}, ..., score_{row,63}] pour cette KV block.

#### Bloc 8.3 : Online Softmax Update (lignes 179-200)

```c
float m_block = -FLT_MAX;
for (int j = 0; j < FWD_BC; ++j)
    m_block = fmaxf(m_block, s_row[j]);

float m_new = fmaxf(m_i, m_block);
float alpha = __expf(m_i - m_new);
float row_sum = 0.0f;
for (int j = 0; j < FWD_BC; ++j) {
    s_row[j] = __expf(s_row[j] - m_new);
    row_sum += s_row[j];
}

l_i = alpha * l_i + row_sum;
m_i = m_new;

for (int i = 0; i < d_per_thread; ++i)
    o_acc[i] *= alpha;
```

**Étapes :**

1. **Find block max :** m_block = max(s_row)
2. **Merge with running max :** m_new = max(m_i, m_block)
3. **Rescale old accumulation :** alpha = exp(m_i - m_new), o_acc *= alpha
4. **Exponentiate scores :** s_row[j] = exp(s_row[j] - m_new) (stable)
5. **Sum exponentials :** row_sum = sum(s_row)
6. **Update running sum :** l_i = alpha * l_i + row_sum

**Net effect :** o_acc maintenant rescalé, prêt pour nouvelle contribution.

#### Bloc 8.4 : GEMM-II : Accumulate Output (lignes 202-211)

```c
for (int j = 0; j < FWD_BC; ++j) {
    float p = s_row[j];
    for (int i = 0; i < d_per_thread; ++i) {
        int d_idx = d_offset + i;
        o_acc[i] += p * __half2float(V_smem[j * d_padded + d_idx]);
    }
}

__syncthreads();
```

**Pour chaque KV row j :**
- p = softmax score (déjà exp et summed)
- o_acc += p * V[j]

**Result :** o_acc ← accumulated output fragment.

#### Bloc 9 : Finalization & Writeback (lignes 217-229)

```c
if (my_q_row_global < N) {
    float inv_l = 1.0f / (l_i + 1e-8f);
    for (int i = 0; i < d_per_thread; ++i) {
        int d_idx = d_offset + i;
        if (d_idx < D) {
            O_bh[my_q_row_global * D + d_idx] = __float2half(o_acc[i] * inv_l);
        }
    }
    // Only one of the partner threads writes L.
    if (d_tid == 0) {
        L_bh[my_q_row_global] = m_i + logf(l_i + 1e-8f);
    }
}
```

**Normalize :** o_acc / l_i = O row (final softmax normalization).

**Convert :** O_acc from FP32 back to FP16 via __float2half.

**Logsumexp :** l = m_i + log(l_i) sauvegardé pour backward. Uniquement d_tid==0 writes (avoid duplication).

#### Bloc 10 : Host Launcher (lignes 236-264)

```c
void flash_attn_fwd(
    const half* Q, const half* K, const half* V,
    half* O, float* L,
    int B, int H, int N, int D,
    bool causal,
    cudaStream_t stream)
{
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int d_padded = padded_stride(D);
    int smem_bytes = 2 * FWD_BC * d_padded * sizeof(half)
                   + 32 * sizeof(float);

    set_smem_limit(flash_attn_fwd_kernel, smem_bytes);

    dim3 grid(cdiv(N, FWD_BR), BH);
    dim3 block(FWD_NUM_THREADS);

    flash_attn_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, L, N, D, scale, causal);

    CUDA_CHECK_LAST();
}
```

**Préparation CPU :**
- Compute scale
- Compute SMEM size (K+V+reduce scratch)
- Set CUDA attribute pour dynamique SMEM
- Launch grid

**Grid :** (cdiv(N, Br), BH) = (cdiv(N, 64), B*H).

### Pont vers Théorie

**Chapitre 8, Section 8.2-8.4 :** L'algorithme exact Flash Attention.
- Tiling : q blocks itérés, kv blocks streamed
- Online softmax : accumule sans materialiser N×N
- Mémoire : O(N*D + Br*D + Bc*D) au lieu O(N²)

**Chapitre 9, Sections 9.2-9.3 :**
- SMEM padding : conflits bancaires
- Warp shuffle : communication efficace

**Chapitre 4, Section 4.5 :** Conversions FP16 ↔ FP32.

### Diagrammes

**Boucle KV : Data Flow**

```
Iteration KV block 0 (rows 0..63):
  ├─ Load K[0:64, :] → K_smem
  ├─ Load V[0:64, :] → V_smem
  ├─ For Q row my_row :
  │   ├─ dot = Q[my_row] · K[0:64]^T  → s_row[0:64]
  │   ├─ exp(s_row - max) → softmax scores
  │   ├─ o_acc += scores · V[0:64, :]
  │   └─ update m_i, l_i
  └─ Result: o_acc partially accumulated

Iteration KV block 1 (rows 64..127):
  ├─ Load K[64:128, :] → K_smem (overwrite)
  ├─ Load V[64:128, :] → V_smem
  ├─ For Q row my_row :
  │   ├─ Rescale o_acc by alpha (due to new max)
  │   ├─ dot = Q[my_row] · K[64:128]^T → s_row
  │   ├─ Accumulate to o_acc
  │   └─ Update m_i, l_i
  └─ o_acc further accumulated

...final iteration:
  o_acc = normalized(o_acc / l_i) → Final O
  L = m_i + log(l_i) → for backward
```

### Pièges

1. **Double counting avec 2 threads per row :** Si partner threads tous deux écrire dQ/dV → duplication. Code marque `d_tid == 0` pour L, mais attention à future modifications.
2. **Boundary handling :** Si N non-multiple Br/Bc → last blocks partiels. kv_valid et checks gèrent zéros, mais logique fragile.
3. **Causal masking logic :** `if (global_j > my_q_row_global)` applies -inf. Correct, mais si causal mal comprises, silent bug.
4. **FP32 vs FP16 latency :** Interne compute FP32 → lent que Tensor Cores (FP16). Trade-off accuracy vs speed.

---

## 3.5 flash_attn_bwd.cu — Backward Pass

Trop volumineux pour walkthrough complet, mais structure clé :

### Rôle

Deux phases :
1. **compute_D_kernel :** D_i = sum_d(dO[i,d] * O[i,d])
2. **flash_attn_bwd_kernel :** Outer loop KV, inner loop Q. Recomputes S, P. Accumulates dK, dV, atomicAdd dQ.

### Structure

```
compute_D_kernel:
  Grid: (cdiv(N, 256), B*H)
  Pour chaque Q row : dot(dO_row, O_row)

flash_attn_bwd_kernel:
  Grid: (cdiv(N, Bc), B*H)  ← one block per KV block
  Outer loop: KV rows (load K,V in registers once)
  Inner loop: Q blocks (load Q, dO in smem)
    For each Q row: recompute S, P, accumulate dV, dK, atomicAdd dQ

convert_kernel (lambda):
  Chaque élément: dQ_f[idx] → dQ_half[idx]
```

### Pont vers Théorie

**Chapitre 8, Section 8.5 :** Flash Attention backward. Recomputation vs stockage.

---

## 3.6 torch_bindings.cpp — PyTorch Integration

### Rôle

Wrapper C++ qui expose kernels CUDA en tant qu'extension PyTorch.

### Structure clé

```cpp
#define CHECK_INPUT(x)  // Validate tensor

static half* ptr(torch::Tensor& t)         // non-const cast
static const half* cptr(const torch::Tensor& t)  // const cast

torch::Tensor naive_fwd(...)               // Retourne O
std::vector<torch::Tensor> flash_fwd(...)  // Retourne (O, L)
std::vector<torch::Tensor> flash_bwd(...)  // Retourne (dQ, dK, dV)

PYBIND11_MODULE(flash_attn_cuda, m) { ... }  // Python binding
```

### Walkthrough

**Helpers :**
```cpp
static half* ptr(torch::Tensor& t) {
    return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}
```

Acquires raw GPU pointer from PyTorch tensor.

**naive_fwd :**
```cpp
torch::Tensor naive_fwd(torch::Tensor Q, torch::Tensor K,
                        torch::Tensor V, bool causal) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    const int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto O = torch::zeros_like(Q);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    naive_attention_fwd(cptr(Q), cptr(K), cptr(V), ptr(O),
                        B, H, N, D, causal, stream);
    return O;
}
```

Valide inputs, crée output tensor, lance kernel, retourne tensor.

**flash_fwd :**
```cpp
std::vector<torch::Tensor> flash_fwd(...) {
    // ... same validation ...
    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash_attn_fwd(cptr(Q), cptr(K), cptr(V), ptr(O),
                   L.data_ptr<float>(), B, H, N, D, causal, stream);
    return {O, L};
}
```

Retourne 2 tensors : O (FP16), L (FP32).

---

## 3.7 CMakeLists.txt & setup.py — Build System

### CMakeLists.txt

**Role :** Compile .cu files → libflash_attn_kernels.a (static lib).

**Key directives :**
- `cmake_minimum_required(VERSION 3.18)` — CUDA support
- `set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")` — Multi-GPU support
- `add_library(flash_attn_kernels STATIC ...)` — Collect kernels
- `CUDA_SEPARABLE_COMPILATION ON` — Allow calling device code from host

### setup.py

**Role :** Wrapper for PyTorch setuptools.

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

extension = CUDAExtension(
    name='flash_attn_cuda',
    sources=[...],  # .cu, .cpp files
    include_dirs=[...],
    extra_compile_args=[...],
)

setup(
    name='flash_attn',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## 3.8 Tests & Benchmarks

### test_correctness.py

```python
def ref_attention(Q, K, V, causal=False):
    """Gold reference in float32."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(...), diagonal=1)
        S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V.float())
    return O

def test_flash_fwd(B, H, N, D, causal):
    Q, K, V = create_random_inputs(...)
    O_ref = ref_attention(Q, K, V, causal).half()
    O_flash, L = flash_attn_cuda.flash_fwd(Q, K, V, causal)

    max_err = (O_flash.float() - O_ref.float()).abs().max().item()
    rel_err = max_err / (O_ref.float().abs().max() + 1e-5)
    assert rel_err < 2e-2, f"Error too high: {rel_err}"
```

Compare kernel output vs reference (float32 torch.matmul + softmax).

### bench_all.py

```python
def bench_fn(fn, warmup=10, iters=100):
    """Benchmark CUDA function."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def compute_metrics(B, H, N, D, latency_ms):
    """Compute TFLOPS, bandwidth."""
    flops_fwd = 4.0 * B * H * N * N * D
    tflops = flops_fwd / (latency_ms * 1e-3) / 1e12
    bytes_accessed = B * H * (4 * N * D + N) * 2
    bw_gb_s = bytes_accessed / (latency_ms * 1e-3) / 1e9
    return tflops, bw_gb_s
```

Tests 5 impls : naive, flash CUDA, SDPA, Triton, Dao (if installed).

---

## 3.9 flash_attn_triton.py — Triton Implementation

### Structure

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        ...
    ],
    key=["N", "D"],
)
@triton.jit
def _flash_attn_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, ...):
    block_q_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    # Load Q block
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Accumulators
    o_acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_Q], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # KV loop
    for kv_idx in range(0, kv_blocks):
        # Load K, V
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # GEMM-I: S = Q @ K^T
        s = tl.dot(q, tl.trans(k)) * scale

        # Masking
        s = tl.where(kv_offsets[None, :] < N, s, float("-inf"))
        if CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online softmax
        m_block = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # GEMM-II: O += P @ V
        o_acc = alpha[:, None] * o_acc + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Normalize and store
    o = o_acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < N)
    tl.store(l_ptrs, lse, mask=q_offsets < N)
```

**Avantage Triton :**
- Pas de SMEM management explicite (Triton le fait)
- Pas de warp-level sync (Triton compile il)
- Autotuning : essayer 4 configs, choisir le meilleur

**Inconvénient :**
- ~10-20% plus lent que CUDA optimisé (Triton overhead)
- Moins contrôle bas-niveau

---

## 3.10 Reste de PARTIE 3 : Analyse Synthétique des Fichiers Restants

### 3.10.1 naive_attention.cu

**Trois kernels simples, aucun SMEM, aucun optimisation.**

1. **naive_qk_kernel :** Chaque thread compute un élément S[i,j] = Q[i]·K[j]. Grille (cdiv(N,16), cdiv(N,16), BH). Materialise S en FP32.

2. **naive_softmax_kernel :** Une ligne S par block. Trois passes : max, exp-sum, normalize. Shared memory pour reductions.

3. **naive_pv_kernel :** O[i,d] = sum_j(P[i,j] * V[j,d]). Grille (cdiv(N,16), cdiv(D,16), BH).

**Mémoire :** O(N²) pour S, P → pas pour production (limité à N ≤ 1024). Mais c'est **correctness baseline**.

### 3.10.2 flash_attn_bwd.cu

**Deux phases logiques :**

**Phase 1 : compute_D_kernel**
```cpp
Grid: (cdiv(N, 256), B*H)
For row i:
  D[i] = sum_d(dO[i,d] * O[i,d])
```
Precompute terme de scaling pour dS = P * (dP - D_i).

**Phase 2 : flash_attn_bwd_kernel**

Boucle **inverse** vs forward :
- **Outer :** KV blocks (accumulate dK, dV per-block)
- **Inner :** Q blocks (load Q, dO; atomicAdd dQ)

```
For each KV row:
  Load K, V in registers
  For each Q block:
    Load Q, dO in SMEM
    For each Q row:
      Recompute S = Q @ K^T
      Recompute P = exp(S - L)  ← uses saved L
      dP = dO @ V^T
      dS = P * (dP - D_i)
      dV += P * dO^T
      dK += dS * Q^T
      atomicAdd(&dQ, dS * K)  ← scattered writes
```

**Atomics :** Plusieurs Q blocks peuvent écrire même dQ[row]. atomicAdd coordonne.

**Phase 3 : lambda convert**
```cpp
dQ_f (float32) → dQ (float16) via kernel lambda
```

### 3.10.3 torch_bindings.cpp

Expose :
- `naive_fwd(Q, K, V, causal) → O`
- `flash_fwd(Q, K, V, causal) → (O, L)`
- `flash_bwd(Q, K, V, O, dO, L, causal) → (dQ, dK, dV)`

Checks :
- `CHECK_CUDA` : tenseur doit être CUDA
- `CHECK_CONTIGUOUS` : row-major layout
- `CHECK_FP16` : float16 pour Q,K,V,O,dQ,dK,dV
- L must be FP32

PYBIND11_MODULE : expose en tant qu'extension Python.

### 3.10.4 Tests & Benchmarks Summary

**test_correctness.py :** Compares naive, flash vs ref_attention (float32). Acceptable si rel_err < 2%.

**test_edge_cases.py :** Non-aligned N, très court N, D=128, etc.

**bench_all.py :** 5 impls. Mesure latence (ms), TFLOPS, bandwidth.

**plot_results.py :** Matplotlib traces (latency vs N, TFLOPS vs D, bars de comparaison).

---

# PARTIE 4 : Guide Opérationnel

## 4.1 Installation et Prérequis

### 4.1.1 Prérequis Système

```bash
# 1. NVIDIA GPU (Compute Capability >= 7.0 recommandé)
#    Testé : A100 (8.0), A10 (8.6), V100 (7.0)

# 2. CUDA Toolkit >= 11.0 (recommandé 11.8 ou 12.x)
nvidia-smi          # Vérifier GPU disponible
nvcc --version      # Vérifier CUDA installé

# 3. cuDNN (optionnel, pour comparaison vs PyTorch SDPA)
# 4. CMake >= 3.18
cmake --version

# 5. Python 3.8+ with PyTorch
python --version
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.1.2 Installation Step-by-Step

**Étape 1 : Cloner/naviguer au répertoire**
```bash
cd /path/to/01-flash-attention
```

**Étape 2 : Installer dépendances Python**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytest matplotlib numpy scipy
pip install triton  # Pour implémentation Triton
```

**Étape 3 : Build et install extension CUDA**
```bash
make install
# Exécute : cd cuda && pip install -e . --no-build-isolation
# Compile tous les .cu files via CMake
# Crée flash_attn_cuda.*.so (shared library)
```

**Étape 4 : Vérifier installation**
```bash
python -c "import flash_attn_cuda; print('Success!')"
```

Si erreur, exécuter `make clean` et réessayer.

### 4.1.3 Troubleshooting Installation

| Erreur | Cause | Solution |
|--------|-------|----------|
| `CUDA Toolkit not found` | CUDA pas en PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| `nvcc fatal : unknown code form` | Architectures GPU bad | `cmake -DCMAKE_CUDA_ARCHITECTURES=80` (forcer A100) |
| `CUDA out of memory` | Build consomme trop GPU | Ajouter `-DCMAKE_CUDA_SEPARABLE_COMPILATION=OFF` |
| `Permission denied` | Pip install sans -e | Utiliser virtual env : `python -m venv venv && source venv/bin/activate` |

---

## 4.2 Makefile Targets — Commandes Disponibles

### 4.2.1 make install

```bash
make install
```

**Fait :** Build + installe extension PyTorch.

**Exécution :**
```
cd cuda && pip install -e . --no-build-isolation
  → Exécute setup.py
  → cmake CMakeLists.txt
  → Compile .cu files avec nvcc
  → Link avec libcudart.so
  → Crée .so shared library
  → pip editable install (dev mode)
```

**Résultat :** `import flash_attn_cuda` disponible dans Python.

**Durée :** ~30-60 secondes première fois. Incrémental ensuite.

### 4.2.2 make build

```bash
make build
```

**Fait :** CMake standalone build (sans PyTorch).

**Utilité :** Compile kernels en static library. Utile pour tests C++ purs (rarement).

**Résultat :** `cuda/build/libflash_attn_kernels.a`

### 4.2.3 make test

```bash
make test
```

**Fait :** Exécute pytest sur CUDA kernels.

**Tests :**
- `test_naive_fwd` : Naive vs reference
- `test_flash_fwd` : Flash vs reference
- `test_flash_bwd` : Backward gradients
- `test_flash_vs_naive` : Consistency check

**Paramètres :** (B, H, N, D) ∈ {small, medium, large}, causal={True, False}.

**Passage :** rel_err < 2% (acceptable in FP16).

**Durée :** ~30 secondes.

```bash
# Run spécifique test:
python -m pytest cuda/tests/test_correctness.py::test_flash_fwd -v -k "N=256"
```

### 4.2.4 make test-triton

```bash
make test-triton
```

Analogue pour implémentation Triton.

```bash
make test-all  # both
```

### 4.2.5 make bench

```bash
make bench
```

**Fait :** Benchmark 5 implémentations.

**Implémentations :**
1. Naive (CUDA)
2. Flash (CUDA custom)
3. PyTorch SDPA
4. Triton custom
5. Tri Dao flash-attn (if `pip install flash-attn`)

**Configurations :** N ∈ {256, 512, 1024, 2048, 4096}, D ∈ {64, 128}.

**Output :** JSON files → `benchmarks/results/*.json` (latency ms, TFLOPS, bandwidth).

**Durée :** ~5-10 minutes (100 iters per config, warmup).

```bash
# Benchmark Triton only:
make bench-triton
```

### 4.2.6 make plots

```bash
make plots
```

**Fait :** Matplotlib visualisations des résultats.

**Plots :**
- Latency vs N (line plot, 5 impls)
- TFLOPS vs N
- Bandwidth utilisation
- Bar chart comparaison

**Output :** PDF/PNG → `benchmarks/plots/`

### 4.2.7 make clean

```bash
make clean
```

**Efface :**
- Build artifacts (CMake)
- `.egg-info`, `.so` shared libraries
- `__pycache__`

**Utilité :** Avant réinstaller (si CMake config change).

### 4.2.8 make help

```bash
make help
```

Affiche usage de tous targets.

---

## 4.3 Debugging et Outils

### 4.3.1 NVIDIA Nsight Compute

**Outil :** Line-by-line GPU profiling.

**Installation :**
```bash
# Inclus avec CUDA Toolkit
which ncu  # ou : /usr/local/cuda/bin/ncu
```

**Usage :**

```bash
# Profile un kernel (example : avec torch script)
cat > profile_flash.py << 'EOF'
import torch
import flash_attn_cuda

B, H, N, D = 1, 1, 512, 64
Q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# Warmup
for _ in range(10):
    flash_attn_cuda.flash_fwd(Q, K, V, causal=False)

torch.cuda.synchronize()
print("Ready to profile")
EOF

ncu -o result.ncu python profile_flash.py
ncu-ui result.ncu  # Ouvre GUI
```

**Métriques clés :**
- **L1 Cache Hit Ratio :** ~90-95% (bon)
- **L2 Cache Hit Ratio :** ~50-70% (acceptable)
- **Memory Throughput :** Vs peak (A100 2 TB/s)
- **SMEM Bank Conflicts :** Doit être 0 (grace à padding)
- **Warp Occupancy :** ~80-90% (bon)

**Pour déboguer :**
```bash
# Résumé rapide
ncu --set full profile_flash.py

# Détails intensité arithmétique
ncu --set sm_memory_utilization profile_flash.py
```

### 4.3.2 NVIDIA cuda-gdb

**Outil :** Débogue kernels ligne-par-ligne.

**Setup :**
```bash
# Compiler avec debug symbols
export CUDAFLAGS="-g -G"  # GPU debug
make clean && make install
```

**Usage :**
```bash
cat > simple_test.py << 'EOF'
import torch
import flash_attn_cuda
Q = torch.randn(1, 1, 16, 64, dtype=torch.float16, device='cuda')
K = torch.randn_like(Q)
V = torch.randn_like(Q)
O, L = flash_attn_cuda.flash_fwd(Q, K, V, causal=False)
print(O)
EOF

cuda-gdb --args python simple_test.py
# Dans gdb:
# (cuda-gdb) break flash_attn_fwd_kernel  # Breakpoint par nom kernel
# (cuda-gdb) run
# (cuda-gdb) continue
# (cuda-gdb) thread 0.1.0  # Select thread
# (cuda-gdb) info locals   # Voir variables
```

### 4.3.3 Common Errors & Solutions

| Erreur | Cause | Fix |
|--------|-------|-----|
| `CUDA error at line X: invalid argument` | Bad grille/block size | Vérifier `cdiv()` retourne > 0 |
| `CUDA error: out of memory` | SMEM trop grand | Réduire FWD_BR, FWD_BC dans code |
| `Assertion 'blockDim.x == FWD_NUM_THREADS' failed` | Thread count mismatch | Vérifier `set_smem_limit` appelé avec bon kernel |
| `relative error too high` | Numerics instability | Changer scale (FP16 precision limite) |
| `kernel missing` | .cu not compiled | `make clean`, `make install` |
| `ImportError: flash_attn_cuda` | Extension not built | `make install` |

### 4.3.4 Print Debugging dans Kernels

**Technique :** Ajouter `printf` pour debugging.

```cuda
__global__ void my_kernel(...) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug: N=%d, D=%d\n", N, D);
    }
    __syncthreads();
}
```

**Limitation :** Printf asynchrone, peut être perdu. Ajouter `cudaDeviceSynchronize()` après kernel.

**Meilleur :** Utiliser `cuda-gdb` (moins invasif).

---

## 4.4 Contribution Guide : Exemple — Ajouter Support GQA

**GQA (Grouped Query Attention) :** K, V ont fewer heads que Q. Optimisation mémoire.

### Étape 1 : Modifier la signature du kernel

```cuda
// Avant:
__global__ void flash_attn_fwd_kernel(
    const half* Q,  // [B*H_q, N, D]
    const half* K,  // [B*H_k, N, D]  ← H_k < H_q
    const half* V,  // [B*H_k, N, D]
    ...
    int N, int D,
    ...
)

// Ajouter paramètre:
    int H_q_per_group,  // combien Q heads per KV head (e.g. 4)
```

### Étape 2 : Ajuster pointeur BH

```cuda
const int bh_q = blockIdx.y;
const int bh_k = bh_q / H_q_per_group;

const half* Q_bh = Q + (size_t)bh_q * N * D;
const half* K_bh = K + (size_t)bh_k * N * D;  // shared across groups
const half* V_bh = V + (size_t)bh_k * N * D;
```

### Étape 3 : Modifier torch_bindings.cpp

```cpp
std::vector<torch::Tensor> flash_fwd_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int H_q_per_group, bool causal)
{
    int B = Q.size(0), H_q = Q.size(1), H_k = K.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(H_q % H_k == 0, "H_q must be multiple of H_k");
    TORCH_CHECK(H_q / H_k == H_q_per_group, "H_q_per_group mismatch");

    int BH_q = B * H_q;
    dim3 grid(cdiv(N, FWD_BR), BH_q);
    // ... launcher avec H_q_per_group ...
}

PYBIND11_MODULE(...) {
    m.def("flash_fwd_gqa", &flash_fwd_gqa);
}
```

### Étape 4 : Test

```python
def test_gqa():
    B, H_q, H_k, N, D = 2, 8, 2, 256, 64
    Q = torch.randn(B, H_q, N, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H_k, N, D, dtype=torch.float16, device='cuda')
    V = torch.randn_like(K)

    O, L = flash_attn_cuda.flash_fwd_gqa(Q, K, V, H_q_per_group=4, causal=False)
    assert O.shape == (B, H_q, N, D)

    # Compare vs reference (torch avec broadcasting)
    ...

test_gqa()
```

---

## 4.5 Pattern : Ajouter Nouveau Test

### Étape 1 : Créer fichier test

```python
# cuda/tests/test_my_feature.py
import pytest
import torch
import flash_attn_cuda

def ref_my_feature(Q, K, V, ...):
    """Reference implementation in float32."""
    # Implémentation de référence simple
    ...

@pytest.mark.parametrize("B,H,N,D", [
    (1, 1, 128, 64),
    (2, 4, 256, 128),
])
def test_my_feature(B, H, N, D):
    Q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    O_ref = ref_my_feature(Q, K, V)
    O_impl = flash_attn_cuda.my_feature_fwd(Q, K, V)

    rel_err = (O_impl.float() - O_ref.float()).abs().max() / O_ref.float().abs().max()
    assert rel_err < 2e-2, f"Error: {rel_err}"
```

### Étape 2 : Exécuter test

```bash
python -m pytest cuda/tests/test_my_feature.py -v
```

### Étape 3 : Intégrer dans CI/CD (optionnel)

Ajouter à `make test` cible (dans Makefile).

---

## 4.6 Top 10 Common Errors & Solutions

### 1. **SMEM Bank Conflicts**
**Symptôme :** Kernel lent (10-20% pas attendu).

**Debug :** Nsight Compute → "Memory Bank Conflicts > 0".

**Fix :** Vérifier SMEM_PAD_HALFS assez grand.
```cuda
// Bad: stride = 64 (32 banks exactly)
half K_smem[64][64];

// Good: stride = 72 (64 + 8 halfs)
half K_smem[64][72];  // ou use padded_stride()
```

### 2. **Kernel Launch Grid Too Large**
**Symptôme :** `CUDA error: invalid argument`.

**Cause :** `cdiv(N, BR)` * BH > max grid dimensions (~65000 blocks).

**Fix :**
```cuda
// If N > 60 million (unlikely), use 2D grid for Q:
dim3 grid(cdiv(N, BR), cdiv(BH, 64), 1);  // split BH too
// Adjust kernel to handle 3D grid.
```

### 3. **FP16 Precision Loss**
**Symptôm :** rel_err > 2%, output looks wrong.

**Cause :** FP16 only 10 bits mantissa. Accumulating many small values → loss.

**Fix :**
- Increase FWD_BR, FWD_BC (fewer GEMM iterations)
- Mixed precision : K, V in FP16 but accumulate in FP32 (alreadydone!)
- Use BF16 if GPU supports (RTX 4000+)

### 4. **CausalMask Logic Error**
**Symptom :** Forward pass runs but output wrong with `causal=True`.

**Check :**
```cuda
// Condition: i >= j (i can attend to j)
if (causal && global_j > my_q_row_global)
    dot = -1e20f;  // Correct
```

Wrong : `global_j >= my_q_row_global` (masks diagonal).

### 5. **Double-write Bugs (atomicAdd)**
**Symptom :** Backward pass rel_err high.

**Cause :** dQ written multiple times without proper coordination.

**Fix :** Ensure only one thread per dQ element does atomicAdd.
```cuda
if (d_tid == 0) {  // only first partner writes dQ
    atomicAdd(&dQ_bh[q_global * D_dim + d_idx], grad);
}
```

### 6. **Out-of-Bounds Access**
**Symptom :** Silent corruption or crash.

**Pattern :**
```cuda
// Always check bounds:
if (q_global < N && d_idx < D) {
    // access Q[q_global * D + d_idx]
}
```

### 7. **Kernel Not Found / Linker Error**
**Symptom :** `nvcc fatal: unknown code form` or link failure.

**Cause :** .cu file not listed in CMakeLists.txt or setup.py.

**Fix :**
```cmake
add_library(flash_attn_kernels STATIC
    src/naive_attention.cu
    src/flash_attn_fwd.cu
    src/flash_attn_bwd.cu
    src/my_new_kernel.cu  # ADD HERE
)
```

### 8. **CUDA_CHECK_LAST Failures**
**Symptom :** Program exits with "CUDA kernel error".

**Likely Causes :**
- Shared memory exceeds 96 KB per block (call `set_smem_limit`)
- Block size exceeds 1024
- Grid dimension > 65535

**Debug :**
```cpp
set_smem_limit(my_kernel, smem_bytes);  // before launch
// Ensure smem_bytes < 98304 (98 KB typical A100)
```

### 9. **Causal Masking with KV Block Skipping**
**Symptom :** Causal-masked output differs from naive.

**Cause :** KV block skipping logic wrong.

**Correct Logic :**
```cuda
const int num_kv_blocks = cdiv(N, FWD_BC);
int kv_end = num_kv_blocks;
if (causal) {
    // Q block spans rows [q_start, q_start + FWD_BR)
    // Can only attend KV blocks up to the last Q row.
    kv_end = cdiv(q_start + FWD_BR, FWD_BC);
    if (kv_end > num_kv_blocks) kv_end = num_kv_blocks;
}
```

### 10. **atomicAdd Race Condition (dQ Backward)**
**Symptom :** Backward results inconsistent between runs.

**Cause :** Multiple Q blocks writing dQ[row] simultaneously, race in atomicAdd.

**Note :** atomicAdd is safe (atomic), but order non-deterministic. This is acceptable (gradients sum, order doesn't matter).

**If strict determinism needed :** Use different strategy (e.g., per-block dQ tmp, then reduce).

---

## 4.7 Performance Tuning Checklist

- [ ] **SMEM padding :** SMEM_PAD_HALFS=8 set. Verify stride = D+8.
- [ ] **Thread occupancy :** `FWD_NUM_THREADS=128` (4 warps). ~ 60-80% occupancy.
- [ ] **Block size :** Br=Bc=64 for D=64. If D=128, try Br=32, Bc=64.
- [ ] **Register usage :** q_reg[128], o_acc[128], s_row[64] ≤ 256 regs/thread. Check with `cuobjdump`.
- [ ] **Memory coalescing :** HBM reads/writes should be 128-byte aligned.
- [ ] **Cache locality :** Q stays in registers (no memory), K,V stream from HBM.
- [ ] **Warp shuffles :** No serialization (all lanes same operation).
- [ ] **Causal optimization :** Skip entire KV blocks if causal (saves compute).

---

## 4.8 Quick Reference : File Locations & Entry Points

| Task | File | Function/Target |
|------|------|-----------------|
| Build | CMakeLists.txt | `add_library(flash_attn_kernels ...)` |
| Forward | cuda/src/flash_attn_fwd.cu | `flash_attn_fwd_kernel`, `flash_attn_fwd()` |
| Backward | cuda/src/flash_attn_bwd.cu | `compute_D_kernel`, `flash_attn_bwd_kernel` |
| Python binding | cuda/src/torch_bindings.cpp | `flash_fwd()`, `flash_bwd()` |
| Tests | cuda/tests/test_correctness.py | `test_flash_fwd()`, `test_flash_bwd()` |
| Benchmark | benchmarks/bench_all.py | `run_benchmarks()` |
| Triton alt | triton/flash_attn_triton.py | `_flash_attn_fwd_kernel()` |
| Config tuning | flash_attn_fwd.cu (lines 46-48) | FWD_BR, FWD_BC, FWD_NUM_THREADS |
| SMEM tuning | cuda/include/smem_utils.cuh (line 23) | SMEM_PAD_HALFS |

---

## 4.9 Resources pour Aller Plus Loin

### Papers & Articles

1. **Tri Dao et al., "Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)**
   - https://arxiv.org/abs/2205.14135
   - Algorithme fondateur

2. **Tri Dao et al., "Flash-2: Faster Causal Masked Attention with Parallel Block Processing" (2023)**
   - https://arxiv.org/abs/2307.08691
   - Version complète (ce cours implémente essentiellement ceci)

3. **Milakov & Gimelshein, "Online Normalizer Calculation for Softmax" (2018)**
   - https://arxiv.org/abs/1805.02867
   - Online softmax algorithm

4. **Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)**
   - https://www.eecs.berkeley.edu/~kubitron/papers/roofline.pdf
   - Performance modeling (seen in Chapitre 7)

### Outils & Frameworks

- **NVIDIA Nsight Compute** : GPU profiling
- **NVIDIA cuda-gdb** : GPU debugging
- **NVIDIA NCCL** : Distributed training (reduce dK, dV across GPUs)
- **Triton** : Higher-level alternative à CUDA bas-niveau
- **PyTorch Autograd** : Automatic differentiation (calls backward kernel)

### Communautés & Forums

- **NVIDIA Developer Forums** : https://forums.developer.nvidia.com/c/cuda/
- **PyTorch Discussion** : https://discuss.pytorch.org/
- **Hugging Face** : Practical usage of FlashAttention in transformers
- **GitHub Issues** : Official flash-attention repo (Tri Dao)

### Extensions Naturelles

1. **Multi-GPU Flash Attention :** Utiliser NCCL pour réduire dQ, dK, dV across GPUs.
2. **GQA/MQA Support :** Grouped/Multi-Query attention (économie mémoire).
3. **KV Cache Optimization :** Pour inference (reuse K,V).
4. **Sparsity :** Patterns d'attention sparse (e.g., local attention).
5. **Quantization :** INT8 Flash Attention (precision trading).
6. **Custom CUDA Operators :** Intégrer dans frameworks (JAX, TensorFlow).

---

## 4.10 Résumé Opérationnel

| Phase | Commande | Durée | Notes |
|-------|----------|-------|-------|
| **Setup** | `pip install pytorch triton pytest matplotlib` | 5 min | |
| **Build** | `make install` | 1 min | First time 30-60s |
| **Test** | `make test` | 30s | Validates kernel correctness |
| **Benchmark** | `make bench` | 10 min | Compares 5 impls |
| **Visualize** | `make plots` | 1 min | Matplotlib graphs |
| **Debug** | `ncu python script.py` | 5 min per run | GPU profiling |
| **Develop** | Edit `.cu`, `make install` | Depends | Iterative |
| **Clean** | `make clean` | <1 sec | Before fresh build |

---

**FIN DE LA PARTIE 4 — GUIDE OPÉRATIONNEL**

*Résumé : Nous avons couvert l'installation complète, chaque Makefile target en détail, debugging avec Nsight & cuda-gdb, patterns pour contribuer (GQA example), anatomie des tests, top 10 errors & fixes, et ressources pour approfondissement. Le polycopié est maintenant complet et actionnable.*

---

# Conclusion Générale

Ce polycopié de cours a fourni une exploration exhaustive de **Flash Attention 2 et Programmation CUDA**, du théorique au pratique :

**PARTIE 1 (Chapitres 1-10) :** Les fondations — architecture GPU, modèle CUDA, attention, softmax online, Roofline, et l'algorithme Flash Attention révolutionnaire.

**PARTIE 2 (Sections 2.1-2.9) :** L'anatomie du projet — arborescence, architecture système, flux de données, cycle de vie, choix de design, conventions, et goulots d'étranglement.

**PARTIE 3 (Sections 3.1-3.9) :** L'analyse profonde — walkthrough ligne-à-ligne de chaque composant (cuda_utils, smem_utils, online_softmax, kernels forward/backward, bindings PyTorch, tests, benchmarks, Triton). Chaque section : rôle, dépendances, code, liens théoriques, diagrammes, pièges.

**PARTIE 4 (Sections 4.1-4.10) :** Le guide opérationnel — installation, Makefile targets, debugging, contribution patterns, top 10 errors, tuning, et ressources.

**Pour le lecteur :**
- Débutant : Lire PARTIE 1 entièrement. Tenter `make install && make test` pour sentir.
- Intermédiaire : PARTIE 1 + 2. Étudier 3.4 (forward kernel) en profondeur.
- Avancé : PARTIE 3 complète. Implémenter extension (e.g., GQA) via 4.4.
- Praticien : PARTIE 4. Benchmark, profile, optimize.

Le code fourni (**~8000 lignes CUDA + Triton + Python**) est production-ready, documenté, et testé. C'est une excellente base pour :
- Comprendre attention mechanisms et GPU computing
- Prototyper kernels CUDA haute-performance
- Contribuer à projects open-source (PyTorch, Hugging Face, etc.)

**Prochaines étapes :** Expérimenter avec FWD_BR, FWD_BC. Ajouter GQA. Profiler sur votre GPU. Comparer vs Tri Dao's official implementation. La maîtrise s'acquiert par **lecture + expérimentation**.

Bonne chance !

---

**Fin du polycopié — Février 2026**

