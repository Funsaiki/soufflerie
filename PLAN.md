# Soufflerie numérique : journal technique

Ce document rassemble les choix faits, les mesures de validation, ce qui a été compris en cours
de route et ce qui reste à faire. L'utilisation est décrite dans le [`README.md`](README.md).

État au 12/10/2026 : l'application est fonctionnelle (rendu, mesures, import, perturbations).
La validation quantitative a été interrompue en cours de route. Les chiffres ci-dessous sont
ceux effectivement mesurés.

---

## 1. Architecture retenue

- **Web + WebGPU** (Vite + TypeScript, aucune dépendance d'exécution). Le solveur et le rendu
  partagent les mêmes tampons GPU : rien ne revient au processeur sauf 32 octets de forces par image.
- **Solveur LBM D3Q19**, schéma « pull », deux tampons (A/B), en float32 avec *DDF shifting*
  (on stocke f − w pour garder la précision des petites perturbations autour du repos).
- Les boucles sur les 19 directions sont **déroulées en TypeScript** à la génération du WGSL :
  toutes les populations restent en registres et les constantes sont des littéraux.
- **Collision BGK régularisée** (projection hermitienne d'ordre 2) + **LES Smagorinsky** locale
  (Hou et al. 1996, viscosité turbulente tirée de Π_neq, sans lire les voisins).
  Une collision **BGK** simple reste disponible (`model: 1`) pour comparaison.
- **Forces** par échange de quantité de mouvement sur chaque lien fluide–solide, réduites par
  groupe de travail puis par un second noyau, accumulées sur tous les pas d'une image.
- **Parois** : *bounce-back* **interpolé de Bouzidi** (voir §3.4). Le sol mobile utilise le rebond de Ladd.
- **Voxelisation** par parité de rayons sur les 3 axes avec vote majoritaire (tolère de petits trous).
  Option « épaissir les surfaces » pour les maillages ouverts ou très fins.
- **Rendu** : coupe colorée (9 grandeurs), filets de fumée (particules RK2, peigne d'injection),
  isosurface du critère Q par lancer de rayons avec saut des briques vides (8³), brume de vorticité,
  Cp sur la maquette.
- **Import** : STL (binaire/ASCII), OBJ, glTF/GLB (sans Draco ni .bin externe).
- **Corps intégrés** : sphère, cylindre, ailes NACA 4 chiffres, corps d'Ahmed 25°/35° et voiture
  simplifiée (surfaces implicites maillées par *surface nets*), cube, plaque.

## 2. Pièges rencontrés

| Problème | Cause | Correction |
|---|---|---|
| Écran noir, rien ne se passe | `macro` est un **mot réservé** en WGSL | Renommé `mac`. Les erreurs GPU non capturées sont maintenant affichées dans la console |
| Chrome n'atteint pas le serveur | Vite écoutait sur `[::1]` seulement | `npx vite --host 127.0.0.1` |
| Onglet en arrière-plan : quelques images/s | Chrome ralentit `requestAnimationFrame` sur les onglets cachés | La validation passe par `src/validate.ts`, sans boucle de rendu |

## 3. Validation : mesures et enseignements

Machine de test : **AMD Radeon intégrée** (iGPU), Chrome. Débit mesuré : **≈ 110–125 MLUPS**
(cohérent avec la bande passante mémoire d'un iGPU). C'est pourquoi la résolution par défaut est 1 M cellules.

Cas lancés depuis la console : `const v = await import('/src/validate.ts')`, puis `v.runCase(...)` ou `v.probeCase(...)`.

### 3.1 Sphère, Re = 100 (référence Cx = 1,09 ; Clift, Johnson & Patel)

| Grille | D (cellules) | Veine / blocage | Conditions latérales | Paroi | Cx | Écart |
|---|---|---|---|---|---|---|
| 1 M | 25,4 | standard 6,4 % | équilibre (u et ρ imposés) | escalier | 1,365 | +25 % |
| 1 M | 25,4 | standard 6,4 % | **ouvertes** (ρ = 1, u extrapolée) | escalier | 1,277 | +17 % |
| 0,25 M | 16 | standard 6,6 % | ouvertes | escalier | 1,286 | +18 % |
| 1 M | ≈ 17 | large ≈ 3 % | ouvertes | escalier | ≈ 1,24 | +13 % |
| 2 M | 13,5 | très large 1 % | ouvertes | escalier | 1,196 | +10 % |
| 2 M, u = 0,04 | 13,5 | très large 1 % | ouvertes | escalier | 1,214 | +11 % |
| 1 M | 25,4 | standard 6,4 % | ouvertes | **Bouzidi** | 1,246 | +14 % |
| 1 M, BGK | 25,4 | standard 6,4 % | ouvertes | escalier | 1,28 | +17 % |

Enseignements :
- **Ni la résolution** (16 → 25 cellules : Cx inchangé), **ni le Mach** (u = 0,04 ou 0,08), **ni la collision**
  (régularisée ou BGK : même valeur) ne sont la cause principale.
- **La taille de veine domine** : l'écart diminue nettement quand le blocage baisse (6,4 % → 1 %).
  À bas Reynolds, la perturbation décroît lentement avec la distance, donc l'effet est fort.
- Le passage des faces latérales de « équilibre » à « ouvertes » a corrigé 7 points d'un coup.

### 3.2 Sphère, Re = 1000 (référence Cx ≈ 0,47)

| Grille | D | Veine | LES Cs | Collision | Paroi | Cx | σ |
|---|---|---|---|---|---|---|---|
| 1 M | 25,4 | standard | 0,14 | régularisée | escalier | 0,72 | 0,024 |
| 1 M | 25,4 | standard | 0 | régularisée | escalier | 0,672 | 0,003 |
| 1 M | 25,4 | standard | 0,14 | BGK | escalier | 0,692 | 0,013 |
| 2 M | 22 | large | 0,14 | régularisée | escalier | 0,732 | 0,039 |
| 4 M | 40,3 | standard | 0,14 | régularisée | escalier | 0,681 | 0,034 |
| 1 M | 25,4 | standard | 0,14 | régularisée | **Bouzidi** | 0,675 | 0,051 |

Enseignements :
- L'écart (+45 %) vient de la **couche limite non résolue** : son épaisseur est d'environ D/√Re,
  soit **moins d'une cellule** à D = 25 et à peine plus à D = 40. Le décollement d'une surface lisse
  arrive alors au mauvais endroit et la traînée de culot est surestimée. La convergence en
  résolution existe (0,72 → 0,68) mais elle est lente.
- La LES ajoute environ 7 % ici (0,672 sans, 0,72 avec), un effet secondaire.
- Bouzidi gagne environ 6 % à résolution égale.
- **La sphère est un cas défavorable** (décollement sur surface lisse). Pour l'aérodynamique
  pratique, les corps à décollement fixé par des arêtes sont plus représentatifs (§3.3).

### 3.3 Cube face au vent, Re = 10⁵ (référence Cx ≈ 1,05)

| Grille | Veine / blocage | Cx | Remarque |
|---|---|---|---|
| 1 M | standard 7,9 % | 1,409 (+34 %) | valeur finale |
| 1 M | large | ≈ 1,07 | **valeur instantanée à 73 % du calcul, non moyennée** (validation interrompue) |

Le cas « plaque carrée » de la même série n'est pas valable : sa longueur de référence est son
épaisseur (1 cellule), donc le temps convectif n'avait pas de sens. Il faut le relancer avec
une longueur de référence égale au côté.

### 3.4 Diagnostic par sondes (cube, 0,5 M, veine standard)

`probeCase` relit le champ de vitesse et de densité :
- Sur l'axe en amont, la vitesse vaut 0,98 U en x = 1, 0,91 U à 0,8 D de la face, 0,76 U à 0,55 D.
- **Pression d'entrée Cp ≈ 0,42 et Cp au point d'arrêt ≈ 1,46**, alors que le maximum physique est 1.
  Une vitesse imposée à seulement une longueur en amont d'un corps bluff agit comme une pompe :
  toute la zone amont est surpressurisée et le corps est poussé trop fort.
- Le débit à travers le plan du corps vaut 0,946 : les faces latérales ouvertes laissent bien
  l'écoulement s'écarter, comme prévu.

### 3.5 Conditions d'entrée essayées puis rejetées

| Essai | Résultat |
|---|---|
| Entrée à vitesse imposée et densité flottante (tirée de l'intérieur) | Ondes acoustiques réfléchies : Cx oscille entre 0,3 et 1,3 |
| Entrée en pression totale (p + ½ρu² imposée, vitesse de l'intérieur) | Instable : vitesses transverses amplifiées (uz ≈ 0,7 U en amont) |
| Pression totale avec direction imposée | Encore instable : refoulement en amont du corps |
| **Retenu** : équilibre à U et ρ = 1 à l'entrée, faces latérales ouvertes, sortie en pression | Stable. Le défaut de surpression amont se corrige avec une veine plus longue en amont |

### 3.6 Bouzidi (paroi interpolée)

- `src/geometry/links.ts` lance un rayon sur le maillage réel pour chaque lien fluide→solide et
  calcule q, la distance relative de la paroi le long du lien (stockée sur 8 bits). Les triangles
  sont rangés dans une grille par cellule. L'indice de la cellule frontière est stocké dans les bits
  hauts du mot de drapeau (`type | index << 4`), donc tous les shaders masquent `& 15u`.
- Contrôle : sphère, NACA et Ahmed donnent **0 lien sans impact**, et q est réparti uniformément sur la sphère.
- Gain mesuré : −2,5 % sur Re 100, −6 % sur Re 1000, à résolution égale.

## 4. Conclusions provisoires

1. Le schéma est cohérent : les mêmes valeurs sortent avec BGK ou régularisé, avec u = 0,04 ou 0,08,
   et le calcul est stable jusqu'à τ = 0,50006 (Re 10⁵) grâce à la LES.
2. **La première source d'erreur est la taille de la veine** (blocage et proximité de l'entrée),
   **la seconde est la résolution de la couche limite** pour les surfaces lisses.
3. Pour des valeurs absolues fiables, il faut un blocage d'au plus 3 % (veine « large ») et au moins
   30 à 40 cellules sur la longueur. Pour comparer des variantes, la configuration par défaut suffit
   à condition de garder la même grille et la même veine.

## 5. Suite à donner (par ordre de priorité)

- [ ] **Finir la validation du cube** en veine large et très large (moyennes complètes), puis
      relancer la plaque carrée avec la bonne longueur de référence.
- [ ] **Agrandir les marges par défaut** : amont ≥ 1,5 D et côtés ≈ 1,75–2 D (proche de « large »),
      ou proposer un préréglage « Précision ». Ajouter un avertissement plus visible au-delà de 3 % de blocage.
- [ ] Envisager une **correction de blocage** (type Maskell) affichée à côté des valeurs brutes.
- [ ] Valider le **corps d'Ahmed** 25° (Cx ≈ 0,285) et 35° (≈ 0,26) avec sol fixe, et une **aile NACA**
      (pente de portance, comparaison à la théorie de la ligne portante pour un allongement de 4).
- [ ] Explorer un **amortissement de la LES près des parois** (WALE ou Vreman, ou Smagorinsky
      amorti) et un **modèle de paroi** pour les très grands Reynolds.
- [ ] Performance : écrire les grandeurs macroscopiques uniquement au dernier pas de chaque image
      (≈ 9 % de trafic mémoire en moins) ; stockage FP16 optionnel (`shader-f16`) après validation de sa précision.
- [ ] Exposer dans l'interface le choix escalier / Bouzidi et la veine « très large » (`huge`, utilisée
      seulement en validation pour l'instant).
- [ ] Retirer le bouton « Pause » cliqué par les scripts de validation : prévoir un mode `?validate`
      qui ne démarre pas l'application.
