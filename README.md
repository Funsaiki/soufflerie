# Soufflerie numérique

Soufflerie 3D temps réel dans le navigateur : on importe une maquette (STL, OBJ, glTF/GLB) ou on
choisit un corps de référence, et on observe l'écoulement autour. On mesure les coefficients
aérodynamiques et on perturbe l'air à la souris. Le calcul tourne entièrement sur la carte graphique (WebGPU).

## Lancer

```bash
npm install
npm run dev        # puis ouvrir http://127.0.0.1:5173
npm run build      # version statique dans dist/
```

Navigateur requis : Chrome, Edge ou Opera récents (WebGPU), avec l'accélération matérielle.

## Ce qu'on peut faire

- **Maquettes intégrées** : sphère, cylindre, ailes NACA 2412 / 0012, corps d'Ahmed 25° et 35°
  (référence de l'aérodynamique automobile), voiture simplifiée, cube, plaque.
- **Import** de STL (binaire ou ASCII), OBJ et glTF/GLB, par le bouton ou par glisser-déposer.
  Le modèle est voxelisé, mis à l'échelle et placé automatiquement. On règle l'axe vertical du
  fichier, le lacet, l'incidence et le roulis. La longueur réelle sert au nombre de Reynolds.
- **Sol** : aucun, fixe, ou tapis roulant (sol à la vitesse de l'air, comme en soufflerie automobile),
  avec garde au sol réglable.
- **Écoulement** : air ou eau, vitesse, longueur, d'où le nombre de Reynolds réel. Turbulence
  amont synthétique réglable et rafales.
- **Perturbations à la souris** : outil « Perturber » (ou Ctrl + glisser). Glisser remue l'air dans le
  plan face à la caméra ; maintenir le clic sans bouger souffle dans la direction du regard.
- **Mesures** : Cx, Cz, Cy instantanés, moyennés et écarts-types ; traînée et portance en newtons ;
  finesse ; SCx ; nombre de Strouhal du lâcher tourbillonnaire. Historique en temps convectif t* = t·U/L.
- **Visualisation** : plan de coupe (vitesse, composantes, Cp, pression totale, vorticité,
  vorticité axiale, critère Q) ; filets de fumée émis par un peigne réglable ; isosurfaces du
  critère Q colorées par la vitesse ou brume de vorticité ; pression Cp sur la maquette.

## Le modèle physique

| Élément | Choix |
|---|---|
| Méthode | Boltzmann sur réseau (LBM), D3Q19, schéma « pull », en float32 avec décalage des populations (*DDF shifting*) |
| Collision | BGK régularisée (projection hermitienne d'ordre 2), stable aux grands Reynolds |
| Turbulence | LES Smagorinsky, viscosité turbulente locale calculée sans voisins (Hou et al. 1996) |
| Parois | Rebond interpolé de Bouzidi (position exacte de la paroi par lancer de rayons sur le maillage) ; sol mobile par rebond de Ladd |
| Entrée | Équilibre à vitesse imposée, avec turbulence synthétique (modes de Fourier aléatoires à divergence nulle) |
| Parois latérales | Champ lointain ouvert : pression de référence, vitesse extrapolée de l'intérieur |
| Sortie | Pression imposée, vitesse extrapolée |
| Forces | Échange de quantité de mouvement sur chaque lien fluide–solide |
| Forçage souris | Force volumique gaussienne (schéma de Guo) |

Les unités : la vitesse réseau u (0,08 par défaut, Mach réseau 0,14) est fixe et la viscosité
réseau découle du Reynolds : ν = u·L/Re, τ = 3ν + ½. Δx = L / L_cellules, Δt = u·Δx / U.

## Précision : ce qu'il faut savoir

Voir la section *Validation* de [`PLAN.md`](PLAN.md) pour les chiffres mesurés.

- La précision dépend surtout du **nombre de cellules sur la maquette** (affiché dans
  « Maquette ») et de la **taille de la veine** (blocage affiché dans « Soufflerie »).
- Aux Reynolds réels d'une voiture ou d'une aile (10⁵ à 10⁷), la couche limite est bien plus fine
  qu'une cellule : c'est une **LES sans loi de paroi**. Les décollements francs (arêtes vives, culot,
  lunette d'Ahmed) sont bien captés. Le décollement sur surface lisse (crise de traînée de la
  sphère, décrochage progressif d'un profil) ne l'est que qualitativement.
- Pour comparer des variantes (un aileron, un angle, une forme d'arrière), gardez la même
  résolution et la même veine : les **écarts** entre variantes sont bien plus fiables que les
  valeurs absolues.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `src/lbm/shaders.ts` | Génération du WGSL du solveur (boucles déroulées), champs dérivés |
| `src/lbm/solver.ts` | Tampons GPU, pas de temps, réduction des forces |
| `src/lbm/units.ts` | Conversion physique ↔ réseau |
| `src/geometry/loaders.ts` | Lecture STL, OBJ, glTF/GLB |
| `src/geometry/primitives.ts` | Corps de référence (dont surfaces implicites par *surface nets*) |
| `src/geometry/domain.ts` | Mise à l'échelle, voxelisation par parité de rayons, conditions aux limites |
| `src/render/*` | Caméra, rendu WebGPU (coupe, maquette, fumée, isosurfaces par lancer de rayons) |
| `src/ui/*` | Mesures, graphique, légende |
| `src/main.ts` | Assemblage, interface, souris |
| `src/validate.ts` | Cas de validation sans affichage |
