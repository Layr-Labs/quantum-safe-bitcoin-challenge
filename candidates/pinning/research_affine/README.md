# Nouvelle piste : additions affines et service d'inversion asynchrone

Recherche du 23 septembre 2026, construite depuis le candidat
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, puis intégrée pour un test Yukon.
Le parcours actif est maintenant dans `../affine_search.cuh` et
`../affine_driver.cuh`, sélectionné par défaut dans `../pinning.cu`.
Les modifications restent intégralement dans `candidates/pinning/`.
La note publique anglaise est [SUBMISSION-AFFINE.md](../SUBMISSION-AFFINE.md).

## L'idée et son intérêt

Le candidat actuel évite les inversions pendant sa multiplication scalaire :
il conserve des coordonnées projectives et effectue davantage de produits à
chaque addition. Une addition affine serait moins coûteuse, mais attendrait
normalement une inversion. Les essais précédents d'inversion dans une seule
lane d'un bloc ont été très lents.

La proposition sépare les deux rôles dans un **même lancement coopératif** :

1. Chaque bloc de travail fait avancer 128 candidats, construit un arbre de
   produits des dénominateurs dans 12 Kio de mémoire partagée et publie sa racine.
2. Chaque lane d'un warp de service inverse la racine d'un bloc différent.
   Un warp peut donc traiter 32 racines simultanément.
3. Chaque bloc attend uniquement sa réponse, redescend son arbre et continue.
   Il n'y a pas de barrière globale entre les 14 étapes.
4. La table du dernier chiffre stocke directement `L+R` et `L−R`. La dernière
   étape produit les deux clés publiques sans matérialiser le point intermédiaire
   `P+L`, puis faire séparément les deux opérations de récupération.

Les points des workers restent en registres ; leurs arbres restent en mémoire
partagée. Seules les racines, leurs inverses et les compteurs de génération
passent par les boîtes aux lettres. Le lancement doit respecter l'occupation
réelle du kernel pour garantir la résidence de ses blocs de service et de travail.

Les signes du dernier chiffre ont une subtilité : pour `−L`, il faut **échanger
les deux entrées et négativer leurs ordonnées**. Cette règle préserve les recids.
La table finale passe de 4 à 8 Mio ; le total des tables passe de 64 à 68 Mio.
Il s'agit des tables actives : cette première intégration conserve aussi les
4 Mio inutilisés de l'ancienne dernière fenêtre, soit 72 Mio alloués au total.

## Décompte honnête du potentiel

`M` = multiplication de corps ; `S` = carré spécialisé ; `W` = calcul partiel
de parité ; `I` = inversion. Les colonnes ne sont pas des unités de temps.

| Par candidat | M | S | W | I |
|---|---:|---:|---:|---:|
| Candidat actuel, batch de 2²³ | 106,1484 | 28 | 2 | 1 / 8 388 608 |
| Architecture affine visée | 72,6719 | 15 | 2 | 14 / 128 |
| Prototype de référence implémenté | 89,6719 | 0 | 0 | 14 / 128 |

La cible retire environ **31,5 % des multiplications explicites et 46,4 % des
carrés**, mais demande beaucoup plus d'inversions. Le prototype privilégie des
produits complets pour les carrés et les parités. Il ne réalise donc pas encore
le décompte optimisé. Les additions, normalisations, accès mémoire, barrières,
SHA256 et travail interne des inversions sont exclus de ce tableau.

Le service ne supprime pas le coût des inversions. Avec 128 SM, quatre blocs
résidents par SM et 128 threads par bloc, il reste 508 blocs de travail et
quatre blocs de service. Pour dépasser 814 millions de candidats par seconde,
les 14 étapes doivent prendre en moyenne **moins de 5,706 µs chacune**, calcul
et attente compris. C'est une condition nécessaire ; ce n'est pas une mesure.
Pour dépasser de 1 % la frontière observée de 813 651 852 candidats/s, il faut
au moins 821 788 371 candidats/s : le même budget se resserre à **5,652 µs**.

La preuve et le décompte reproductible sont dans [SHIFTED-TAIL.md](SHIFTED-TAIL.md),
[operation_counts.json](operation_counts.json) et [INVERSE-SERVICE.md](INVERSE-SERVICE.md).

## Ce qui est concret

- [persistent_affine_probe.cu](persistent_affine_probe.cu) contient le kernel
  complet de l'expérience : 13 additions, la sortie double, les arbres, les
  boîtes aux lettres, le lancement coopératif et une comparaison OpenSSL.
- [inverse_service.cuh](inverse_service.cuh) utilise des publications release
  et lectures acquire, une génération par échange et un arrêt explicite. Les
  votes des warps empêchent les attentes divergentes entre lanes de service.
- [inverse_tree.cuh](inverse_tree.cuh) réalise exactement `3×128−3 = 381`
  multiplications par arbre d'inversion. Le produit partagé utilise par défaut
  les retenues complètes : une erreur de racine contaminerait tout un bloc.
- [shifted_tail.cuh](shifted_tail.cuh) construit la table réelle, charge les
  chiffres signés et implémente les formules de sortie double.

Le prototype recharge les entrées finales après réception de l'inverse, puis
termine une clé à la fois. Cela réduit la durée de vie des temporaires. La
première version déroulée débordait des registres ; cette disposition compile
sur `sm_89` avec 128 registres, 12 Kio partagés et aucun spill déclaré à la
borne de quatre blocs. La pile de 120 octets et les accès locaux du service
d'inversion sont examinés séparément dans le rapport de compilation.

| Borne de blocs par SM | Registres | Mémoire partagée | Spills écrits / lus |
|---|---:|---:|---:|
| 3 | 148 | 12 288 octets | 0 / 0 |
| 4 | 128 | 12 288 octets | 0 / 0 |
| 6 | 80 | 12 288 octets | 356 / 248 octets |

L'inspection des instructions confirme zéro `LDL`/`STL` dans les workers à la
borne quatre ; les accès locaux restants appartiennent au service d'inversion.
Les sources, commandes, hashes et ressources figurent dans
[persistent_resources.json](persistent_resources.json). Ces nombres concernent
le code machine natif produit par CUDA 12.8, pas un débit observé.

Une revue a également détecté et corrigé une course : le prochain arbre pouvait
réécrire ses feuilles avant qu'un warp lent ait lu les anciens frères. Une
barrière de bloc termine maintenant chaque addition ordinaire.

## Vérifications et limites

Les résultats conservés distinguent les tests CPU, la compilation CUDA et
l'exécution GPU. Les tests de composants comprennent :

- 13 440 inverses de feuilles dans 112 arbres, dont valeurs limites, masques,
  alias et bornes des buffers ; zéro écart avec une référence indépendante.
- 2 368 points de table, 543 paires de clés, 1 086 SHA256 de clés comprimées,
  225 recodages scalaires et les cas singuliers ; zéro écart.
- 34 641 soustractions modulaires issues du PTX réel et 69 282 vérifications
  d'alias ; zéro écart.
- 770 états du protocole de publication, deux contrôles négatifs, ainsi que
  l'exécution multithread du header réel avec atomiques CPU.
- Le corps du kernel complet extrait du source, deux blocs de travail et un
  warp de service exécutés sur CPU avec barrières et arithmétique exacte :
  56 échanges de racines, 512 paires de clés, 1 024 points comparés à OpenSSL,
  zéro écart ou refus. Les 45 912 produits comptés correspondent exactement
  au décompte du prototype de référence. Les résultats et hashes des sources
  sont dans [persistent_cpu_result.json](persistent_cpu_result.json).

Le probe de recherche initial utilise des tables synthétiques de points valides et compare des
résultats de chaque bloc avec OpenSSL. Ses tables sont petites : son éventuel
temps d'exécution ne serait **pas** un score du benchmark pinning.
Le candidat intégré utilise les vrais scalaires SHA256, les tables complètes,
des masques de résultats sans débordement et un secours OpenSSL pour les cas
singuliers. Il vérifie les deux recids sur CPU avant toute publication. Les
carrés et parités spécialisés restent à implémenter : cette première soumission
conserve les produits complets du prototype de référence.

L'audit CPU du **nouveau kernel réel** couvre 513 candidats et 1 026 clés,
70 échanges de racines et 127 itérations de lanes inactives, sans écart. Les
primitives SHA, recodage et hash de clé sont aussi comparées indépendamment
à OpenSSL. Le parcours de publication est testé sur 10 020 emplacements,
y compris les faux candidats, les exceptions et les doubles résultats.
Voir `search_cpu_result.json`, `search_primitives_result.json` et
`affine_publish_result.json`.

Au démarrage distant, le solveur exécutera encore 641 candidats sur GPU et
comparera leurs 1 282 clés ainsi que les masques de succès à OpenSSL, avant le
parcours mesuré. Une grille réduite teste les tiles multiples et l'arrêt décalé
des workers. Ce contrôle GPU n'a pas pu être exécuté localement.

Aucun GPU CUDA utilisable n'est présent dans cet environnement. Le lancement
local échoue à `cudaGetDeviceProperties`; le runner Yukon de baseline manque
aussi de `/opt/starkware-challenge/bench-exec.sh`. Aucun débit, gain classé ou
rappel GPU n'est établi. La compilation ne valide pas l'ordonnanceur CUDA ni
l'exécution des instructions. La note de soumission distingue ces limites des
résultats CPU et des compilations réussies.

## Expérience GPU qui décidera

Depuis le répertoire du benchmark, avec CUDA et OpenSSL disponibles :

```sh
rtk proxy python3 candidates/pinning/research_affine/audit_persistent_cpu.py
rtk proxy python3 candidates/pinning/research_affine/compile_probe.py --ptx
rtk proxy timeout 120s candidates/pinning/research_affine/_build/persistent_affine_probe_b4 2
```

Le script accepte `--nvcc` et `--cuobjdump` pour sélectionner une installation
CUDA. La compilation locale a utilisé `/tmp/qsb-cuda/nvcc-local`, qui choisit
une version de GCC compatible avec le toolkit en cache.

1. Compiler et exécuter le probe sur `sm_89`, puis lancer les outils CUDA de
   vérification mémoire et synchronisation. Exiger zéro erreur et zéro refus
   sur ses entrées synthétiques non singulières.
2. Mesurer l'occupation réelle et le temps complet des chaînes, puis le temps
   demande/réponse sous charge. La borne de 5,706 µs permet de rejeter rapidement
   la configuration si la seule inversion ou l'attente est déjà trop lente.
3. Comparer le solveur intégré au baseline avec le runner Yukon. Si l'inversion
   et la synchronisation laissent une marge suffisante, mesurer ensuite les
   carrés et parités spécialisés comme une expérience séparée.

La ressource limitante peut être la latence d'inversion, l'occupation, les
barrières ou la lecture des tables. L'alternative d'inversion de Fermat compile
avec moins de registres, mais ses 272 produits complets peuvent dépasser à eux
seuls le budget de temps. Elle n'est pas présumée plus rapide.

## Portée de l'originalité

Les coordonnées affines, l'inversion par lots de Montgomery et le déplacement
de constantes dans une table sont des techniques connues. Le travail proposé
est leur combinaison adaptée à ce parcours de 15 fenêtres : blocs persistants,
service asynchrone de racines indépendantes et récupération double absorbée
par la dernière table. Le [code auteur de gECC](https://github.com/CGCL-codes/gECC/blob/322c1c143d6a886747308f5b4d88603d6e30a48a/include/gecc/arith/batch_ec.h#L153)
fait inverser la racine localement par une lane du bloc, avant synchronisation
et descente. Il décrit déjà la fusion du parcours scalaire ; la distinction
examinée ici est le service séparé de racines avec boîtes aux lettres, pas la
fusion seule. La recherche dans les propositions publiques consultées
n'a pas trouvé cette combinaison précise ; cela ne prouve pas une première
mondiale. Les antécédents et sources primaires sont cités dans les deux notes
techniques liées ci-dessus.
