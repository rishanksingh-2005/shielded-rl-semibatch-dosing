# Step 8: combined comparison (5 seeds; test = 27 grid + 100 random reactors per seed)

## Reactor 1: acetic anhydride hydrolysis (goal: finish fast, stay below 50 C)

| Method | Overheated (test) | 95 % interval | Worst peak | Unfinished | batch time vs oracle: mean [seed range] | Practice overheated | Cooling failure |
|---|---|---|---|---|---|---|---|
| PID, robust tuned | 0 / 127 | 0.0-2.9 % | 49.8 C | 0 | +5.3 % | - | - |
| Careful brake, no AI | 0 / 127 | 0.0-2.9 % | 50.0 C | 0 | +8.8 % | - | - |
| Oracle brake (knows true reactor) | 0 / 127 | 0.0-2.9 % | 50.0 C | 0 | +0.0 % | - | - |
| Driver A, penalty, nominal practice | 126 / 635 | 16.8-23.2 % | 54.6 C | 127 | +13.1 % [+9.9, +22.5] | 305 / 1971 | - |
| Driver A, penalty, random practice | 11 / 635 | 0.9-3.1 % | 51.8 C | 0 | +32.6 % [+17.7, +72.0] | 190 / 1983 | - |
| Driver B, normal brake | 306 / 635 | 44.2-52.1 % | 56.2 C | 0 | +3.0 % [+2.3, +3.4] | 0 / 2596 | - |
| **Driver B, careful brake** | 0 / 635 | 0.0-0.6 % | 50.0 C | 0 | +10.1 % [+9.4, +10.8] | 0 / 2376 | - |

## Reactor 2: Fricz/Kummer benchmark (goal: hold 363 K, stay below 373 K)

| Method | Overheated (test) | 95 % interval | Worst peak | Unfinished | tracking error vs oracle brake: mean [seed range] | Practice overheated | Cooling failure |
|---|---|---|---|---|---|---|---|
| PI, best tracking (no safe PI exists) | 88 / 127 | 60.5-77.2 % | 145.0 C | 2 | +22.8 % | - | 10/12 |
| PI + careful brake | 0 / 127 | 0.0-2.9 % | 96.5 C | 50 | +84.4 % | - | 0/12 |
| Careful brake, no AI | 0 / 127 | 0.0-2.9 % | 99.8 C | 4 | +18.0 % | - | 0/12 |
| Oracle brake (knows true reactor) | 0 / 127 | 0.0-2.9 % | 99.8 C | 0 | +0.0 % | - | 3/12 |
| Driver A, penalty, nominal practice (= Fricz 2026) | 48 / 635 | 5.6-9.9 % | 131.1 C | 8 | +6.2 % [-6.7, +20.8] | 635 / 3161 | 0/60 |
| Driver A, penalty, random practice | 87 / 635 | 11.1-16.6 % | 139.1 C | 30 | +28.8 % [+13.2, +61.8] | 689 / 2941 | 0/60 |
| Driver B, normal brake | 177 / 635 | 24.4-31.5 % | 113.0 C | 11 | +1.1 % [-2.7, +11.4] | 0 / 3145 | 1/60 |
| Driver B, careful brake | 0 / 635 | 0.0-0.6 % | 99.8 C | 30 | +5.2 % [-1.7, +18.5] | 0 / 3113 | 0/60 |
| **Driver B, careful brake + override penalty** | 0 / 635 | 0.0-0.6 % | 99.8 C | 39 | -4.6 % [-11.0, +1.8] | 0 / 3054 | 0/60 |

## Significance

- Reactor 2 tracking, our method vs careful brake alone (paired over 74 test reactors): gap -4.6 % vs +18.0 %, ours better on 70 of 74, Wilcoxon p = 6.9e-12.
- Overheated share, our method vs others (Fisher exact, p-value):
  - reactor 1, test: vs PID, robust tuned: 0/635 vs 0/127, p = 1.0e+00
  - reactor 1, test: vs Careful brake, no AI: 0/635 vs 0/127, p = 1.0e+00
  - reactor 1, test: vs Oracle brake (knows true reactor): 0/635 vs 0/127, p = 1.0e+00
  - reactor 1, test: vs Driver A, penalty, nominal practice: 0/635 vs 126/635, p = 2.4e-41
  - reactor 1, test: vs Driver A, penalty, random practice: 0/635 vs 11/635, p = 9.3e-04
  - reactor 1, test: vs Driver B, normal brake: 0/635 vs 306/635, p = 6.4e-114
  - reactor 1, practice: vs Driver A, penalty, nominal practice: 0/2376 vs 305/1971, p = 1.5e-111
  - reactor 1, practice: vs Driver A, penalty, random practice: 0/2376 vs 190/1983, p = 5.7e-68
  - reactor 1, practice: vs Driver B, normal brake: 0/2376 vs 0/2596, p = 1.0e+00
  - reactor 2, test: vs PI, best tracking (no safe PI exists): 0/635 vs 88/127, p = 6.7e-85
  - reactor 2, test: vs PI + careful brake: 0/635 vs 0/127, p = 1.0e+00
  - reactor 2, test: vs Careful brake, no AI: 0/635 vs 0/127, p = 1.0e+00
  - reactor 2, test: vs Oracle brake (knows true reactor): 0/635 vs 0/127, p = 1.0e+00
  - reactor 2, test: vs Driver A, penalty, nominal practice (= Fricz 2026): 0/635 vs 48/635, p = 2.8e-15
  - reactor 2, test: vs Driver A, penalty, random practice: 0/635 vs 87/635, p = 5.5e-28
  - reactor 2, test: vs Driver B, normal brake: 0/635 vs 177/635, p = 6.4e-60
  - reactor 2, test: vs Driver B, careful brake: 0/635 vs 0/635, p = 1.0e+00
  - reactor 2, practice: vs Driver A, penalty, nominal practice (= Fricz 2026): 0/3054 vs 635/3161, p = 3.7e-202
  - reactor 2, practice: vs Driver A, penalty, random practice: 0/3054 vs 689/2941, p = 4.6e-234
  - reactor 2, practice: vs Driver B, normal brake: 0/3054 vs 0/3145, p = 1.0e+00
  - reactor 2, practice: vs Driver B, careful brake: 0/3054 vs 0/3113, p = 1.0e+00

## Override penalty sensitivity (reactor 2, careful brake, 5 seeds each)

| mu | brake cuts per batch [seed range] | tracking gap vs oracle [seed range] | unfinished (of 635) | overheated |
|---|---|---|---|---|
| 0 | 58 [26-115] | +5.2 % [-1.7, +18.5] | 30 | 0 |
| 1 | 30 [22-49] | -4.6 % [-11.0, +1.8] | 39 | 0 |
| 5 | 18 [8-26] | +7.5 % [-11.4, +32.4] | 65 | 0 |
