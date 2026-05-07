## GSK8K(5-shot) - LLaDA (Gen length 256/512)

| Method | Speed (tokens/s) @256 | Speed (tokens/s) @512 | Accuracy @256 | Accuracy @512 |
|---|---:|---:|---:|---:|
| Baseline | 7.34 | 3.55 | 0.7793 | 0.7665 |
| Cache | 23.22 | 10.45 | 0.7779 | 0.7619 |
| Cache variable | 24.56 | 16.54 | 0.7779 | 0.7619 |
| Parallel | 23.79 | 19.14 | 0.7816 | 0.7756 |
| Cache+parallel | 62.56 | 38.78 | 0.7748 | 0.7650 |

## GSK8K - Dream (Gen length 256/512)

| Method | Speed (tokens/s) @256 | Speed (tokens/s) @512 | Accuracy @256 | Accuracy @512 |
|---|---:|---:|---:|---:|
| Baseline | 9.72 | 8.12 | 0.7536 | 0.7551 |
| Cache | 33.35 | 25.98 | 0.7491 | 0.7619 |
| Parallel | 14.59 | 15.08 | 0.7271 | 0.7756 |
| Cache+parallel | 48.54 | 42.54 | 0.7339 | 0.7650 |

## HumanEval - LLaDA (Gen length 256/512)

| Method | Speed (tokens/s) @256 | Speed (tokens/s) @512 | Accuracy @256 | Accuracy @512 |
|---|---:|---:|---:|---:|
| Baseline | 20.02 | 13.36 | 0.4146 | 0.4268 |
| Cache | 23.25 | 20.17 | 0.4329 | 0.4329 |
| Parallel | 64.74 | 41.93 | 0.4268 | 0.4268 |
| Cache+parallel | 69.03 | 58.10 | 0.4329 | 0.4146 |

## HumanEval - Dream (Gen length 256/512)

| Method | Speed (tokens/s) @256 | Speed (tokens/s) @512 | Accuracy @256 | Accuracy @512 |
|---|---:|---:|---:|---:|
| Baseline | 24.53 | 17.29 | 0.4878 | 0.5488 |
| Cache | 34.09 | 27.92 | 0.5183 | 0.5305 |
| Parallel | 40.16 | 28.28 | 0.4695 | 0.5244 |
| Cache+parallel | 56.98 | 47.57 | 0.5549 | 0.5610 |
| Dual-cache-parallel | 46.83 | 47.54 | 0.5244 | 0.5183 |

## Table 4 Performance and Speedup Comparison on LLaDA Between 5-Shot and 8-Shot Settings at Generation Length 1024. 
| Setting | LLaDA(baseline) | parallel+no cache | parallel+PrefixCache | parallel+ DualCache |
|---|---:|---:|---:|---:|
| 5-shot | 0.7809+1.19 | 0.7854+11.55 | 0.7559+16.53 | 0.7589+24.39 |
| 8-shot |  | 0.7733+9.20 | 0.7506+14.22 | 0.7491+20.05 |

## Table 5: Impact of Generation Length on Accuracy and Speedup Under 8-Shot for LLaDA.(parallel)

| Len | LLaDA(baseline) | parallel+no cache | parallel+PrefixCache | parallel+ DualCache |
|---|---:|---:|---:|---:|
| 256 | 0.7801+5.12 | 0.7832+16.17 | 0.7779+51.67 | 0.7741+49.09 |
| 512 | 0.7847+2.62 | 0.7892+13.79 | 0.7551+32.45 | 0.7544+37.57 |
| 1024 | 0.4375+0.92 | 0.7733+9.20 | 0.7506+14.22 | 0.7491+20.05 |
