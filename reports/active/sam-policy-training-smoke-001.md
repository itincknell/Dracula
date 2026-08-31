# Sam policy training smoke 001

## Frozen corpus

- Committed decks: 16
- Committed rows: 524,256
- Terminal leaves: 786,432
- Corpus content digest: `7b1aa431b0ac9f147566cc11f4fb2de524cb5f3fe0420aca46c65ce1fef79079`
- Snapshot digest: `7939d0f67b91ae74d3ec6b1f587b2baa4602c3563d3834882250ee079d6edf4e`
- Training decks/rows: 14 / 458,724
- Validation decks/rows: 2 / 65,532
- Original corpus split labels: `{'training': 16}`
- Snapshot load: 818.79 seconds (640 rows/s)
- Referenced deck bytes: 954,060,734

### Placement rows

| Placement | Training | Validation |
| ---: | ---: | ---: |
| 1 | 84 | 12 |
| 2 | 336 | 48 |
| 3 | 1,344 | 192 |
| 4 | 5,376 | 768 |
| 5 | 21,504 | 3,072 |
| 6 | 86,016 | 12,288 |
| 7 | 344,064 | 49,152 |

- Queen/King rows: 262,128 / 262,128
- Dealer/non-dealer rows: 104,832 / 419,424
- Unique observations: 495,631
- Duplicate observation rows: 28,625
- Conflicting observations: 4,501
- Conflicting duplicate rows: 4,508
- Representative-action-count distribution: `{'2': 393216, '4': 7023, '6': 96548, '8': 19405, '9': 1089, '12': 2925, '15': 4050}`
- Target frequencies by action index: `{'0': 23708, '1': 11555, '2': 22709, '3': 11783, '4': 11245, '5': 21552, '6': 10768, '7': 20868, '8': 23153, '9': 11628, '10': 22621, '11': 10383, '12': 10931, '13': 21243, '14': 10824, '15': 21090, '16': 22480, '17': 10478, '18': 21092, '19': 11267, '20': 10956, '21': 21302, '22': 9274, '23': 20378, '24': 23076, '25': 10499, '26': 22546, '27': 11474, '28': 11131, '29': 20948, '30': 10555, '31': 20739}`
- Privacy verification: passed
- Schema and digest verification: passed

## Bounded optimization

- Stratified training rows: 2,980
- Training cross-entropy: `1.97666 → 1.96674 → 1.96235 → 1.94274`
- Maximum pre-clip gradient norm: 0.28602
- Embedding update L2: 0.109144
- Shared-body update L2: 1.594013
- Pair-head update L2: 0.688803
- Value parameters: 0
- Exact CPU resume: True model / True optimizer
- Exported inference exact: True
- Validation CE/top-1/top-2/top-3: 0.98630 / 0.4232 / 0.8329 / 0.8714

### Validation by placement

| Group | Rows | CE | Top-1 | Top-2 | Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 2.09711 | 0.1667 | 0.2500 | 0.4167 |
| 2 | 48 | 2.47333 | 0.0833 | 0.1875 | 0.2500 |
| 3 | 192 | 2.44246 | 0.0625 | 0.1354 | 0.2188 |
| 4 | 768 | 2.60360 | 0.0794 | 0.1654 | 0.2435 |
| 5 | 3,072 | 2.03190 | 0.1331 | 0.2660 | 0.4004 |
| 6 | 12,288 | 1.76610 | 0.1944 | 0.3620 | 0.5269 |
| 7 | 49,152 | 0.69332 | 0.5057 | 1.0000 | 1.0000 |

### Validation by role

| Group | Rows | CE | Top-1 | Top-2 | Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| king | 32,766 | 0.98721 | 0.4091 | 0.8362 | 0.8743 |
| queen | 32,766 | 0.98538 | 0.4372 | 0.8296 | 0.8684 |

### Validation by dealer status

| Group | Rows | CE | Top-1 | Top-2 | Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dealer | 13,104 | 1.81777 | 0.1873 | 0.3498 | 0.5092 |
| non-dealer | 52,428 | 0.77848 | 0.4821 | 0.9537 | 0.9619 |

## Device benchmark

| Device | Finite | Examples/s | Full epoch | Peak RSS | Swap growth |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | True | 3530.1 | 129.9s | 647,348,224 | 0 |
| MPS | True | 2635.2 | 174.1s | 647,348,224 | 70327992 |

- CPU/MPS maximum parameter difference after identical updates: 0.01150355115532875
- Selected-device projected epoch including train/validation metrics: 180.5s
- Selected full-run device: **cpu**

The full snapshot optimization was not started.
