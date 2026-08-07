# Oracle predictability report (M2)

- traces: 100 from `runs\traces\Qwen3-0.6B` (target `Qwen/Qwen3-0.6B`)
- overall speedup upper bound: 2.617
- **verdict: GO**

| class | positions | mean recovered | cov@d>=1 | cov@d>=2 | cov@d>=4 |
| --- | --- | --- | --- | --- | --- |
| chat | 2935 | 0.322 | 0.060 | 0.056 | 0.055 |
| code | 3433 | 4.135 | 0.569 | 0.543 | 0.540 |
| math | 4800 | 1.244 | 0.205 | 0.201 | 0.201 |
| structured | 2664 | 0.473 | 0.309 | 0.041 | 0.033 |

Gate: {
  "verdict": "GO",
  "exit_code": 0,
  "passing_classes": [
    "code",
    "structured"
  ],
  "required_classes_ok": true,
  "class_count_ok": true,
  "speedup_ok": true,
  "gate": {
    "min_classes_passing": 2,
    "required_classes": [
      "code",
      "structured"
    ],
    "min_position_coverage_at_1_token": 0.25,
    "min_oracle_expected_speedup": 1.5,
    "speedup_horizon": 8
  }
}
