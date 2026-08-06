# Oracle predictability report (M2)

- traces: 100 from `runs\traces\Qwen3-0.6B` (target `Qwen/Qwen3-0.6B`)
- overall speedup upper bound: 2.565
- **verdict: NO-GO**

| class | positions | mean recovered | cov@d>=1 | cov@d>=2 | cov@d>=4 |
| --- | --- | --- | --- | --- | --- |
| chat | 2935 | 0.320 | 0.058 | 0.056 | 0.055 |
| code | 3433 | 4.134 | 0.569 | 0.543 | 0.540 |
| math | 4800 | 1.240 | 0.201 | 0.201 | 0.201 |
| structured | 2664 | 0.211 | 0.053 | 0.033 | 0.033 |

Gate: {
  "verdict": "NO-GO",
  "exit_code": 77,
  "passing_classes": [
    "code"
  ],
  "required_classes_ok": false,
  "class_count_ok": false,
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
