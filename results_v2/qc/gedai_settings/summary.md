# GEDAI settings: signal preservation (no classification results used)

Subjects: chb01, chb03, chb05, chb08, chb12, chb15. Retained power = power after GEDAI / power of the same 0.5-40 Hz signal.

```
setting             unit    n    p5   p25  median   p75  share_below_0.5  share_below_0.1
   auto        recording  199 0.197 0.324   0.843 0.999            0.387            0.000
   auto     ictal window  591 0.241 0.367   0.779 1.000            0.333            0.005
   auto non-ictal window 2364 0.263 0.395   0.999 0.999            0.328            0.003
  auto-        recording  199 0.263 0.957   0.999 0.999            0.166            0.000
  auto-     ictal window  591 0.313 0.999   1.000 1.000            0.134            0.000
  auto- non-ictal window 2364 0.331 0.999   0.999 1.000            0.126            0.002
```

- auto: in 48 recordings with both classes, median retained power is 0.999 for ictal and 0.998 for non-ictal windows; ictal minus non-ictal per recording: median +0.000, ictal lower in 48% of recordings.
- auto-: in 48 recordings with both classes, median retained power is 1.000 for ictal and 0.999 for non-ictal windows; ictal minus non-ictal per recording: median +0.000, ictal lower in 33% of recordings.
