# Break-Even Summary

Dual wins when `compute + comm - overlap <= single latency`.

| Phase | Setup | Batch | Seq | Context | Steps | Single p50 (ms) | Dual p50 (ms) | Dual compute (ms) | Dual comm (ms) | Measured overlap (ms) | Required overlap to tie (ms) | Additional overlap needed (ms) | Comm budget to tie (ms) | Dual speedup | Dual wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefill | dual_die_tensor_optimized | 2 | 2048 | 0 | 0 | 14.3506 | 410.7458 | 151.8780 | 258.8678 | 0.0000 | 396.3953 | 396.3953 | 0.0000 | 0.0349 | False |
| prefill | dual_die_request_sharded | 2 | 2048 | 0 | 0 | 14.3506 | 23.8242 | 23.8242 | 0.0000 | 0.0000 | 9.4737 | 9.4737 | 0.0000 | 0.6024 | False |
| prefill | dual_die_tensor_optimized | 2 | 4096 | 0 | 0 | 36.7173 | 1646.0509 | 572.6591 | 1073.3918 | 0.0000 | 1609.3336 | 1609.3336 | 0.0000 | 0.0223 | False |
| prefill | dual_die_request_sharded | 2 | 4096 | 0 | 0 | 36.7173 | 40.6145 | 40.6145 | 0.0000 | 0.0000 | 3.8972 | 3.8972 | 0.0000 | 0.9040 | False |
| decode | dual_die_tensor_optimized | 8 | 1 | 2048 | 4 | 48.7047 | 173.6420 | 31.2211 | 142.4209 | 0.0000 | 124.9373 | 124.9373 | 17.4837 | 0.2805 | False |
| decode | dual_die_request_sharded | 8 | 1 | 2048 | 4 | 48.7047 | 43.4234 | 43.4234 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.2814 | 1.1216 | True |
| decode | dual_die_tensor_optimized | 16 | 1 | 2048 | 4 | 52.7816 | 184.2053 | 31.4581 | 152.7472 | 0.0000 | 131.4237 | 131.4237 | 21.3235 | 0.2865 | False |
| decode | dual_die_request_sharded | 16 | 1 | 2048 | 4 | 52.7816 | 40.7226 | 40.7226 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.0590 | 1.2961 | True |
