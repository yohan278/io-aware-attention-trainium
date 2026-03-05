# Decode Throughput At SLO

| Setup | Context | SLO (ms) | Best throughput (tokens/s) | Concurrency | p90 latency (ms) | Feasible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dual_die_request_sharded | 2048 | 250.00 | 1571.61 | 16 | 40.9002 | True |
| dual_die_request_sharded | 2048 | 500.00 | 1571.61 | 16 | 40.9002 | True |
| dual_die_request_sharded | 2048 | 1000.00 | 1571.61 | 16 | 40.9002 | True |
| dual_die_tensor_optimized | 2048 | 250.00 | 347.44 | 16 | 227.7617 | True |
| dual_die_tensor_optimized | 2048 | 500.00 | 347.44 | 16 | 227.7617 | True |
| dual_die_tensor_optimized | 2048 | 1000.00 | 347.44 | 16 | 227.7617 | True |
| single_die | 2048 | 250.00 | 1212.54 | 16 | 58.9357 | True |
| single_die | 2048 | 500.00 | 1212.54 | 16 | 58.9357 | True |
| single_die | 2048 | 1000.00 | 1212.54 | 16 | 58.9357 | True |
