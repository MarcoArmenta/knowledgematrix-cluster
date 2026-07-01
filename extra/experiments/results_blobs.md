# Training a network via knowledge matrices

Data: synthetic blobs: d=16, classes=4, n_train=2000, n_test=500
Config: hidden=64, epochs=40, lr=0.001, batch_size=64, seed=0

Knowledge-matrix check vs library computer: abs_diff=0.000e+00, rel_diff=0.000e+00 (OK)

## Knowledge matrices  (loss = ||M(x) - E_ii||^2)
  [km_eii     ] epoch   1/40  loss=0.8668  train_acc=0.7285  test_acc=0.7160
  [km_eii     ] epoch   2/40  loss=0.5386  train_acc=0.8590  test_acc=0.8360
  [km_eii     ] epoch   3/40  loss=0.4436  train_acc=0.8455  test_acc=0.8240
  [km_eii     ] epoch   4/40  loss=0.3967  train_acc=0.8585  test_acc=0.8520
  [km_eii     ] epoch   5/40  loss=0.3683  train_acc=0.8950  test_acc=0.8680
  [km_eii     ] epoch   6/40  loss=0.3664  train_acc=0.8755  test_acc=0.8540
  [km_eii     ] epoch   7/40  loss=0.3860  train_acc=0.8495  test_acc=0.8320
  [km_eii     ] epoch   8/40  loss=0.3970  train_acc=0.7965  test_acc=0.7820
  [km_eii     ] epoch   9/40  loss=0.4163  train_acc=0.7600  test_acc=0.7400
  [km_eii     ] epoch  10/40  loss=0.4159  train_acc=0.7970  test_acc=0.7840
  [km_eii     ] epoch  11/40  loss=0.4049  train_acc=0.8425  test_acc=0.8360
  [km_eii     ] epoch  12/40  loss=0.4285  train_acc=0.8055  test_acc=0.8160
  [km_eii     ] epoch  13/40  loss=0.4448  train_acc=0.7315  test_acc=0.7440
  [km_eii     ] epoch  14/40  loss=0.4677  train_acc=0.8220  test_acc=0.8380
  [km_eii     ] epoch  15/40  loss=0.4832  train_acc=0.8380  test_acc=0.8560
  [km_eii     ] epoch  16/40  loss=0.4827  train_acc=0.8380  test_acc=0.8460
  [km_eii     ] epoch  17/40  loss=0.4708  train_acc=0.7800  test_acc=0.7840
  [km_eii     ] epoch  18/40  loss=0.4672  train_acc=0.8020  test_acc=0.8100
  [km_eii     ] epoch  19/40  loss=0.4556  train_acc=0.7755  test_acc=0.7500
  [km_eii     ] epoch  20/40  loss=0.4452  train_acc=0.7750  test_acc=0.7600
  [km_eii     ] epoch  21/40  loss=0.4473  train_acc=0.7630  test_acc=0.7440
  [km_eii     ] epoch  22/40  loss=0.4620  train_acc=0.7575  test_acc=0.7380
  [km_eii     ] epoch  23/40  loss=0.4799  train_acc=0.7715  test_acc=0.7620
  [km_eii     ] epoch  24/40  loss=0.4908  train_acc=0.7435  test_acc=0.7320
  [km_eii     ] epoch  25/40  loss=0.4837  train_acc=0.7310  test_acc=0.7140
  [km_eii     ] epoch  26/40  loss=0.4631  train_acc=0.7920  test_acc=0.7920
  [km_eii     ] epoch  27/40  loss=0.4519  train_acc=0.7880  test_acc=0.7700
  [km_eii     ] epoch  28/40  loss=0.4746  train_acc=0.8140  test_acc=0.8240
  [km_eii     ] epoch  29/40  loss=0.4818  train_acc=0.7965  test_acc=0.7920
  [km_eii     ] epoch  30/40  loss=0.4993  train_acc=0.7650  test_acc=0.7440
  [km_eii     ] epoch  31/40  loss=0.5209  train_acc=0.7865  test_acc=0.7860
  [km_eii     ] epoch  32/40  loss=0.5143  train_acc=0.7370  test_acc=0.7360
  [km_eii     ] epoch  33/40  loss=0.4898  train_acc=0.7250  test_acc=0.7180
  [km_eii     ] epoch  34/40  loss=0.4796  train_acc=0.7560  test_acc=0.7540
  [km_eii     ] epoch  35/40  loss=0.4811  train_acc=0.7340  test_acc=0.7240
  [km_eii     ] epoch  36/40  loss=0.4807  train_acc=0.7565  test_acc=0.7420
  [km_eii     ] epoch  37/40  loss=0.4852  train_acc=0.7635  test_acc=0.7520
  [km_eii     ] epoch  38/40  loss=0.4921  train_acc=0.7435  test_acc=0.7360
  [km_eii     ] epoch  39/40  loss=0.5018  train_acc=0.7300  test_acc=0.7200
  [km_eii     ] epoch  40/40  loss=0.4992  train_acc=0.7525  test_acc=0.7380

## Knowledge matrices  (loss = off-class rows -> 0, true logit -> 1)
  [km_offclass] epoch   1/40  loss=0.3571  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   2/40  loss=0.0762  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   3/40  loss=0.0490  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   4/40  loss=0.0389  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   5/40  loss=0.0319  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   6/40  loss=0.0277  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   7/40  loss=0.0255  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   8/40  loss=0.0230  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch   9/40  loss=0.0210  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  10/40  loss=0.0202  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  11/40  loss=0.0185  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  12/40  loss=0.0171  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  13/40  loss=0.0168  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  14/40  loss=0.0161  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  15/40  loss=0.0153  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  16/40  loss=0.0147  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  17/40  loss=0.0142  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  18/40  loss=0.0140  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  19/40  loss=0.0139  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  20/40  loss=0.0138  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  21/40  loss=0.0141  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  22/40  loss=0.0137  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  23/40  loss=0.0132  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  24/40  loss=0.0137  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  25/40  loss=0.0136  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  26/40  loss=0.0140  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  27/40  loss=0.0139  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  28/40  loss=0.0136  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  29/40  loss=0.0129  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  30/40  loss=0.0122  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  31/40  loss=0.0119  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  32/40  loss=0.0116  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  33/40  loss=0.0114  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  34/40  loss=0.0110  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  35/40  loss=0.0106  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  36/40  loss=0.0104  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  37/40  loss=0.0102  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  38/40  loss=0.0103  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  39/40  loss=0.0100  train_acc=1.0000  test_acc=1.0000
  [km_offclass] epoch  40/40  loss=0.0101  train_acc=1.0000  test_acc=1.0000

## Knowledge matrices  (loss = cross-entropy on row norms)
  [km_rownorm_ce] epoch   1/40  loss=0.9849  train_acc=0.2020  test_acc=0.2020
  [km_rownorm_ce] epoch   2/40  loss=0.2984  train_acc=0.1295  test_acc=0.1200
  [km_rownorm_ce] epoch   3/40  loss=0.0401  train_acc=0.1280  test_acc=0.1260
  [km_rownorm_ce] epoch   4/40  loss=0.0127  train_acc=0.1305  test_acc=0.1260
  [km_rownorm_ce] epoch   5/40  loss=0.0070  train_acc=0.1350  test_acc=0.1340
  [km_rownorm_ce] epoch   6/40  loss=0.0047  train_acc=0.1410  test_acc=0.1380
  [km_rownorm_ce] epoch   7/40  loss=0.0034  train_acc=0.1435  test_acc=0.1380
  [km_rownorm_ce] epoch   8/40  loss=0.0025  train_acc=0.1475  test_acc=0.1400
  [km_rownorm_ce] epoch   9/40  loss=0.0020  train_acc=0.1515  test_acc=0.1400
  [km_rownorm_ce] epoch  10/40  loss=0.0016  train_acc=0.1555  test_acc=0.1420
  [km_rownorm_ce] epoch  11/40  loss=0.0013  train_acc=0.1570  test_acc=0.1460
  [km_rownorm_ce] epoch  12/40  loss=0.0011  train_acc=0.1585  test_acc=0.1480
  [km_rownorm_ce] epoch  13/40  loss=0.0010  train_acc=0.1590  test_acc=0.1480
  [km_rownorm_ce] epoch  14/40  loss=0.0008  train_acc=0.1605  test_acc=0.1500
  [km_rownorm_ce] epoch  15/40  loss=0.0007  train_acc=0.1610  test_acc=0.1500
  [km_rownorm_ce] epoch  16/40  loss=0.0006  train_acc=0.1615  test_acc=0.1500
  [km_rownorm_ce] epoch  17/40  loss=0.0006  train_acc=0.1625  test_acc=0.1520
  [km_rownorm_ce] epoch  18/40  loss=0.0005  train_acc=0.1625  test_acc=0.1520
  [km_rownorm_ce] epoch  19/40  loss=0.0004  train_acc=0.1640  test_acc=0.1540
  [km_rownorm_ce] epoch  20/40  loss=0.0004  train_acc=0.1645  test_acc=0.1540
  [km_rownorm_ce] epoch  21/40  loss=0.0004  train_acc=0.1645  test_acc=0.1540
  [km_rownorm_ce] epoch  22/40  loss=0.0003  train_acc=0.1675  test_acc=0.1560
  [km_rownorm_ce] epoch  23/40  loss=0.0003  train_acc=0.1675  test_acc=0.1560
  [km_rownorm_ce] epoch  24/40  loss=0.0003  train_acc=0.1680  test_acc=0.1580
  [km_rownorm_ce] epoch  25/40  loss=0.0003  train_acc=0.1685  test_acc=0.1580
  [km_rownorm_ce] epoch  26/40  loss=0.0002  train_acc=0.1690  test_acc=0.1580
  [km_rownorm_ce] epoch  27/40  loss=0.0002  train_acc=0.1690  test_acc=0.1580
  [km_rownorm_ce] epoch  28/40  loss=0.0002  train_acc=0.1690  test_acc=0.1620
  [km_rownorm_ce] epoch  29/40  loss=0.0002  train_acc=0.1690  test_acc=0.1620
  [km_rownorm_ce] epoch  30/40  loss=0.0002  train_acc=0.1700  test_acc=0.1640
  [km_rownorm_ce] epoch  31/40  loss=0.0002  train_acc=0.1705  test_acc=0.1640
  [km_rownorm_ce] epoch  32/40  loss=0.0002  train_acc=0.1715  test_acc=0.1640
  [km_rownorm_ce] epoch  33/40  loss=0.0002  train_acc=0.1715  test_acc=0.1640
  [km_rownorm_ce] epoch  34/40  loss=0.0001  train_acc=0.1715  test_acc=0.1640
  [km_rownorm_ce] epoch  35/40  loss=0.0001  train_acc=0.1725  test_acc=0.1640
  [km_rownorm_ce] epoch  36/40  loss=0.0001  train_acc=0.1730  test_acc=0.1640
  [km_rownorm_ce] epoch  37/40  loss=0.0001  train_acc=0.1730  test_acc=0.1640
  [km_rownorm_ce] epoch  38/40  loss=0.0001  train_acc=0.1730  test_acc=0.1640
  [km_rownorm_ce] epoch  39/40  loss=0.0001  train_acc=0.1730  test_acc=0.1640
  [km_rownorm_ce] epoch  40/40  loss=0.0001  train_acc=0.1730  test_acc=0.1640

## Knowledge matrices  (loss = margin on row norms)
  [km_rownorm_margin] epoch   1/40  loss=0.5687  train_acc=0.1975  test_acc=0.2000
  [km_rownorm_margin] epoch   2/40  loss=0.0233  train_acc=0.1900  test_acc=0.1880
  [km_rownorm_margin] epoch   3/40  loss=0.0008  train_acc=0.1855  test_acc=0.1840
  [km_rownorm_margin] epoch   4/40  loss=0.0004  train_acc=0.1930  test_acc=0.1920
  [km_rownorm_margin] epoch   5/40  loss=0.0004  train_acc=0.2000  test_acc=0.2000
  [km_rownorm_margin] epoch   6/40  loss=0.0003  train_acc=0.2000  test_acc=0.2000
  [km_rownorm_margin] epoch   7/40  loss=0.0002  train_acc=0.2010  test_acc=0.2020
  [km_rownorm_margin] epoch   8/40  loss=0.0002  train_acc=0.2015  test_acc=0.2020
  [km_rownorm_margin] epoch   9/40  loss=0.0002  train_acc=0.2025  test_acc=0.2000
  [km_rownorm_margin] epoch  10/40  loss=0.0001  train_acc=0.2015  test_acc=0.2000
  [km_rownorm_margin] epoch  11/40  loss=0.0001  train_acc=0.2015  test_acc=0.2000
  [km_rownorm_margin] epoch  12/40  loss=0.0001  train_acc=0.2035  test_acc=0.2000
  [km_rownorm_margin] epoch  13/40  loss=0.0000  train_acc=0.2025  test_acc=0.2000
  [km_rownorm_margin] epoch  14/40  loss=0.0000  train_acc=0.2070  test_acc=0.2040
  [km_rownorm_margin] epoch  15/40  loss=0.0000  train_acc=0.2080  test_acc=0.2040
  [km_rownorm_margin] epoch  16/40  loss=0.0000  train_acc=0.2070  test_acc=0.2040
  [km_rownorm_margin] epoch  17/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  18/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  19/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  20/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  21/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  22/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  23/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  24/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  25/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  26/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  27/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  28/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  29/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  30/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  31/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  32/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  33/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  34/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  35/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  36/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  37/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  38/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  39/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  40/40  loss=0.0000  train_acc=0.2065  test_acc=0.2040

## Vanilla             (loss = cross-entropy)
  [vanilla    ] epoch   1/40  loss=0.5886  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   2/40  loss=0.0277  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   3/40  loss=0.0053  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   4/40  loss=0.0029  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   5/40  loss=0.0019  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   6/40  loss=0.0013  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   7/40  loss=0.0010  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   8/40  loss=0.0007  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch   9/40  loss=0.0005  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  10/40  loss=0.0004  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  11/40  loss=0.0003  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  12/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  13/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  14/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  15/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  16/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  17/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  18/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  19/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  20/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  21/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  22/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  23/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  24/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  25/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  26/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  27/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  28/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  29/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  30/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  31/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  32/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  33/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  34/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  35/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  36/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  37/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  38/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  39/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla    ] epoch  40/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000

## Final comparison (identical init & hyper-parameters)

| training                                       | final train acc | final test acc |
|------------------------------------------------|-----------------|----------------|
| KM loss  ||M(x) - E_ii||^2                     | 0.7525          | 0.7380         |
| KM loss  off-class rows->0, true logit->1      | 1.0000          | 1.0000         |
| KM loss  cross-entropy on row norms            | 0.1730          | 0.1640         |
| KM loss  margin on row norms                   | 0.2065          | 0.2040         |
| vanilla  cross-entropy                         | 1.0000          | 1.0000         |

## Observations

- The differentiable knowledge matrix matches the library `KnowledgeMatrixComputer` exactly, so every KM loss is computed on the true M(W,f)(x).
- Chance level is 0.25. Ranking by test accuracy: off-class (1.00) > vanilla (1.00) > E_ii (0.74) > row-norm margin (0.20) > row-norm CE (0.16).
- `E_ii` is the most rigid target: it pins *every* entry of M(x) to a fixed sparse matrix; on MNIST-1D it barely clears chance (0.74).
- The `off-class` loss (wrong rows -> 0, true logit -> 1) frees the per-feature attribution and trains best among the KM losses (1.00).
- The row-norm losses only shape *which* class row carries the mass: cross-entropy on row norms (0.16) and a hinge margin on row norms (0.20). They are the loosest KM targets -- they never constrain the column sums (the actual output), only the row magnitudes.
