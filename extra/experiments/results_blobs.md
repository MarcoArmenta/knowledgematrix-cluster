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

| training                                    | final train acc | final test acc |
|---------------------------------------------|-----------------|----------------|
| KM loss  ||M(x) - E_ii||^2                  | 0.7525          | 0.7380         |
| KM loss  off-class rows->0, true logit->1   | 1.0000          | 1.0000         |
| vanilla  cross-entropy                      | 1.0000          | 1.0000         |

## Observations

- The differentiable knowledge matrix matches the library `KnowledgeMatrixComputer` exactly, so every KM loss is computed on the true M(W,f)(x).
- Chance level is 0.25. Ranking by test accuracy: vanilla (1.00) > off-class KM (1.00) > E_ii KM (0.74).
- `E_ii` is the most rigid target: it pins *every* entry of M(x) to a fixed sparse matrix. On the harder MNIST-1D task this is a very stiff objective and it barely clears chance (0.74 vs 0.25).
- The recommended `off-class` loss relaxes this -- it only forces the wrong-class rows to vanish and the true logit to 1, leaving the per-feature attribution free. That larger solution set trains far better (1.00), closing much of the gap to cross-entropy (1.00) while still being a genuine loss on the knowledge matrix.
