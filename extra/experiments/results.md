# Training a network via knowledge matrices

Config: d=16, classes=4, hidden=64, n_train=2000, n_test=500, epochs=40, lr=0.001, batch_size=64, seed=0

Knowledge-matrix check vs library computer: abs_diff=0.000e+00, rel_diff=0.000e+00 (OK)

## Knowledge-matrix training  (loss = ||M(x) - E_ii||^2)
  [km     ] epoch   1/40  loss=0.8668  train_acc=0.7285  test_acc=0.7160
  [km     ] epoch   2/40  loss=0.5386  train_acc=0.8590  test_acc=0.8360
  [km     ] epoch   3/40  loss=0.4436  train_acc=0.8455  test_acc=0.8240
  [km     ] epoch   4/40  loss=0.3967  train_acc=0.8585  test_acc=0.8520
  [km     ] epoch   5/40  loss=0.3683  train_acc=0.8950  test_acc=0.8680
  [km     ] epoch   6/40  loss=0.3664  train_acc=0.8755  test_acc=0.8540
  [km     ] epoch   7/40  loss=0.3860  train_acc=0.8495  test_acc=0.8320
  [km     ] epoch   8/40  loss=0.3970  train_acc=0.7965  test_acc=0.7820
  [km     ] epoch   9/40  loss=0.4163  train_acc=0.7600  test_acc=0.7400
  [km     ] epoch  10/40  loss=0.4159  train_acc=0.7970  test_acc=0.7840
  [km     ] epoch  11/40  loss=0.4049  train_acc=0.8425  test_acc=0.8360
  [km     ] epoch  12/40  loss=0.4285  train_acc=0.8055  test_acc=0.8160
  [km     ] epoch  13/40  loss=0.4448  train_acc=0.7315  test_acc=0.7440
  [km     ] epoch  14/40  loss=0.4677  train_acc=0.8220  test_acc=0.8380
  [km     ] epoch  15/40  loss=0.4832  train_acc=0.8380  test_acc=0.8560
  [km     ] epoch  16/40  loss=0.4827  train_acc=0.8380  test_acc=0.8460
  [km     ] epoch  17/40  loss=0.4708  train_acc=0.7800  test_acc=0.7840
  [km     ] epoch  18/40  loss=0.4672  train_acc=0.8020  test_acc=0.8100
  [km     ] epoch  19/40  loss=0.4556  train_acc=0.7755  test_acc=0.7500
  [km     ] epoch  20/40  loss=0.4452  train_acc=0.7750  test_acc=0.7600
  [km     ] epoch  21/40  loss=0.4473  train_acc=0.7630  test_acc=0.7440
  [km     ] epoch  22/40  loss=0.4620  train_acc=0.7575  test_acc=0.7380
  [km     ] epoch  23/40  loss=0.4799  train_acc=0.7715  test_acc=0.7620
  [km     ] epoch  24/40  loss=0.4908  train_acc=0.7435  test_acc=0.7320
  [km     ] epoch  25/40  loss=0.4837  train_acc=0.7310  test_acc=0.7140
  [km     ] epoch  26/40  loss=0.4631  train_acc=0.7920  test_acc=0.7920
  [km     ] epoch  27/40  loss=0.4519  train_acc=0.7880  test_acc=0.7700
  [km     ] epoch  28/40  loss=0.4746  train_acc=0.8140  test_acc=0.8240
  [km     ] epoch  29/40  loss=0.4818  train_acc=0.7965  test_acc=0.7920
  [km     ] epoch  30/40  loss=0.4993  train_acc=0.7650  test_acc=0.7440
  [km     ] epoch  31/40  loss=0.5209  train_acc=0.7865  test_acc=0.7860
  [km     ] epoch  32/40  loss=0.5143  train_acc=0.7370  test_acc=0.7360
  [km     ] epoch  33/40  loss=0.4898  train_acc=0.7250  test_acc=0.7180
  [km     ] epoch  34/40  loss=0.4796  train_acc=0.7560  test_acc=0.7540
  [km     ] epoch  35/40  loss=0.4811  train_acc=0.7340  test_acc=0.7240
  [km     ] epoch  36/40  loss=0.4807  train_acc=0.7565  test_acc=0.7420
  [km     ] epoch  37/40  loss=0.4852  train_acc=0.7635  test_acc=0.7520
  [km     ] epoch  38/40  loss=0.4921  train_acc=0.7435  test_acc=0.7360
  [km     ] epoch  39/40  loss=0.5018  train_acc=0.7300  test_acc=0.7200
  [km     ] epoch  40/40  loss=0.4992  train_acc=0.7525  test_acc=0.7380

## Vanilla training  (loss = cross-entropy)
  [vanilla] epoch   1/40  loss=0.5886  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   2/40  loss=0.0277  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   3/40  loss=0.0053  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   4/40  loss=0.0029  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   5/40  loss=0.0019  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   6/40  loss=0.0013  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   7/40  loss=0.0010  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   8/40  loss=0.0007  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch   9/40  loss=0.0005  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  10/40  loss=0.0004  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  11/40  loss=0.0003  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  12/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  13/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  14/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  15/40  loss=0.0002  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  16/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  17/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  18/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  19/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  20/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  21/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  22/40  loss=0.0001  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  23/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  24/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  25/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  26/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  27/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  28/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  29/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  30/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  31/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  32/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  33/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  34/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  35/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  36/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  37/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  38/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  39/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000
  [vanilla] epoch  40/40  loss=0.0000  train_acc=1.0000  test_acc=1.0000

## Final comparison (identical init & hyper-parameters)

| training            | final train acc | final test acc |
|---------------------|-----------------|----------------|
| knowledge matrices  | 0.7525          | 0.7380         |
| vanilla (cross-ent) | 1.0000          | 1.0000         |

## Observations

- The differentiable knowledge matrix matches the library `KnowledgeMatrixComputer` exactly, so the loss is computed on the true M(W,f)(x).
- Training via `||M(x) - E_ii||^2` does learn the task (test acc 0.74 >> 0.25 chance) and generalizes (train 0.75 vs test 0.74).
- It is, however, a much harder optimization target than cross-entropy and plateaus below it: E_ii pins down *every* entry of the local affine decomposition (the whole matrix must become a fixed sparse matrix), whereas cross-entropy only constrains the column sums (the output). With identical init and hyper-parameters, vanilla reaches 1.00 test accuracy.
