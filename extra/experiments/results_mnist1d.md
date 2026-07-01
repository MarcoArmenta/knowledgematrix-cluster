# Training a network via knowledge matrices

Data: MNIST-1D (github.com/greydanus/mnist1d): d=40, classes=10, n_train=4000, n_test=1000
Config: hidden=64, epochs=60, lr=0.001, batch_size=64, seed=0

Knowledge-matrix check vs library computer: abs_diff=0.000e+00, rel_diff=0.000e+00 (OK)

## Knowledge matrices  (loss = ||M(x) - E_ii||^2)
  [km_eii     ] epoch   1/60  loss=1.0238  train_acc=0.1215  test_acc=0.1250
  [km_eii     ] epoch   2/60  loss=0.9811  train_acc=0.1437  test_acc=0.1590
  [km_eii     ] epoch   3/60  loss=0.9644  train_acc=0.1455  test_acc=0.1600
  [km_eii     ] epoch   4/60  loss=0.9545  train_acc=0.1425  test_acc=0.1580
  [km_eii     ] epoch   5/60  loss=0.9486  train_acc=0.1423  test_acc=0.1600
  [km_eii     ] epoch   6/60  loss=0.9444  train_acc=0.1443  test_acc=0.1590
  [km_eii     ] epoch   7/60  loss=0.9423  train_acc=0.1425  test_acc=0.1540
  [km_eii     ] epoch   8/60  loss=0.9403  train_acc=0.1425  test_acc=0.1530
  [km_eii     ] epoch   9/60  loss=0.9384  train_acc=0.1465  test_acc=0.1510
  [km_eii     ] epoch  10/60  loss=0.9375  train_acc=0.1447  test_acc=0.1550
  [km_eii     ] epoch  11/60  loss=0.9364  train_acc=0.1440  test_acc=0.1530
  [km_eii     ] epoch  12/60  loss=0.9362  train_acc=0.1480  test_acc=0.1550
  [km_eii     ] epoch  13/60  loss=0.9357  train_acc=0.1500  test_acc=0.1500
  [km_eii     ] epoch  14/60  loss=0.9356  train_acc=0.1525  test_acc=0.1460
  [km_eii     ] epoch  15/60  loss=0.9342  train_acc=0.1552  test_acc=0.1580
  [km_eii     ] epoch  16/60  loss=0.9347  train_acc=0.1513  test_acc=0.1580
  [km_eii     ] epoch  17/60  loss=0.9336  train_acc=0.1535  test_acc=0.1540
  [km_eii     ] epoch  18/60  loss=0.9328  train_acc=0.1528  test_acc=0.1560
  [km_eii     ] epoch  19/60  loss=0.9320  train_acc=0.1513  test_acc=0.1590
  [km_eii     ] epoch  20/60  loss=0.9311  train_acc=0.1513  test_acc=0.1570
  [km_eii     ] epoch  21/60  loss=0.9306  train_acc=0.1550  test_acc=0.1570
  [km_eii     ] epoch  22/60  loss=0.9305  train_acc=0.1517  test_acc=0.1600
  [km_eii     ] epoch  23/60  loss=0.9301  train_acc=0.1485  test_acc=0.1620
  [km_eii     ] epoch  24/60  loss=0.9309  train_acc=0.1500  test_acc=0.1530
  [km_eii     ] epoch  25/60  loss=0.9301  train_acc=0.1495  test_acc=0.1540
  [km_eii     ] epoch  26/60  loss=0.9301  train_acc=0.1507  test_acc=0.1510
  [km_eii     ] epoch  27/60  loss=0.9302  train_acc=0.1478  test_acc=0.1550
  [km_eii     ] epoch  28/60  loss=0.9301  train_acc=0.1493  test_acc=0.1560
  [km_eii     ] epoch  29/60  loss=0.9297  train_acc=0.1488  test_acc=0.1520
  [km_eii     ] epoch  30/60  loss=0.9298  train_acc=0.1462  test_acc=0.1550
  [km_eii     ] epoch  31/60  loss=0.9298  train_acc=0.1488  test_acc=0.1570
  [km_eii     ] epoch  32/60  loss=0.9301  train_acc=0.1507  test_acc=0.1560
  [km_eii     ] epoch  33/60  loss=0.9297  train_acc=0.1513  test_acc=0.1550
  [km_eii     ] epoch  34/60  loss=0.9289  train_acc=0.1523  test_acc=0.1580
  [km_eii     ] epoch  35/60  loss=0.9298  train_acc=0.1515  test_acc=0.1640
  [km_eii     ] epoch  36/60  loss=0.9285  train_acc=0.1523  test_acc=0.1610
  [km_eii     ] epoch  37/60  loss=0.9293  train_acc=0.1532  test_acc=0.1610
  [km_eii     ] epoch  38/60  loss=0.9293  train_acc=0.1532  test_acc=0.1610
  [km_eii     ] epoch  39/60  loss=0.9291  train_acc=0.1513  test_acc=0.1580
  [km_eii     ] epoch  40/60  loss=0.9288  train_acc=0.1542  test_acc=0.1570
  [km_eii     ] epoch  41/60  loss=0.9291  train_acc=0.1505  test_acc=0.1610
  [km_eii     ] epoch  42/60  loss=0.9294  train_acc=0.1517  test_acc=0.1640
  [km_eii     ] epoch  43/60  loss=0.9291  train_acc=0.1550  test_acc=0.1610
  [km_eii     ] epoch  44/60  loss=0.9291  train_acc=0.1540  test_acc=0.1610
  [km_eii     ] epoch  45/60  loss=0.9287  train_acc=0.1548  test_acc=0.1610
  [km_eii     ] epoch  46/60  loss=0.9291  train_acc=0.1538  test_acc=0.1570
  [km_eii     ] epoch  47/60  loss=0.9296  train_acc=0.1532  test_acc=0.1640
  [km_eii     ] epoch  48/60  loss=0.9294  train_acc=0.1545  test_acc=0.1620
  [km_eii     ] epoch  49/60  loss=0.9299  train_acc=0.1555  test_acc=0.1640
  [km_eii     ] epoch  50/60  loss=0.9294  train_acc=0.1580  test_acc=0.1680
  [km_eii     ] epoch  51/60  loss=0.9296  train_acc=0.1532  test_acc=0.1610
  [km_eii     ] epoch  52/60  loss=0.9298  train_acc=0.1528  test_acc=0.1620
  [km_eii     ] epoch  53/60  loss=0.9305  train_acc=0.1540  test_acc=0.1550
  [km_eii     ] epoch  54/60  loss=0.9294  train_acc=0.1560  test_acc=0.1570
  [km_eii     ] epoch  55/60  loss=0.9304  train_acc=0.1525  test_acc=0.1610
  [km_eii     ] epoch  56/60  loss=0.9295  train_acc=0.1550  test_acc=0.1630
  [km_eii     ] epoch  57/60  loss=0.9298  train_acc=0.1515  test_acc=0.1570
  [km_eii     ] epoch  58/60  loss=0.9305  train_acc=0.1538  test_acc=0.1630
  [km_eii     ] epoch  59/60  loss=0.9306  train_acc=0.1558  test_acc=0.1650
  [km_eii     ] epoch  60/60  loss=0.9308  train_acc=0.1542  test_acc=0.1650

## Knowledge matrices  (loss = off-class rows -> 0, true logit -> 1)
  [km_offclass] epoch   1/60  loss=0.6187  train_acc=0.2375  test_acc=0.2030
  [km_offclass] epoch   2/60  loss=0.4261  train_acc=0.2567  test_acc=0.2260
  [km_offclass] epoch   3/60  loss=0.4041  train_acc=0.2788  test_acc=0.2300
  [km_offclass] epoch   4/60  loss=0.3907  train_acc=0.3220  test_acc=0.2750
  [km_offclass] epoch   5/60  loss=0.3805  train_acc=0.3243  test_acc=0.2640
  [km_offclass] epoch   6/60  loss=0.3727  train_acc=0.2993  test_acc=0.2680
  [km_offclass] epoch   7/60  loss=0.3646  train_acc=0.3217  test_acc=0.2880
  [km_offclass] epoch   8/60  loss=0.3563  train_acc=0.3810  test_acc=0.3090
  [km_offclass] epoch   9/60  loss=0.3478  train_acc=0.3760  test_acc=0.3340
  [km_offclass] epoch  10/60  loss=0.3402  train_acc=0.3618  test_acc=0.3060
  [km_offclass] epoch  11/60  loss=0.3332  train_acc=0.3565  test_acc=0.3090
  [km_offclass] epoch  12/60  loss=0.3265  train_acc=0.3555  test_acc=0.3120
  [km_offclass] epoch  13/60  loss=0.3203  train_acc=0.3582  test_acc=0.3130
  [km_offclass] epoch  14/60  loss=0.3155  train_acc=0.3755  test_acc=0.3290
  [km_offclass] epoch  15/60  loss=0.3097  train_acc=0.3825  test_acc=0.3290
  [km_offclass] epoch  16/60  loss=0.3048  train_acc=0.4128  test_acc=0.3520
  [km_offclass] epoch  17/60  loss=0.3012  train_acc=0.3848  test_acc=0.3430
  [km_offclass] epoch  18/60  loss=0.2961  train_acc=0.3860  test_acc=0.3540
  [km_offclass] epoch  19/60  loss=0.2935  train_acc=0.3470  test_acc=0.2990
  [km_offclass] epoch  20/60  loss=0.2892  train_acc=0.3740  test_acc=0.3340
  [km_offclass] epoch  21/60  loss=0.2870  train_acc=0.4193  test_acc=0.3520
  [km_offclass] epoch  22/60  loss=0.2841  train_acc=0.4115  test_acc=0.3600
  [km_offclass] epoch  23/60  loss=0.2820  train_acc=0.4017  test_acc=0.3530
  [km_offclass] epoch  24/60  loss=0.2800  train_acc=0.4390  test_acc=0.3750
  [km_offclass] epoch  25/60  loss=0.2777  train_acc=0.4098  test_acc=0.3750
  [km_offclass] epoch  26/60  loss=0.2767  train_acc=0.3808  test_acc=0.3110
  [km_offclass] epoch  27/60  loss=0.2761  train_acc=0.4005  test_acc=0.3600
  [km_offclass] epoch  28/60  loss=0.2743  train_acc=0.4030  test_acc=0.3580
  [km_offclass] epoch  29/60  loss=0.2735  train_acc=0.3890  test_acc=0.3140
  [km_offclass] epoch  30/60  loss=0.2718  train_acc=0.4280  test_acc=0.3780
  [km_offclass] epoch  31/60  loss=0.2712  train_acc=0.3840  test_acc=0.3470
  [km_offclass] epoch  32/60  loss=0.2704  train_acc=0.4047  test_acc=0.3540
  [km_offclass] epoch  33/60  loss=0.2696  train_acc=0.4965  test_acc=0.4090
  [km_offclass] epoch  34/60  loss=0.2691  train_acc=0.4263  test_acc=0.3630
  [km_offclass] epoch  35/60  loss=0.2686  train_acc=0.3983  test_acc=0.3580
  [km_offclass] epoch  36/60  loss=0.2675  train_acc=0.4363  test_acc=0.3880
  [km_offclass] epoch  37/60  loss=0.2677  train_acc=0.3983  test_acc=0.3530
  [km_offclass] epoch  38/60  loss=0.2676  train_acc=0.4223  test_acc=0.3610
  [km_offclass] epoch  39/60  loss=0.2667  train_acc=0.4255  test_acc=0.3670
  [km_offclass] epoch  40/60  loss=0.2666  train_acc=0.4218  test_acc=0.3610
  [km_offclass] epoch  41/60  loss=0.2663  train_acc=0.3620  test_acc=0.3360
  [km_offclass] epoch  42/60  loss=0.2665  train_acc=0.4212  test_acc=0.3620
  [km_offclass] epoch  43/60  loss=0.2658  train_acc=0.4345  test_acc=0.3740
  [km_offclass] epoch  44/60  loss=0.2655  train_acc=0.4460  test_acc=0.4020
  [km_offclass] epoch  45/60  loss=0.2643  train_acc=0.4200  test_acc=0.3760
  [km_offclass] epoch  46/60  loss=0.2648  train_acc=0.4552  test_acc=0.3870
  [km_offclass] epoch  47/60  loss=0.2633  train_acc=0.4750  test_acc=0.4200
  [km_offclass] epoch  48/60  loss=0.2638  train_acc=0.4248  test_acc=0.3600
  [km_offclass] epoch  49/60  loss=0.2628  train_acc=0.4080  test_acc=0.3610
  [km_offclass] epoch  50/60  loss=0.2634  train_acc=0.4518  test_acc=0.3980
  [km_offclass] epoch  51/60  loss=0.2631  train_acc=0.4790  test_acc=0.4300
  [km_offclass] epoch  52/60  loss=0.2631  train_acc=0.4358  test_acc=0.3800
  [km_offclass] epoch  53/60  loss=0.2629  train_acc=0.4780  test_acc=0.4270
  [km_offclass] epoch  54/60  loss=0.2615  train_acc=0.5038  test_acc=0.4340
  [km_offclass] epoch  55/60  loss=0.2615  train_acc=0.4330  test_acc=0.3900
  [km_offclass] epoch  56/60  loss=0.2614  train_acc=0.4178  test_acc=0.3680
  [km_offclass] epoch  57/60  loss=0.2610  train_acc=0.4848  test_acc=0.4360
  [km_offclass] epoch  58/60  loss=0.2604  train_acc=0.4642  test_acc=0.4020
  [km_offclass] epoch  59/60  loss=0.2605  train_acc=0.4717  test_acc=0.4040
  [km_offclass] epoch  60/60  loss=0.2600  train_acc=0.4347  test_acc=0.4020

## Vanilla             (loss = cross-entropy)
  [vanilla    ] epoch   1/60  loss=2.1496  train_acc=0.2113  test_acc=0.2050
  [vanilla    ] epoch   2/60  loss=1.7857  train_acc=0.3113  test_acc=0.2580
  [vanilla    ] epoch   3/60  loss=1.6662  train_acc=0.3680  test_acc=0.3020
  [vanilla    ] epoch   4/60  loss=1.5724  train_acc=0.3985  test_acc=0.3490
  [vanilla    ] epoch   5/60  loss=1.4830  train_acc=0.4570  test_acc=0.3880
  [vanilla    ] epoch   6/60  loss=1.4016  train_acc=0.4918  test_acc=0.4310
  [vanilla    ] epoch   7/60  loss=1.3385  train_acc=0.5058  test_acc=0.4410
  [vanilla    ] epoch   8/60  loss=1.2863  train_acc=0.5345  test_acc=0.4650
  [vanilla    ] epoch   9/60  loss=1.2472  train_acc=0.5587  test_acc=0.4910
  [vanilla    ] epoch  10/60  loss=1.2143  train_acc=0.5527  test_acc=0.4950
  [vanilla    ] epoch  11/60  loss=1.1889  train_acc=0.5740  test_acc=0.4960
  [vanilla    ] epoch  12/60  loss=1.1634  train_acc=0.5782  test_acc=0.5100
  [vanilla    ] epoch  13/60  loss=1.1404  train_acc=0.5870  test_acc=0.5110
  [vanilla    ] epoch  14/60  loss=1.1197  train_acc=0.5880  test_acc=0.5160
  [vanilla    ] epoch  15/60  loss=1.0959  train_acc=0.6055  test_acc=0.5200
  [vanilla    ] epoch  16/60  loss=1.0780  train_acc=0.6075  test_acc=0.5270
  [vanilla    ] epoch  17/60  loss=1.0569  train_acc=0.6120  test_acc=0.5290
  [vanilla    ] epoch  18/60  loss=1.0389  train_acc=0.6290  test_acc=0.5450
  [vanilla    ] epoch  19/60  loss=1.0217  train_acc=0.6305  test_acc=0.5250
  [vanilla    ] epoch  20/60  loss=1.0013  train_acc=0.6385  test_acc=0.5450
  [vanilla    ] epoch  21/60  loss=0.9846  train_acc=0.6470  test_acc=0.5460
  [vanilla    ] epoch  22/60  loss=0.9685  train_acc=0.6610  test_acc=0.5500
  [vanilla    ] epoch  23/60  loss=0.9440  train_acc=0.6580  test_acc=0.5470
  [vanilla    ] epoch  24/60  loss=0.9294  train_acc=0.6697  test_acc=0.5690
  [vanilla    ] epoch  25/60  loss=0.9108  train_acc=0.6747  test_acc=0.5570
  [vanilla    ] epoch  26/60  loss=0.8976  train_acc=0.6862  test_acc=0.5580
  [vanilla    ] epoch  27/60  loss=0.8871  train_acc=0.6910  test_acc=0.5540
  [vanilla    ] epoch  28/60  loss=0.8693  train_acc=0.6900  test_acc=0.5530
  [vanilla    ] epoch  29/60  loss=0.8565  train_acc=0.7007  test_acc=0.5590
  [vanilla    ] epoch  30/60  loss=0.8421  train_acc=0.7042  test_acc=0.5620
  [vanilla    ] epoch  31/60  loss=0.8285  train_acc=0.7095  test_acc=0.5730
  [vanilla    ] epoch  32/60  loss=0.8200  train_acc=0.7178  test_acc=0.5550
  [vanilla    ] epoch  33/60  loss=0.8071  train_acc=0.7290  test_acc=0.5530
  [vanilla    ] epoch  34/60  loss=0.7927  train_acc=0.7222  test_acc=0.5570
  [vanilla    ] epoch  35/60  loss=0.7842  train_acc=0.7325  test_acc=0.5510
  [vanilla    ] epoch  36/60  loss=0.7737  train_acc=0.7355  test_acc=0.5590
  [vanilla    ] epoch  37/60  loss=0.7637  train_acc=0.7318  test_acc=0.5600
  [vanilla    ] epoch  38/60  loss=0.7547  train_acc=0.7372  test_acc=0.5650
  [vanilla    ] epoch  39/60  loss=0.7445  train_acc=0.7305  test_acc=0.5720
  [vanilla    ] epoch  40/60  loss=0.7367  train_acc=0.7525  test_acc=0.5680
  [vanilla    ] epoch  41/60  loss=0.7247  train_acc=0.7520  test_acc=0.5760
  [vanilla    ] epoch  42/60  loss=0.7192  train_acc=0.7535  test_acc=0.5640
  [vanilla    ] epoch  43/60  loss=0.7121  train_acc=0.7555  test_acc=0.5650
  [vanilla    ] epoch  44/60  loss=0.7015  train_acc=0.7620  test_acc=0.5650
  [vanilla    ] epoch  45/60  loss=0.6944  train_acc=0.7635  test_acc=0.5620
  [vanilla    ] epoch  46/60  loss=0.6856  train_acc=0.7548  test_acc=0.5760
  [vanilla    ] epoch  47/60  loss=0.6751  train_acc=0.7717  test_acc=0.5810
  [vanilla    ] epoch  48/60  loss=0.6651  train_acc=0.7747  test_acc=0.5710
  [vanilla    ] epoch  49/60  loss=0.6598  train_acc=0.7750  test_acc=0.5670
  [vanilla    ] epoch  50/60  loss=0.6552  train_acc=0.7800  test_acc=0.5870
  [vanilla    ] epoch  51/60  loss=0.6458  train_acc=0.7897  test_acc=0.5710
  [vanilla    ] epoch  52/60  loss=0.6378  train_acc=0.7805  test_acc=0.5760
  [vanilla    ] epoch  53/60  loss=0.6375  train_acc=0.7887  test_acc=0.5550
  [vanilla    ] epoch  54/60  loss=0.6254  train_acc=0.7962  test_acc=0.5680
  [vanilla    ] epoch  55/60  loss=0.6168  train_acc=0.7965  test_acc=0.5760
  [vanilla    ] epoch  56/60  loss=0.6081  train_acc=0.8035  test_acc=0.5780
  [vanilla    ] epoch  57/60  loss=0.6057  train_acc=0.8037  test_acc=0.5670
  [vanilla    ] epoch  58/60  loss=0.5976  train_acc=0.8033  test_acc=0.5660
  [vanilla    ] epoch  59/60  loss=0.5882  train_acc=0.8033  test_acc=0.5790
  [vanilla    ] epoch  60/60  loss=0.5816  train_acc=0.8052  test_acc=0.5660

## Final comparison (identical init & hyper-parameters)

| training                                    | final train acc | final test acc |
|---------------------------------------------|-----------------|----------------|
| KM loss  ||M(x) - E_ii||^2                  | 0.1542          | 0.1650         |
| KM loss  off-class rows->0, true logit->1   | 0.4347          | 0.4020         |
| vanilla  cross-entropy                      | 0.8052          | 0.5660         |

## Observations

- The differentiable knowledge matrix matches the library `KnowledgeMatrixComputer` exactly, so every KM loss is computed on the true M(W,f)(x).
- Chance level is 0.10. Ranking by test accuracy: vanilla (0.57) > off-class KM (0.40) > E_ii KM (0.17).
- `E_ii` is the most rigid target: it pins *every* entry of M(x) to a fixed sparse matrix. On the harder MNIST-1D task this is a very stiff objective and it barely clears chance (0.17 vs 0.10).
- The recommended `off-class` loss relaxes this -- it only forces the wrong-class rows to vanish and the true logit to 1, leaving the per-feature attribution free. That larger solution set trains far better (0.40), closing much of the gap to cross-entropy (0.57) while still being a genuine loss on the knowledge matrix.
