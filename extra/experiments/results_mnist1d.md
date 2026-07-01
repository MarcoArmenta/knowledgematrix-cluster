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

## Knowledge matrices  (loss = cross-entropy on row norms)
  [km_rownorm_ce] epoch   1/60  loss=2.2739  train_acc=0.1782  test_acc=0.1940
  [km_rownorm_ce] epoch   2/60  loss=2.1240  train_acc=0.1875  test_acc=0.2010
  [km_rownorm_ce] epoch   3/60  loss=1.9245  train_acc=0.2087  test_acc=0.2280
  [km_rownorm_ce] epoch   4/60  loss=1.7742  train_acc=0.2173  test_acc=0.2320
  [km_rownorm_ce] epoch   5/60  loss=1.6901  train_acc=0.2215  test_acc=0.2210
  [km_rownorm_ce] epoch   6/60  loss=1.6270  train_acc=0.2205  test_acc=0.2130
  [km_rownorm_ce] epoch   7/60  loss=1.5729  train_acc=0.2188  test_acc=0.2230
  [km_rownorm_ce] epoch   8/60  loss=1.5179  train_acc=0.2227  test_acc=0.2280
  [km_rownorm_ce] epoch   9/60  loss=1.4723  train_acc=0.2190  test_acc=0.2220
  [km_rownorm_ce] epoch  10/60  loss=1.4285  train_acc=0.2323  test_acc=0.2410
  [km_rownorm_ce] epoch  11/60  loss=1.3919  train_acc=0.2215  test_acc=0.2300
  [km_rownorm_ce] epoch  12/60  loss=1.3547  train_acc=0.2327  test_acc=0.2330
  [km_rownorm_ce] epoch  13/60  loss=1.3151  train_acc=0.2335  test_acc=0.2240
  [km_rownorm_ce] epoch  14/60  loss=1.2843  train_acc=0.2345  test_acc=0.2310
  [km_rownorm_ce] epoch  15/60  loss=1.2483  train_acc=0.2373  test_acc=0.2380
  [km_rownorm_ce] epoch  16/60  loss=1.2198  train_acc=0.2368  test_acc=0.2330
  [km_rownorm_ce] epoch  17/60  loss=1.1934  train_acc=0.2380  test_acc=0.2390
  [km_rownorm_ce] epoch  18/60  loss=1.1687  train_acc=0.2387  test_acc=0.2380
  [km_rownorm_ce] epoch  19/60  loss=1.1392  train_acc=0.2323  test_acc=0.2280
  [km_rownorm_ce] epoch  20/60  loss=1.1137  train_acc=0.2325  test_acc=0.2310
  [km_rownorm_ce] epoch  21/60  loss=1.0935  train_acc=0.2432  test_acc=0.2360
  [km_rownorm_ce] epoch  22/60  loss=1.0708  train_acc=0.2355  test_acc=0.2370
  [km_rownorm_ce] epoch  23/60  loss=1.0508  train_acc=0.2428  test_acc=0.2390
  [km_rownorm_ce] epoch  24/60  loss=1.0330  train_acc=0.2488  test_acc=0.2500
  [km_rownorm_ce] epoch  25/60  loss=1.0117  train_acc=0.2470  test_acc=0.2460
  [km_rownorm_ce] epoch  26/60  loss=0.9911  train_acc=0.2415  test_acc=0.2420
  [km_rownorm_ce] epoch  27/60  loss=0.9808  train_acc=0.2477  test_acc=0.2510
  [km_rownorm_ce] epoch  28/60  loss=0.9713  train_acc=0.2403  test_acc=0.2310
  [km_rownorm_ce] epoch  29/60  loss=0.9482  train_acc=0.2407  test_acc=0.2320
  [km_rownorm_ce] epoch  30/60  loss=0.9365  train_acc=0.2477  test_acc=0.2490
  [km_rownorm_ce] epoch  31/60  loss=0.9217  train_acc=0.2525  test_acc=0.2510
  [km_rownorm_ce] epoch  32/60  loss=0.9080  train_acc=0.2535  test_acc=0.2550
  [km_rownorm_ce] epoch  33/60  loss=0.8958  train_acc=0.2517  test_acc=0.2510
  [km_rownorm_ce] epoch  34/60  loss=0.8792  train_acc=0.2545  test_acc=0.2600
  [km_rownorm_ce] epoch  35/60  loss=0.8705  train_acc=0.2540  test_acc=0.2480
  [km_rownorm_ce] epoch  36/60  loss=0.8621  train_acc=0.2595  test_acc=0.2560
  [km_rownorm_ce] epoch  37/60  loss=0.8495  train_acc=0.2505  test_acc=0.2450
  [km_rownorm_ce] epoch  38/60  loss=0.8369  train_acc=0.2540  test_acc=0.2520
  [km_rownorm_ce] epoch  39/60  loss=0.8252  train_acc=0.2553  test_acc=0.2490
  [km_rownorm_ce] epoch  40/60  loss=0.8248  train_acc=0.2650  test_acc=0.2590
  [km_rownorm_ce] epoch  41/60  loss=0.8109  train_acc=0.2623  test_acc=0.2570
  [km_rownorm_ce] epoch  42/60  loss=0.8117  train_acc=0.2575  test_acc=0.2510
  [km_rownorm_ce] epoch  43/60  loss=0.7993  train_acc=0.2657  test_acc=0.2600
  [km_rownorm_ce] epoch  44/60  loss=0.7913  train_acc=0.2600  test_acc=0.2550
  [km_rownorm_ce] epoch  45/60  loss=0.7824  train_acc=0.2545  test_acc=0.2530
  [km_rownorm_ce] epoch  46/60  loss=0.7821  train_acc=0.2595  test_acc=0.2550
  [km_rownorm_ce] epoch  47/60  loss=0.7771  train_acc=0.2605  test_acc=0.2610
  [km_rownorm_ce] epoch  48/60  loss=0.7684  train_acc=0.2562  test_acc=0.2550
  [km_rownorm_ce] epoch  49/60  loss=0.7611  train_acc=0.2592  test_acc=0.2580
  [km_rownorm_ce] epoch  50/60  loss=0.7602  train_acc=0.2520  test_acc=0.2470
  [km_rownorm_ce] epoch  51/60  loss=0.7510  train_acc=0.2585  test_acc=0.2620
  [km_rownorm_ce] epoch  52/60  loss=0.7416  train_acc=0.2500  test_acc=0.2490
  [km_rownorm_ce] epoch  53/60  loss=0.7431  train_acc=0.2668  test_acc=0.2650
  [km_rownorm_ce] epoch  54/60  loss=0.7294  train_acc=0.2578  test_acc=0.2540
  [km_rownorm_ce] epoch  55/60  loss=0.7293  train_acc=0.2700  test_acc=0.2690
  [km_rownorm_ce] epoch  56/60  loss=0.7286  train_acc=0.2607  test_acc=0.2570
  [km_rownorm_ce] epoch  57/60  loss=0.7151  train_acc=0.2680  test_acc=0.2690
  [km_rownorm_ce] epoch  58/60  loss=0.7057  train_acc=0.2637  test_acc=0.2670
  [km_rownorm_ce] epoch  59/60  loss=0.7040  train_acc=0.2612  test_acc=0.2670
  [km_rownorm_ce] epoch  60/60  loss=0.6984  train_acc=0.2657  test_acc=0.2690

## Knowledge matrices  (loss = margin on row norms)
  [km_rownorm_margin] epoch   1/60  loss=1.0186  train_acc=0.1245  test_acc=0.1310
  [km_rownorm_margin] epoch   2/60  loss=1.0016  train_acc=0.1287  test_acc=0.1420
  [km_rownorm_margin] epoch   3/60  loss=0.9931  train_acc=0.1320  test_acc=0.1270
  [km_rownorm_margin] epoch   4/60  loss=0.9852  train_acc=0.1295  test_acc=0.1330
  [km_rownorm_margin] epoch   5/60  loss=0.9759  train_acc=0.1415  test_acc=0.1430
  [km_rownorm_margin] epoch   6/60  loss=0.9599  train_acc=0.1417  test_acc=0.1490
  [km_rownorm_margin] epoch   7/60  loss=0.9388  train_acc=0.1447  test_acc=0.1490
  [km_rownorm_margin] epoch   8/60  loss=0.9092  train_acc=0.1462  test_acc=0.1540
  [km_rownorm_margin] epoch   9/60  loss=0.8884  train_acc=0.1475  test_acc=0.1470
  [km_rownorm_margin] epoch  10/60  loss=0.8729  train_acc=0.1520  test_acc=0.1580
  [km_rownorm_margin] epoch  11/60  loss=0.8627  train_acc=0.1593  test_acc=0.1650
  [km_rownorm_margin] epoch  12/60  loss=0.8491  train_acc=0.1723  test_acc=0.1740
  [km_rownorm_margin] epoch  13/60  loss=0.8318  train_acc=0.1787  test_acc=0.1860
  [km_rownorm_margin] epoch  14/60  loss=0.8219  train_acc=0.1883  test_acc=0.2040
  [km_rownorm_margin] epoch  15/60  loss=0.8094  train_acc=0.1928  test_acc=0.2070
  [km_rownorm_margin] epoch  16/60  loss=0.7954  train_acc=0.2000  test_acc=0.2100
  [km_rownorm_margin] epoch  17/60  loss=0.7866  train_acc=0.2062  test_acc=0.2020
  [km_rownorm_margin] epoch  18/60  loss=0.7721  train_acc=0.2142  test_acc=0.2120
  [km_rownorm_margin] epoch  19/60  loss=0.7588  train_acc=0.2065  test_acc=0.2040
  [km_rownorm_margin] epoch  20/60  loss=0.7446  train_acc=0.2247  test_acc=0.2280
  [km_rownorm_margin] epoch  21/60  loss=0.7348  train_acc=0.2225  test_acc=0.2260
  [km_rownorm_margin] epoch  22/60  loss=0.7290  train_acc=0.2132  test_acc=0.2130
  [km_rownorm_margin] epoch  23/60  loss=0.7246  train_acc=0.2275  test_acc=0.2250
  [km_rownorm_margin] epoch  24/60  loss=0.7203  train_acc=0.2130  test_acc=0.2310
  [km_rownorm_margin] epoch  25/60  loss=0.7091  train_acc=0.2240  test_acc=0.2310
  [km_rownorm_margin] epoch  26/60  loss=0.7053  train_acc=0.2335  test_acc=0.2250
  [km_rownorm_margin] epoch  27/60  loss=0.6999  train_acc=0.2380  test_acc=0.2420
  [km_rownorm_margin] epoch  28/60  loss=0.6971  train_acc=0.2163  test_acc=0.2280
  [km_rownorm_margin] epoch  29/60  loss=0.6914  train_acc=0.2250  test_acc=0.2180
  [km_rownorm_margin] epoch  30/60  loss=0.6883  train_acc=0.2233  test_acc=0.2140
  [km_rownorm_margin] epoch  31/60  loss=0.6860  train_acc=0.2323  test_acc=0.2240
  [km_rownorm_margin] epoch  32/60  loss=0.6837  train_acc=0.2288  test_acc=0.2220
  [km_rownorm_margin] epoch  33/60  loss=0.6797  train_acc=0.2097  test_acc=0.2110
  [km_rownorm_margin] epoch  34/60  loss=0.6764  train_acc=0.2175  test_acc=0.2100
  [km_rownorm_margin] epoch  35/60  loss=0.6742  train_acc=0.2180  test_acc=0.2150
  [km_rownorm_margin] epoch  36/60  loss=0.6716  train_acc=0.2240  test_acc=0.2310
  [km_rownorm_margin] epoch  37/60  loss=0.6688  train_acc=0.2095  test_acc=0.1940
  [km_rownorm_margin] epoch  38/60  loss=0.6637  train_acc=0.1960  test_acc=0.1950
  [km_rownorm_margin] epoch  39/60  loss=0.6633  train_acc=0.2062  test_acc=0.2070
  [km_rownorm_margin] epoch  40/60  loss=0.6613  train_acc=0.2023  test_acc=0.2050
  [km_rownorm_margin] epoch  41/60  loss=0.6545  train_acc=0.1867  test_acc=0.1730
  [km_rownorm_margin] epoch  42/60  loss=0.6491  train_acc=0.1890  test_acc=0.1840
  [km_rownorm_margin] epoch  43/60  loss=0.6480  train_acc=0.2093  test_acc=0.2000
  [km_rownorm_margin] epoch  44/60  loss=0.6450  train_acc=0.2030  test_acc=0.1980
  [km_rownorm_margin] epoch  45/60  loss=0.6387  train_acc=0.1895  test_acc=0.1800
  [km_rownorm_margin] epoch  46/60  loss=0.6345  train_acc=0.1743  test_acc=0.1870
  [km_rownorm_margin] epoch  47/60  loss=0.6336  train_acc=0.1793  test_acc=0.1640
  [km_rownorm_margin] epoch  48/60  loss=0.6286  train_acc=0.1778  test_acc=0.1850
  [km_rownorm_margin] epoch  49/60  loss=0.6270  train_acc=0.1657  test_acc=0.1760
  [km_rownorm_margin] epoch  50/60  loss=0.6237  train_acc=0.1810  test_acc=0.1860
  [km_rownorm_margin] epoch  51/60  loss=0.6230  train_acc=0.1790  test_acc=0.1780
  [km_rownorm_margin] epoch  52/60  loss=0.6190  train_acc=0.1695  test_acc=0.1760
  [km_rownorm_margin] epoch  53/60  loss=0.6147  train_acc=0.1875  test_acc=0.1850
  [km_rownorm_margin] epoch  54/60  loss=0.6139  train_acc=0.1758  test_acc=0.1700
  [km_rownorm_margin] epoch  55/60  loss=0.6097  train_acc=0.1793  test_acc=0.1670
  [km_rownorm_margin] epoch  56/60  loss=0.6084  train_acc=0.1778  test_acc=0.1770
  [km_rownorm_margin] epoch  57/60  loss=0.6064  train_acc=0.1762  test_acc=0.1760
  [km_rownorm_margin] epoch  58/60  loss=0.6040  train_acc=0.1745  test_acc=0.1800
  [km_rownorm_margin] epoch  59/60  loss=0.6024  train_acc=0.1768  test_acc=0.1850
  [km_rownorm_margin] epoch  60/60  loss=0.6007  train_acc=0.1720  test_acc=0.1720

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

| training                                       | final train acc | final test acc |
|------------------------------------------------|-----------------|----------------|
| KM loss  ||M(x) - E_ii||^2                     | 0.1542          | 0.1650         |
| KM loss  off-class rows->0, true logit->1      | 0.4347          | 0.4020         |
| KM loss  cross-entropy on row norms            | 0.2657          | 0.2690         |
| KM loss  margin on row norms                   | 0.1720          | 0.1720         |
| vanilla  cross-entropy                         | 0.8052          | 0.5660         |

## Observations

- The differentiable knowledge matrix matches the library `KnowledgeMatrixComputer` exactly, so every KM loss is computed on the true M(W,f)(x).
- Chance level is 0.10. Ranking by test accuracy: vanilla (0.57) > off-class (0.40) > row-norm CE (0.27) > row-norm margin (0.17) > E_ii (0.17).
- `E_ii` is the most rigid target: it pins *every* entry of M(x) to a fixed sparse matrix; on MNIST-1D it barely clears chance (0.17).
- The `off-class` loss (wrong rows -> 0, true logit -> 1) frees the per-feature attribution and trains best among the KM losses (0.40).
- The row-norm losses only shape *which* class row carries the mass: cross-entropy on row norms (0.27) and a hinge margin on row norms (0.17). They are the loosest KM targets -- they never constrain the column sums (the actual output), only the row magnitudes.
