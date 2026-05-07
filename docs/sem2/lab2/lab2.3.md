# Отчет по 3 заданию

В эксперименте transfer learning модель сначала обучалась на 8 классах CIFAR-10, затем сверточная часть была заморожена,
а полносвязная часть дообучена на 10 классах. При этом точность на исключенных классах оказалась ниже, чем у базовой модели,
обученной сразу на 10 классах (airplane: -2.1 п.п., automobile: -6.2 п.п.). Следовательно,
в данной конфигурации transfer learning не дал прироста качества для ранее исключенных классов,
хотя общая сходимость модели осталась стабильной.

```sh
(ml-labs) the80hz@MacBook-Pro-Daniil ml-labs % uv run '/Users/the80hz/Developer/ml-labs/scripts/sem2/run_lab2.py'
/Users/the80hz/Developer/ml-labs/.venv/lib/python3.13/site-packages/keras/src/datasets/cifar.py:18: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  d = cPickle.load(f, encoding="bytes")
Train sample length: 50000
Test sample length: 10000

Train classes distribution:
Class 0 (airplane): 5000 (10.00%)
Class 1 (automobile): 5000 (10.00%)
Class 2 (bird): 5000 (10.00%)
Class 3 (cat): 5000 (10.00%)
Class 4 (deer): 5000 (10.00%)
Class 5 (dog): 5000 (10.00%)
Class 6 (frog): 5000 (10.00%)
Class 7 (horse): 5000 (10.00%)
Class 8 (ship): 5000 (10.00%)
Class 9 (truck): 5000 (10.00%)

Transfer learning mode...
Excluded classes on stage 1: [0, 1] -> ['airplane', 'automobile']

Stage 1/2: train base model on 8 classes...
Epoch 1/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 16ms/step - accuracy: 0.4525 - loss: 1.4607 - val_accuracy: 0.5471 - val_loss: 1.2322
Epoch 2/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.5744 - loss: 1.1654 - val_accuracy: 0.6285 - val_loss: 1.0255
Epoch 3/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.6261 - loss: 1.0376 - val_accuracy: 0.6614 - val_loss: 0.9343
Epoch 4/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 19s 19ms/step - accuracy: 0.6538 - loss: 0.9665 - val_accuracy: 0.6826 - val_loss: 0.8836
Epoch 5/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.6733 - loss: 0.9087 - val_accuracy: 0.7210 - val_loss: 0.7781
Epoch 6/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.6910 - loss: 0.8660 - val_accuracy: 0.7241 - val_loss: 0.7761
Epoch 7/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7020 - loss: 0.8323 - val_accuracy: 0.7244 - val_loss: 0.7625
Epoch 8/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7146 - loss: 0.8001 - val_accuracy: 0.7390 - val_loss: 0.7293
Epoch 9/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 16s 16ms/step - accuracy: 0.7223 - loss: 0.7781 - val_accuracy: 0.7461 - val_loss: 0.7228
Epoch 10/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7300 - loss: 0.7545 - val_accuracy: 0.7542 - val_loss: 0.7092
Epoch 11/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7393 - loss: 0.7298 - val_accuracy: 0.7469 - val_loss: 0.7055
Epoch 12/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 16s 16ms/step - accuracy: 0.7437 - loss: 0.7200 - val_accuracy: 0.7423 - val_loss: 0.7165
Epoch 13/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 16s 16ms/step - accuracy: 0.7508 - loss: 0.6967 - val_accuracy: 0.7517 - val_loss: 0.7016
Epoch 14/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7567 - loss: 0.6875 - val_accuracy: 0.7651 - val_loss: 0.6675
Epoch 15/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7594 - loss: 0.6686 - val_accuracy: 0.7646 - val_loss: 0.6685
Epoch 16/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7646 - loss: 0.6572 - val_accuracy: 0.7644 - val_loss: 0.6688
Epoch 17/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7682 - loss: 0.6441 - val_accuracy: 0.7667 - val_loss: 0.6619
Epoch 18/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7724 - loss: 0.6337 - val_accuracy: 0.7704 - val_loss: 0.6552
Epoch 19/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7745 - loss: 0.6344 - val_accuracy: 0.7713 - val_loss: 0.6460
Epoch 20/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7784 - loss: 0.6177 - val_accuracy: 0.7728 - val_loss: 0.6493
Epoch 21/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7842 - loss: 0.6078 - val_accuracy: 0.7619 - val_loss: 0.6838
Epoch 22/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7833 - loss: 0.6043 - val_accuracy: 0.7714 - val_loss: 0.6536
Epoch 23/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7880 - loss: 0.5890 - val_accuracy: 0.7666 - val_loss: 0.6721
Epoch 24/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7922 - loss: 0.5794 - val_accuracy: 0.7619 - val_loss: 0.6803
Epoch 25/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 17s 17ms/step - accuracy: 0.7913 - loss: 0.5779 - val_accuracy: 0.7751 - val_loss: 0.6424
Epoch 26/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7963 - loss: 0.5706 - val_accuracy: 0.7797 - val_loss: 0.6375
Epoch 27/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.7973 - loss: 0.5646 - val_accuracy: 0.7780 - val_loss: 0.6325
Epoch 28/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.8018 - loss: 0.5554 - val_accuracy: 0.7754 - val_loss: 0.6283
Epoch 29/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.8012 - loss: 0.5564 - val_accuracy: 0.7750 - val_loss: 0.6541
Epoch 30/30
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 18s 18ms/step - accuracy: 0.8029 - loss: 0.5482 - val_accuracy: 0.7835 - val_loss: 0.6226

Stage 2/2: train transfer head on 10 classes...
Epoch 1/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 14s 11ms/step - accuracy: 0.6493 - loss: 0.9951 - val_accuracy: 0.7477 - val_loss: 0.7398
Epoch 2/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 11ms/step - accuracy: 0.6975 - loss: 0.8564 - val_accuracy: 0.7549 - val_loss: 0.7073
Epoch 3/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7151 - loss: 0.8110 - val_accuracy: 0.7692 - val_loss: 0.6751
Epoch 4/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7254 - loss: 0.7808 - val_accuracy: 0.7740 - val_loss: 0.6549
Epoch 5/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7343 - loss: 0.7473 - val_accuracy: 0.7773 - val_loss: 0.6490
Epoch 6/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7478 - loss: 0.7191 - val_accuracy: 0.7778 - val_loss: 0.6396
Epoch 7/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7501 - loss: 0.7053 - val_accuracy: 0.7822 - val_loss: 0.6359
Epoch 8/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 12s 10ms/step - accuracy: 0.7567 - loss: 0.6885 - val_accuracy: 0.7814 - val_loss: 0.6317
Epoch 9/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7592 - loss: 0.6787 - val_accuracy: 0.7821 - val_loss: 0.6315
Epoch 10/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7697 - loss: 0.6536 - val_accuracy: 0.7792 - val_loss: 0.6332
Epoch 11/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7723 - loss: 0.6448 - val_accuracy: 0.7799 - val_loss: 0.6245
Epoch 12/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7742 - loss: 0.6370 - val_accuracy: 0.7812 - val_loss: 0.6239
Epoch 13/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7776 - loss: 0.6230 - val_accuracy: 0.7813 - val_loss: 0.6284
Epoch 14/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7815 - loss: 0.6214 - val_accuracy: 0.7852 - val_loss: 0.6243
Epoch 15/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7826 - loss: 0.6130 - val_accuracy: 0.7831 - val_loss: 0.6202
Epoch 16/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7863 - loss: 0.6012 - val_accuracy: 0.7827 - val_loss: 0.6265
Epoch 17/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7887 - loss: 0.5940 - val_accuracy: 0.7800 - val_loss: 0.6243
Epoch 18/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7916 - loss: 0.5883 - val_accuracy: 0.7787 - val_loss: 0.6350
Epoch 19/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7930 - loss: 0.5798 - val_accuracy: 0.7874 - val_loss: 0.6146
Epoch 20/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7956 - loss: 0.5797 - val_accuracy: 0.7834 - val_loss: 0.6222
Epoch 21/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.8013 - loss: 0.5652 - val_accuracy: 0.7862 - val_loss: 0.6209
Epoch 22/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.7982 - loss: 0.5649 - val_accuracy: 0.7892 - val_loss: 0.6221
Epoch 23/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 11ms/step - accuracy: 0.8008 - loss: 0.5583 - val_accuracy: 0.7848 - val_loss: 0.6312
Epoch 24/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 14s 11ms/step - accuracy: 0.8019 - loss: 0.5554 - val_accuracy: 0.7853 - val_loss: 0.6283
Epoch 25/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 11ms/step - accuracy: 0.8050 - loss: 0.5464 - val_accuracy: 0.7855 - val_loss: 0.6314
Epoch 26/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 14s 11ms/step - accuracy: 0.8046 - loss: 0.5501 - val_accuracy: 0.7877 - val_loss: 0.6275
Epoch 27/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 11ms/step - accuracy: 0.8087 - loss: 0.5368 - val_accuracy: 0.7842 - val_loss: 0.6328
Epoch 28/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 11ms/step - accuracy: 0.8097 - loss: 0.5394 - val_accuracy: 0.7887 - val_loss: 0.6271
Epoch 29/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.8109 - loss: 0.5333 - val_accuracy: 0.7820 - val_loss: 0.6341
Epoch 30/30
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 13s 10ms/step - accuracy: 0.8112 - loss: 0.5335 - val_accuracy: 0.7873 - val_loss: 0.6327

Compare excluded classes accuracy with base model from task 2...
/Users/the80hz/Developer/ml-labs/.venv/lib/python3.13/site-packages/keras/src/saving/saving_lib.py:797: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 22 variables. 
  saveable.load_own_variables(weights_store.get(inner_path))
Class 0 (airplane): base=0.786, transfer=0.765, delta=-0.021
Class 1 (automobile): base=0.907, transfer=0.845, delta=-0.062
```
