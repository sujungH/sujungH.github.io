```python
from google.colab import drive
drive.mount('/content/gdrive')

!jupyter nbconvert --to markdown "/content/gdrive/MyDrive/Colab Notebooks/Untitled14.ipynb"
```

    Mounted at /content/gdrive
    [NbConvertApp] Converting notebook /content/gdrive/MyDrive/Colab Notebooks/Untitled14.ipynb to markdown
    /usr/local/lib/python3.8/dist-packages/nbconvert/filters/datatypefilter.py:39: UserWarning: Your element with mimetype(s) dict_keys(['application/vnd.colab-display-data+json']) is not able to be represented.
      warn("Your element with mimetype(s) {mimetypes}"
    [NbConvertApp] Writing 25963 bytes to /content/gdrive/MyDrive/Colab Notebooks/Untitled14.md



```python
from __future__ import absolute_import, division, print_function, unicode_literals
!pip install tensorflow-gpu==2.0.0-rc1
!pip install 'h5py==2.10.0' --force-reinstall
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    [31mERROR: Could not find a version that satisfies the requirement tensorflow-gpu==2.0.0-rc1 (from versions: 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.3.0, 2.3.1, 2.3.2, 2.3.3, 2.3.4, 2.4.0, 2.4.1, 2.4.2, 2.4.3, 2.4.4, 2.5.0, 2.5.1, 2.5.2, 2.5.3, 2.6.0, 2.6.1, 2.6.2, 2.6.3, 2.6.4, 2.6.5, 2.7.0rc0, 2.7.0rc1, 2.7.0, 2.7.1, 2.7.2, 2.7.3, 2.7.4, 2.8.0rc0, 2.8.0rc1, 2.8.0, 2.8.1, 2.8.2, 2.8.3, 2.8.4, 2.9.0rc0, 2.9.0rc1, 2.9.0rc2, 2.9.0, 2.9.1, 2.9.2, 2.9.3, 2.10.0rc0, 2.10.0rc1, 2.10.0rc2, 2.10.0rc3, 2.10.0, 2.10.1, 2.11.0rc0, 2.11.0rc1, 2.11.0rc2, 2.11.0)[0m
    [31mERROR: No matching distribution found for tensorflow-gpu==2.0.0-rc1[0m
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting h5py==2.10.0
      Downloading h5py-2.10.0-cp38-cp38-manylinux1_x86_64.whl (2.9 MB)
    [K     |████████████████████████████████| 2.9 MB 5.4 MB/s 
    [?25hCollecting six
      Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
    Collecting numpy>=1.7
      Downloading numpy-1.24.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.3 MB)
    [K     |████████████████████████████████| 17.3 MB 583 kB/s 
    [?25hInstalling collected packages: six, numpy, h5py
      Attempting uninstall: six
        Found existing installation: six 1.15.0
        Uninstalling six-1.15.0:
          Successfully uninstalled six-1.15.0
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.6
        Uninstalling numpy-1.21.6:
          Successfully uninstalled numpy-1.21.6
      Attempting uninstall: h5py
        Found existing installation: h5py 3.1.0
        Uninstalling h5py-3.1.0:
          Successfully uninstalled h5py-3.1.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    ipython 7.9.0 requires jedi>=0.10, which is not installed.
    scipy 1.7.3 requires numpy<1.23.0,>=1.16.5, but you have numpy 1.24.0 which is incompatible.
    numba 0.56.4 requires numpy<1.24,>=1.18, but you have numpy 1.24.0 which is incompatible.[0m
    Successfully installed h5py-2.10.0 numpy-1.24.0 six-1.16.0





```python
from google.colab import drive
drive.mount('/content/gdrive')

```

    Mounted at /content/gdrive



```python
# 데이터 다운로드
import numpy as np
rows = np.loadtxt("/content/gdrive/MyDrive/Colab Notebooks/lotto.csv", delimiter=",")
row_count = len(rows)

# 당첨번호를 원핫인코딩벡터(ohbin)으로 변환
def numbers2ohbin(numbers):

    ohbin = np.zeros(45) #45개의 빈 칸을 만듬

    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []
    
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    
    return numbers
    
numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count-1]
y_samples = ohbins[1:row_count]
```


```python
# 데이터 나누기
train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))
```


```python
!pip uninstall -y numpy
!pip uninstall -y setuptools
!pip install setuptools 
!pip install bumpy 
```

    Found existing installation: numpy 1.24.0
    Uninstalling numpy-1.24.0:
      Successfully uninstalled numpy-1.24.0
    Found existing installation: setuptools 57.4.0
    Uninstalling setuptools-57.4.0:
      Successfully uninstalled setuptools-57.4.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting setuptools
      Downloading setuptools-65.6.3-py3-none-any.whl (1.2 MB)
    [K     |████████████████████████████████| 1.2 MB 5.3 MB/s 
    [?25hInstalling collected packages: setuptools
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    thinc 8.1.5 requires numpy>=1.15.0, which is not installed.
    tensorflow 2.9.2 requires numpy>=1.20, which is not installed.
    tensorboard 2.9.1 requires numpy>=1.12.0, which is not installed.
    spacy 3.4.4 requires numpy>=1.15.0, which is not installed.
    prophet 1.1.1 requires numpy>=1.15.4, which is not installed.
    pandas-gbq 0.17.9 requires numpy>=1.16.6, which is not installed.
    numba 0.56.4 requires numpy<1.24,>=1.18, which is not installed.
    mlxtend 0.14.0 requires numpy>=1.10.4, which is not installed.
    librosa 0.8.1 requires numpy>=1.15.0, which is not installed.
    kapre 0.3.7 requires numpy>=1.18.5, which is not installed.
    ipython 7.9.0 requires jedi>=0.10, which is not installed.
    httpstan 4.6.1 requires numpy<2.0,>=1.16, which is not installed.
    datascience 0.17.5 requires numpy, which is not installed.
    cufflinks 0.17.3 requires numpy>=1.9.2, which is not installed.
    arviz 0.12.1 requires numpy>=1.12, which is not installed.
    aesara 2.7.9 requires numpy>=1.17.0, which is not installed.
    aeppl 0.0.33 requires numpy>=1.18.1, which is not installed.[0m
    Successfully installed setuptools-65.6.3




    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting bumpy
      Downloading bumpy-0.4.3.tar.gz (8.9 kB)
    [33mWARNING: Discarding https://files.pythonhosted.org/packages/fd/7a/fefd35a661587ef023b41b7bc79145bbbc3a0c6829cc8376096e618ed4dd/bumpy-0.4.3.tar.gz#sha256=1bff350934a0ca609612ffe5d7d54c22ca0ef520008bb4bd95768f377bc69a9c (from https://pypi.org/simple/bumpy/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.[0m
      Downloading bumpy-0.4.2.tar.gz (8.9 kB)
    [33mWARNING: Discarding https://files.pythonhosted.org/packages/ea/b3/cfb7e15400cdd990acdfbe336b17e6caf7a494e4f9de5f00f6680ecf7141/bumpy-0.4.2.tar.gz#sha256=36e387dea826c8469d57832deafa0c8829a82d8c857b721eaddacd8c0eb5335b (from https://pypi.org/simple/bumpy/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.[0m
      Downloading bumpy-0.4.1.tar.gz (8.8 kB)
    [33mWARNING: Discarding https://files.pythonhosted.org/packages/91/d2/7d7cc19518f30bdd3c7b198178811a2386b111e2473cd92776084e77cf6a/bumpy-0.4.1.tar.gz#sha256=723a940696e6d4ab7e4cb3f81689e7221204ea36aebff09c00d78e0d9e4a7580 (from https://pypi.org/simple/bumpy/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.[0m
      Downloading bumpy-0.4.0.tar.gz (8.8 kB)
    [33mWARNING: Discarding https://files.pythonhosted.org/packages/23/2e/7615634f31af911444624acacb1a43b622265c82554126a703f640c33a2d/bumpy-0.4.0.tar.gz#sha256=9160f464b2989f7fddf0939641e09143763a87405433c5c1c39dea1f94bf5b08 (from https://pypi.org/simple/bumpy/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.[0m
      Downloading bumpy-0.3.0.tar.gz (9.1 kB)
    [33mWARNING: Discarding https://files.pythonhosted.org/packages/d0/6f/83341399c7e3badd1e85fb5a5d48cfc58958b31d72d487d5d2d1dddd8745/bumpy-0.3.0.tar.gz#sha256=7bd9eac59ad53a6363c0df6a8085f20c66b4ce0488b1e6204f3ed8a04fb3d2bc (from https://pypi.org/simple/bumpy/). Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.[0m
      Downloading bumpy-0.2.4.tar.gz (6.3 kB)
    Building wheels for collected packages: bumpy
      Building wheel for bumpy (setup.py) ... [?25l[?25hdone
      Created wheel for bumpy: filename=bumpy-0.2.4-py3-none-any.whl size=6535 sha256=c5b941e329a78dcb2fc9e0bcc9f93720a299d6817343378d5731a4225271505a
      Stored in directory: /root/.cache/pip/wheels/42/96/97/5388dae194bde55cacc26725ebf9155201d1a93329f037bf29
    Successfully built bumpy
    Installing collected packages: bumpy
    Successfully installed bumpy-0.2.4



```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# 모델을 정의합니다.
model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
])

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 매 에포크마다 훈련과 검증의 손실 및 정확도를 기록하기 위한 변수
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# 최대 100번 에포크까지 수행
for epoch in range(100):

    model.reset_states() # 중요! 매 에포크마다 1회부터 다시 훈련하므로 상태 초기화 필요

    batch_train_loss = []
    batch_train_acc = []
    
    for i in range(train_idx[0], train_idx[1]):
        
        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.train_on_batch(xs, ys) #배치만큼 모델에 학습시킴

        batch_train_loss.append(loss)
        batch_train_acc.append(acc)

    train_loss.append(np.mean(batch_train_loss))
    train_acc.append(np.mean(batch_train_acc))

    batch_val_loss = []
    batch_val_acc = []

    for i in range(val_idx[0], val_idx[1]):

        xs = x_samples[i].reshape(1, 1, 45)
        ys = y_samples[i].reshape(1, 45)
        
        loss, acc = model.test_on_batch(xs, ys) #배치만큼 모델에 입력하여 나온 답을 정답과 비교함
        
        batch_val_loss.append(loss)
        batch_val_acc.append(acc)

    val_loss.append(np.mean(batch_val_loss))
    val_acc.append(np.mean(batch_val_acc))

    print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))
```

    epoch    0 train acc 0.024 loss 0.409 val acc 0.051 loss 0.399
    epoch    1 train acc 0.022 loss 0.396 val acc 0.051 loss 0.398
    epoch    2 train acc 0.025 loss 0.394 val acc 0.051 loss 0.398
    epoch    3 train acc 0.025 loss 0.392 val acc 0.030 loss 0.399
    epoch    4 train acc 0.021 loss 0.388 val acc 0.020 loss 0.401
    epoch    5 train acc 0.020 loss 0.384 val acc 0.030 loss 0.404
    epoch    6 train acc 0.035 loss 0.378 val acc 0.020 loss 0.407
    epoch    7 train acc 0.048 loss 0.373 val acc 0.010 loss 0.411
    epoch    8 train acc 0.050 loss 0.368 val acc 0.010 loss 0.414
    epoch    9 train acc 0.058 loss 0.362 val acc 0.010 loss 0.417
    epoch   10 train acc 0.059 loss 0.356 val acc 0.010 loss 0.419
    epoch   11 train acc 0.059 loss 0.350 val acc 0.030 loss 0.422
    epoch   12 train acc 0.065 loss 0.344 val acc 0.020 loss 0.426
    epoch   13 train acc 0.071 loss 0.336 val acc 0.020 loss 0.431
    epoch   14 train acc 0.077 loss 0.328 val acc 0.020 loss 0.436
    epoch   15 train acc 0.081 loss 0.320 val acc 0.020 loss 0.441
    epoch   16 train acc 0.094 loss 0.311 val acc 0.020 loss 0.446
    epoch   17 train acc 0.095 loss 0.301 val acc 0.010 loss 0.452
    epoch   18 train acc 0.098 loss 0.291 val acc 0.000 loss 0.459
    epoch   19 train acc 0.110 loss 0.280 val acc 0.000 loss 0.466
    epoch   20 train acc 0.125 loss 0.269 val acc 0.010 loss 0.473
    epoch   21 train acc 0.128 loss 0.259 val acc 0.010 loss 0.482
    epoch   22 train acc 0.131 loss 0.248 val acc 0.010 loss 0.492
    epoch   23 train acc 0.145 loss 0.238 val acc 0.020 loss 0.502
    epoch   24 train acc 0.144 loss 0.228 val acc 0.020 loss 0.512
    epoch   25 train acc 0.139 loss 0.218 val acc 0.030 loss 0.520
    epoch   26 train acc 0.134 loss 0.208 val acc 0.030 loss 0.530
    epoch   27 train acc 0.124 loss 0.199 val acc 0.030 loss 0.537
    epoch   28 train acc 0.140 loss 0.190 val acc 0.030 loss 0.548
    epoch   29 train acc 0.135 loss 0.181 val acc 0.030 loss 0.558
    epoch   30 train acc 0.129 loss 0.175 val acc 0.020 loss 0.564
    epoch   31 train acc 0.133 loss 0.168 val acc 0.040 loss 0.578
    epoch   32 train acc 0.138 loss 0.160 val acc 0.030 loss 0.587
    epoch   33 train acc 0.120 loss 0.152 val acc 0.010 loss 0.602
    epoch   34 train acc 0.128 loss 0.144 val acc 0.030 loss 0.610
    epoch   35 train acc 0.136 loss 0.140 val acc 0.020 loss 0.621
    epoch   36 train acc 0.154 loss 0.134 val acc 0.010 loss 0.637
    epoch   37 train acc 0.156 loss 0.130 val acc 0.020 loss 0.644
    epoch   38 train acc 0.134 loss 0.122 val acc 0.010 loss 0.657
    epoch   39 train acc 0.160 loss 0.116 val acc 0.020 loss 0.671
    epoch   40 train acc 0.150 loss 0.118 val acc 0.030 loss 0.672
    epoch   41 train acc 0.142 loss 0.111 val acc 0.030 loss 0.686
    epoch   42 train acc 0.147 loss 0.102 val acc 0.010 loss 0.693
    epoch   43 train acc 0.135 loss 0.098 val acc 0.010 loss 0.703
    epoch   44 train acc 0.126 loss 0.095 val acc 0.000 loss 0.719
    epoch   45 train acc 0.145 loss 0.086 val acc 0.010 loss 0.737
    epoch   46 train acc 0.149 loss 0.083 val acc 0.000 loss 0.753
    epoch   47 train acc 0.146 loss 0.081 val acc 0.000 loss 0.757
    epoch   48 train acc 0.145 loss 0.079 val acc 0.010 loss 0.777
    epoch   49 train acc 0.140 loss 0.077 val acc 0.010 loss 0.762
    epoch   50 train acc 0.139 loss 0.071 val acc 0.020 loss 0.786
    epoch   51 train acc 0.149 loss 0.064 val acc 0.030 loss 0.800
    epoch   52 train acc 0.161 loss 0.067 val acc 0.010 loss 0.815
    epoch   53 train acc 0.177 loss 0.063 val acc 0.040 loss 0.828
    epoch   54 train acc 0.165 loss 0.059 val acc 0.010 loss 0.839
    epoch   55 train acc 0.169 loss 0.051 val acc 0.010 loss 0.852
    epoch   56 train acc 0.168 loss 0.054 val acc 0.010 loss 0.861
    epoch   57 train acc 0.155 loss 0.054 val acc 0.000 loss 0.866
    epoch   58 train acc 0.182 loss 0.048 val acc 0.000 loss 0.887
    epoch   59 train acc 0.152 loss 0.044 val acc 0.020 loss 0.886
    epoch   60 train acc 0.141 loss 0.046 val acc 0.010 loss 0.894
    epoch   61 train acc 0.131 loss 0.044 val acc 0.010 loss 0.906
    epoch   62 train acc 0.190 loss 0.042 val acc 0.010 loss 0.904
    epoch   63 train acc 0.150 loss 0.041 val acc 0.010 loss 0.935
    epoch   64 train acc 0.126 loss 0.038 val acc 0.020 loss 0.943
    epoch   65 train acc 0.164 loss 0.039 val acc 0.010 loss 0.948
    epoch   66 train acc 0.142 loss 0.036 val acc 0.000 loss 0.951
    epoch   67 train acc 0.150 loss 0.035 val acc 0.020 loss 0.960
    epoch   68 train acc 0.165 loss 0.033 val acc 0.010 loss 0.956
    epoch   69 train acc 0.141 loss 0.032 val acc 0.010 loss 0.977
    epoch   70 train acc 0.171 loss 0.034 val acc 0.010 loss 0.990
    epoch   71 train acc 0.155 loss 0.032 val acc 0.010 loss 0.990
    epoch   72 train acc 0.150 loss 0.026 val acc 0.020 loss 0.999
    epoch   73 train acc 0.168 loss 0.027 val acc 0.030 loss 0.991
    epoch   74 train acc 0.166 loss 0.029 val acc 0.020 loss 1.022
    epoch   75 train acc 0.149 loss 0.030 val acc 0.040 loss 1.061
    epoch   76 train acc 0.155 loss 0.025 val acc 0.010 loss 1.055
    epoch   77 train acc 0.166 loss 0.024 val acc 0.020 loss 1.048
    epoch   78 train acc 0.171 loss 0.022 val acc 0.020 loss 1.080
    epoch   79 train acc 0.165 loss 0.021 val acc 0.030 loss 1.090
    epoch   80 train acc 0.163 loss 0.021 val acc 0.020 loss 1.086
    epoch   81 train acc 0.139 loss 0.018 val acc 0.030 loss 1.102
    epoch   82 train acc 0.152 loss 0.019 val acc 0.030 loss 1.114
    epoch   83 train acc 0.170 loss 0.020 val acc 0.010 loss 1.099
    epoch   84 train acc 0.175 loss 0.019 val acc 0.010 loss 1.146
    epoch   85 train acc 0.144 loss 0.016 val acc 0.010 loss 1.134
    epoch   86 train acc 0.151 loss 0.013 val acc 0.010 loss 1.144
    epoch   87 train acc 0.165 loss 0.013 val acc 0.020 loss 1.162
    epoch   88 train acc 0.131 loss 0.016 val acc 0.020 loss 1.163
    epoch   89 train acc 0.166 loss 0.019 val acc 0.030 loss 1.187
    epoch   90 train acc 0.161 loss 0.019 val acc 0.020 loss 1.183
    epoch   91 train acc 0.151 loss 0.016 val acc 0.020 loss 1.208
    epoch   92 train acc 0.159 loss 0.012 val acc 0.040 loss 1.202
    epoch   93 train acc 0.140 loss 0.014 val acc 0.010 loss 1.211
    epoch   94 train acc 0.161 loss 0.014 val acc 0.020 loss 1.207
    epoch   95 train acc 0.158 loss 0.018 val acc 0.010 loss 1.218
    epoch   96 train acc 0.141 loss 0.017 val acc 0.020 loss 1.249
    epoch   97 train acc 0.155 loss 0.011 val acc 0.030 loss 1.243
    epoch   98 train acc 0.151 loss 0.009 val acc 0.020 loss 1.262
    epoch   99 train acc 0.146 loss 0.010 val acc 0.020 loss 1.249



```python
# 번호 뽑기
def gen_numbers_from_probability(nums_prob):

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls
    
print('receive numbers')

xs = x_samples[-1].reshape(1, 1, 45)

ys_pred = model.predict_on_batch(xs)

list_numbers = []

for n in range(5):
    numbers = gen_numbers_from_probability(ys_pred[0])
    numbers.sort()
    print('{0} : {1}'.format(n, numbers))
    list_numbers.append(numbers)
```

    receive numbers
    0 : [2, 6, 7, 12, 19, 44]
    1 : [12, 17, 19, 21, 27, 29]
    2 : [7, 17, 19, 23, 29, 44]
    3 : [3, 6, 12, 17, 19, 21]
    4 : [2, 7, 12, 16, 19, 29]



```python
import random
for i in range(5):
    lotto = random.sample(range(1,46),6)
    lotto.sort()
    print("자동 로또 번호는 ", lotto)
```

    자동 로또 번호는  [1, 3, 4, 6, 15, 18]
    자동 로또 번호는  [4, 19, 33, 38, 39, 41]
    자동 로또 번호는  [7, 8, 9, 15, 19, 38]
    자동 로또 번호는  [2, 11, 13, 18, 30, 31]
    자동 로또 번호는  [5, 21, 28, 36, 38, 41]

