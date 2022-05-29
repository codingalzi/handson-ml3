#!/usr/bin/env python
# coding: utf-8

# (ch:dimensionalityReduction)=
# # 차원축소

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 차원축소](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_dimensionality_reduction.ipynb)에서 확인할 수 있다.

# **주요 내용**
# 
# 샘플의 특성이 너무 많으면 학습이 매우 느리거나 어려워지는 현상를
# **차원의 저주**라 한다.
# 이 문제를 해결하기 위해 특성 수를 (크게) 줄여서 학습 불가능한 문제를 학습 가능한 문제로 만드는
# **차원축소** 기법을 사용할 수 있다.
# 차원축소로 인한 정보손실을 어느 정도 감안하면서 훈련 속도와 성능을 최대로 유지하는 것을
# 목표로 삼는다.

# 예를 들어, MNIST 데이터셋의 경우 사진의 중앙에만 집중하거나({prf:ref}`exp-MNIST-feature-importance`)
# 인접한 픽셀의 평균값만을 이용해도 숫자 인식에 별 문제 없다.
# 주성분 분석(PCA) 기법을 이용하여 손글씨 사진의 784개 픽셀 대신 154개만 대상으로 삼아도
# 충분히 학습이 가능함을 보일 것이다.
# 
# 차원축소 기법은 또한 데이터 시각화에도 활용된다.
# 데이터의 차원(특성 수)을 2, 3차원으로 줄이면 데이터셋을 시각화할 수 있다.
# 데이터 시각화는 데이터 군집 같은 시각적인 패턴을 감지하여 데이터에 대한 통찰을 얻거나
# 데이터에 대한 정보를 제3자에게 전달하는 데에 도움된다.

# 차원축소를 위한 접근법은 크게 사영 기법과 다양체 학습 기법으로 나뉜다. 
# 사영 기법 알고리즘으로 PCA(주성분 분석)와 임의 사영<font size='2'>Random Projection</font>을,
# 다양체 학습 알고리즘으로 LLE(국소적 선형 임베딩)을 소개한다.

# ## 차원의 저주

# 3차원을 초과하는 고차원의 공간을 상상하는 하는 일은 매우 어렵다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-01.png" width="500"/></div>

# 차원의 커질 수록 두 지점 사이의 거리가 매우 멀어진다.
# 이는 특성 수가 아주 많은 경우, 훈련 샘플 사이의 거리가 매우 커서 과대적합 위험도가 커짐을 의미한다.
# 이유는 두 샘플 사이의 거리가 멀어서 기존 값들을 이용한 추정이 여러 과정을 거쳐야 하기 때문에
# 훈련셋에 과학게 의존하게 되기 때문이다.
# 
# 훈련셋의 크기를 키워야 하지만 고차원의 경우 충분히 많은 샘플 수를 준비하는 일은 사실상 불가능하다는 것이
# **차원의 저주**라는 표현의 핵심이다.

# ## 차원축소 기법

# 훈련 샘플이 고차원 공간의 일부인 저차원 부분공간에 가깝게 놓여 있는 경우가 일반적으로 발생한다.
# 이런 경우 고차원의 데이터셋을 저차원의 데이터셋으로 변환시켜도 정보의 손실이 크지 않게 유지할 수 있다.
# 이것이 차원축소 기법이며 크게 사영 기법과 다양체 학습 기법으로 나뉜다.

# ### 사영 기법

# $n$차원 데이터셋을 $d$($d < n$) 차원 데이터셋으로 사영하는 기법이다.
# 아래 그림은 
# 왼쪽 3차원에 존재하는 데이터셋을 적절한 2차원 평면으로 사영한 결과를 보여준다.
# 이때 오른쪽 이미지에 사용된 축 $z_1$과 $z_2$를 적절하게 찾는 게 주요 과제다.

# <table>
#     <tr>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-1.png" width="400"/></div> </td>
#         <td></td>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-2.png" width="400"/></div> </td>
#     </tr>
# </table>

# 위의 경우는 사영을 통해 데이터셋 분석이 보다 간단해졌다.
# 하지만 경우에 따라 보다 복잡한 결과를 낼 수도 있다.
# 예를 들어 아래 그림은 롤케이크를 $x_1$과 $x_2$ 축으로 
# 이루어진 평면에 사영하면 샘플 구분이 보다 어려워지는 것을 보여준다.

# <table>
#     <tr>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-04.png" width="350"/></div> </td>
#         <td></td>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-06.png" width="320"/></div> </td>
#     </tr>
# </table>

# ### 다양체 학습 기법

# **다양체**
# 
# 고차원 공간에서 저차원 공간을 접거가 접거나 비틀어서 생성한 공간을 
# **다양체**<font size='2'>manifold</font>라 부른다.
# 예를 들어, 롤케이크<font size='2'>Swiss roll</font>는 
# 2차원 평면을 돌돌 말아 3차원 공간상에 존재하는 2D 다양체다. 
# 실제로 롤케이크을 조심해서 펴면 보다 적절한 2차원 데이터셋으로 변환된다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-07.png" width="310"/></div>

# **다양체 가설**

# 롤케이크와 같은 다양체의 경우 사영 보다는 접히거나 비틀어진 것을 잘 펼치면 
# 보다 적절한 2차원 데이터셋으로 변환된다.
# 이처럼 숨겨진 다양체를 찾는 과정이 **다양체 학습**<font size='2'>Manifold Learning</font>이다. 
# 다양체 학습은 대부분의 고차원 데이터셋이 더 낮은 차원의 다양체에 가깝다는가설에 근거한다.

# 다양체 가설은 또한 저차원의 다양체 공간으로 차원축소를 진행하면 보다 간단한 다양체가 된다라는 
# 가설과 함께 사용된다. 
# 하지만 이는 경우에 따라 다르다.
# 아래 그림의 위쪽 데이터셋의 경우는 보다 간단해지지만,
# 아랫쪽 데이터셋의 경우는 차원축소를 진행하면 데이터셋이 보다 복잡해진다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-08.png" width="600"/></div>

# ## PCA(주성분 분석)

# 훈련 데이터셋을 특정 초평면(hyperplane)에 사영하는 기법이다.
# 초평면은 **주성분 분석**<font size='2'>principal component analysis</font>(PCA)을
# 이용하여 결정한다.
# 초평면 결정에 **분산 보존** 개념과 **주성분** 개념이 중요하다.

# **분산 보존**

# 저차원으로 사영할 때 훈련셋의 분산이 최대한 유지되도록 축을 지정해야 한다.
# 아래 그림에서 $c_1$ 벡터가 위치한 실선 축으로 사영하는 경우가 분산을 최대한 보존한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-09.png" width="500"/></div>

# **주성분과 특잇값 분해(SVD)**
# 
# 주성분은 다음 과정으로 차례대로 찾아야 한다. 
# 
# * 첫째 주성분: 분산을 최대한 보존하는 축
# * 둘째 주성분: 첫째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축
# * 셋째 주성분: 첫째, 둘째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축
# * ...
# 
# 하지만 특잇값 분해(SVD) 기법을 이용하면 쉽게 찾을 수 있으며,
# 찾아진 초평면으로의 사영 또한 수학적으로 쉽게 계산된다.

# **사이킷런의 `PCA` 모델**

# 사이킷런의 `PCA` 모델은 SVD 기법을 활용한다.
# 예를 들어 아래 코드는 데이터셋의 차원을 2로 줄인다.
# 
# ```python
# pca = PCA(n_components=2)
# X2D = pca.fit_transform(X)
# ```

# **설명 분산 비율**

# 훈련된 모델의 `explained_variance_ration_` 속성 변수에 각 주성분에 대한 원 데이터셋의 분산 비율이 저장된다.
# 예를 들어 아래 사영 그림에서 설명된 3차원 데이터셋의 경우,
# 새로운 축을 기준으로 원 데이터셋의 분산 비율은 다음과 같다.
# 
# * $z_1$ 축: 75.8%
# * $z_2$ 축: 15.2%
# 
# ```python
# >>> pca.explained_variance_ratio_
# array([0.7578477 , 0.15186921])
# ```

# <table>
#     <tr>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-1.png" width="400"/></div> </td>
#         <td></td>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-2.png" width="400"/></div> </td>
#     </tr>
# </table>

# **적절한 차원**
# 
# 밝혀진 분산 비율의 합이 95% 정도 되도록 하는 주성분들로 구성되도록 차원을 정하는 것이 좋다.
# 반면에 데이터 시각화가 목적인 경우엔 2개 또는 3개의 주성분만을 사용해야 한다.

# **설명 분산 비율 활용**
# 
# 적절한 차원을 결정하기 위해 설명 분산 비율의 합과 차원 사이의 그래프를 활용할 수도 있다.
# 예를 들어 설명 분산의 비율의 합의 증가가 완만하게 변하는 지점(elbow)에 주시하면 좋다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-10.png" width="400"/></div>

# 위 그래프를 통해 설명 분산 비율의 합이 95% 정도가 되려면 154개의 차원이 필요함을 확인할 수 있다.
# 따라서 `n_components = 154` 를 하이퍼파라미터로 지정할 수 있으나
# 이보다는 `n_components = 0.95` 로 지정하는 것이 보다 편리하다.
# 
# ```python
# pca = PCA(n_components = 0.95)
# X_reduced = pca.fit_transform(X_train)
# ```

# **파이프라인과 랜덤 탐색 활용**
# 
# 적절한 차원을 찾기 위해 `PCA` 를 전처리로 사용하는 파이프라인을 생성하여
# 랜덤 탐색을 이용할 수 있다.
# 예를 들어, 아래 코드는 차원 축소와 랜덤 포레스트 모델을 파이프라인으로 엮어서
# 랜텀 탐색을 이용하여 적절한 차원을 찾는다.
# 
# ```python
# clf = make_pipeline(PCA(random_state=42),
#                     RandomForestClassifier(random_state=42))
# param_distrib = {
#     "pca__n_components": np.arange(10, 80),
#     "randomforestclassifier__n_estimators": np.arange(50, 500)
# }
# rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,
#                                 random_state=42)
# rnd_search.fit(X_train[:1000], y_train[:1000])
# ```

# **파일 압축**

# 파일 압축 용도로 PCA를 활용할 수 있다.
# MNIST 데이터셋의 경우 784차원을 154 차원으로 줄이면 
# 데이터셋의 크기가 원래의 20% 수준에 불과해지며]
# 훈련 속도는 훨씬 빨라진다. 
# 하지만 정보는 5% 정도만 잃는다.
# 정보 손실이 크지않음을 아래 두 그림이 확인해준다.
# 왼쪽이 원본이고 오른쪽이 압축된 데이터들이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-11.png" width="400"/></div>

# **랜덤 PCA**

# 주성분 선택을 위해 사용되는 SVD 알고리즘을 확률적으로 작동하도록 만드는 기법이다.
# 보다 빠르게 지정된 개수의 주성분에 대한 근삿값을 찾아준다.
# 
# ```python
# rnd_pca = PCA(n_components = 154, svd_solver="randomized")
# X_reduced = rnd_pca.fit_transform(X_train)
# ```

# **점진적 PCA**

# 훈련세트를 미니배치로 나눈 후 IPCA(점진적 PCA)에 하나씩 주입하는 모델이며,
# 온라인 학습에 활용될 수 있다.
# 단, 훈련에 `partial_fit()` 을 사용한다.
# 
# ```python
# n_batches = 100
# inc_pca = IncrementalPCA(n_components=154)
# for X_batch in np.array_split(X_train, n_batches):
#     inc_pca.partial_fit(X_batch)
# 
# X_reduced = inc_pca.transform(X_train)
# ```

# **`memmap` 클래스 활용**
# 
# 넘파이의 `memmap` 클래스는
# 바이너리 파일로 저장된 (매우 큰) 데이터셋을 마치 메모리에 들어있는 것처럼 취급할 수 있는 도구를
# 제공한다.
# 
# 이를 이용하여 미니배치/온라인 학습이 가능하다.
# 
# ```python
# filename = "my_mnist.mmap"
# X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
# X_mmap[:] = X_train  # could be a loop instead, saving the data chunk by chunk
# X_mmap.flush()
# 
# X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
# batch_size = X_mmap.shape[0] // n_batches
# inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
# inc_pca.fit(X_mmap)
# ```

# ## 임의 사영

# **존슨-린덴슈트라우스 정리**
# 
# 존슨-린덴슈트라우스<font size='2'>Johnson-Lindenstrauss</font> 정리에 의해 
# 고차원의 데이터를 적절한 크기의 저차원으로 임의적으로 사영하더라도
# 데이터셋의 정보를 많이 잃어버리지 않는다.
# 적절한 크기의 차원 $d$는 정보를 얼마나 잃어도 되는가에 따라 결정되며,
# 아래 값을 만족하면 된다.
# 단, $m$ 은 훈련셋의 크기를 나타내며, 
# $\varepsilon$ 은 정보손실 정도를 가리킨다.
# 
# $$
# d \ge \frac{4 \log(m)}{\frac{1}{2} \varepsilon^2 - \frac{1}{3} \varepsilon^3}
# $$
# 
# **임의 사영**<font size='2'>Random Projection</font>은 존슨-린덴슈트라우스 정리를
# 이용한 사영을 가리킨다.

# :::{admonition} $\varepsilon$ 의 역할
# :class: info
# 
# $\varepsilon$ 은 사영된 두 데이터 사이의 거리가 기존의 거리에 비해 차이날 수 있는 정도이다. 
# 예를 들어 $\varepsilon=0.1$ 로 지정하면 
# 기존의 두 데이터의 거리의 제곱에 비해 사영된 두 데이터 사이의 거리의 제곱이 10% 정도의 차이만
# 허용한다는 의미다.
# :::

# **사이키런의 `GaussianRandomProjection` 모델**
# 
# `GaussianRandomProjection` 모델이 앞서 존슨-린덴슈트라우스 정리를 이용한 
# 임의 사영을 실행한다. 
# 
# ```python
# gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
# X_reduced = gaussian_rnd_proj.fit_transform(X)
# ```

# **사이키런의 `SparseRandomProjection` 모델**
# 
# 희소 행렬을 사용하는 `GaussianRandomProjection` 모델이며 보다 빠르고 메모리 효율적이다. 
# 
# ```python
# gaussian_rnd_proj = SparseRandomProjection(eps=ε, random_state=42)
# X_reduced = gaussian_rnd_proj.fit_transform(X)
# ```

# ## LLE(국소적 선형 임베딩)

# 대표적인 다양체 학습 기법이다. 
# 롤케이크 데이터셋의 경우처럼 전체적으론 비선형인 다양체이지만 국소적으로는 데이터가 선형적으로 연관되어
# 있다는 가설을 이용한다.
# 국소적 관계가 가장 잘 보존되는 훈련 세트의 저차원 표현을 찾는다.
# 
# 아래 코드는 롤케이크에 대해 LLE 를 적용한 결과를 보여준다.
# 
# ```python
# X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
# lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
# X_unrolled = lle.fit_transform(X_swiss)
# ```

# <table>
#     <tr>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-04.png" width="350"/></div> </td>
#         <td></td>
#         <td> <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-14.png" width="370"/></div> </td>
#     </tr>
# </table>

# ## 부록: 기타 차원 축소 모델

# 사이킷런에서 제공하는 기타 차원 축소 모델은 다음과 같다.
# 
# * 다차원 스케일링<font size='2'>Multidimensional Scaling</font>(MDS)
# * Isomap
# * t-SNE(t-Distributed Stochasting Neighbor Embedding)
# * 선형 판별 분석<font size='2'>Linear Discriminant Analysis</font>(LDA)
# * 커널 PCA

# 아래 그림은 롤케이크를 각각 MDS, Isomap, t-SNE 방식으로 2차원으로 변환한 결과를 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-15.png" width="700"/></div>

# 아래 그림은 롤케이크를 다양한 커널을 이용하여 커널 PCA로 2차원 데이터셋으로 변환한 결과를 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-16.png" width="730"/></div>
