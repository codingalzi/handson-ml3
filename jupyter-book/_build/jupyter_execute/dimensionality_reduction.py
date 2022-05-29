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

# **고차원 공간**
# 
# * 3차원을 초과하는 고차원의 공간을 상상하기 매우 어려움.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-01.png" width="500"/></div>

# **차원의 저주**
# 
# * 차원의 커질 수록 두 지점 사이의 거리가 매우 커짐.
# 
# * 즉, 특성 수가 아주 많은 경우, 훈련 샘플 사이의 거리가 매우 커서 과대적합 위험도가 커짐.
# 
# * 이유: 두 샘플 사이의 거리가 멀어서 기존 값들을 이용한 추정(예측)이 여러 과정을 거쳐야 하기 때문임.
# 
# * 해결책: 샘플 수 늘리기. 하지만 고차원의 경우 충분히 많은 샘플 수를 준비하는 일은 사실상 불가능.

# ## 차원축소 기법

# **기본 아이디어**
# 
# * 모든 훈련 샘플이 고차원 공간의 일부인 저차원 부분공간에 가깝게 놓여 있는 경우가 일반적으로 발생

# ### 사영 기법

# * $n$차원 공간에 존재하는 $d$차원 부분공간을 $d$차원 공간으로 사영하기. 단, $d < n$.
# 
# * 예제
#     * 왼쪽 3차원에 존재하는 적절한 2차원 평면으로 사영하면 적절한 2차원 상의 이미지를 얻게됨.
#     * 오른쪽 2차원 이미지에 사용된 축 $z_1$과 $z_2$를 적절하게 찾는 게 주요 과제임.

# <table>
#     <tr>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-1.png" width="400"/> </td>
#         <td></td>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-2.png" width="400"/> </td>
#     </tr>
# </table>

# **부적절한 사영**
# 
# * 사영이 경우에 따라 보다 복잡한 결과를 낼 수 있음.
# 
# * 롤케이크를 $x_1$과 $x_2$ 축으로 사영하면 샘플 구분이 보다 어려워짐.

# <table>
#     <tr>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-04.png" width="300"/> </td>
#         <td></td>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-06.png" width="320"/> </td>
#     </tr>
# </table>

# ### 다양체 학습

# **다양체**
# 
# 고차원 공간에서 저차원 공간을 접거가 접거나 비틀어서 생성한 공간을 
# **다양체**<font size='2'>manifold</font>라 부른다.
# 예를 들어, 롤케이크<font size='2'>Swiss roll</font>는 
# 2차원 평면을 돌돌 말아 3차원 공간상에 존재하는 2D 다양체다. 
# 실제로 롤케이크을 조심해서 펴면 보다 적절한 2차원 공간으로 변환된다.

# <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-07.png" width="310"/>

# **다양체 가설**

# 롤케이크와 같은 다양체의 경우 사영 보다는 접히거나 비틀어진 것을 잘 펼치면 
# 보다 적절한 2차원 공간으로 변환된다.
# 이처럼 숨겨진 다양체를 찾는 과정이 **다양체 학습**<font size='2'>Manifold Learning</font>이다. 
# 다양체 학습은 대부분의 고차원 데이터셋이 더 낮은 차원의 다양체에 가깝다는가설에 근거한다.
# 아래 그램의 위쪽 데이터셋의 경우가 그렇다.

# 다양체 가설은 또한 저차원의 다양체 공간으로 차원축소를 진행하면 보다 간단한 다양체가 된다라는 
# 가설과 함께 사용된다. 
# 하지만 이는 경우에 따라 다르다.
# 아래 그림의 아랫쪽 데이터셋의 경우는 차원축소를 진행하면 데이터셋이 보다 복잡해진다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-08.png" width="600"/></div>

# ## PCA(주성분 분석)

# * 훈련 데이터에 가장 가까운 초평면(hyperplane)을 정의한 다음, 그 평면에 사영하는 기법
# 
# * **주성분 분석**<font size='2'>principal component analysis</font>(PCA)이 핵심.
# 
# * 분산 보존 개념과 주성분 개념이 중요함.

# ### 분산 보존

# * 분산 보존: 저차원으로 사영할 때 훈련 세트의 분산이 최대한 유지되도록 축을 지정해야 함.
# 
# * 예제: 아래 그림에서 $c_1$ 벡터가 위치한 실선 축으로 사영하는 경우가 분산을 최대한 보존함.
#     그러면 $c_1$에 수직이면서 분산을 최대로 보존하는 축은 $c_2$임.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-09.png" width="600"/></div>

# **주성분**

# * 첫째 주성분: 분산을 최대한 보존하는 축
# 
# * 둘째 주성분: 첫째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축
# 
# * 셋째 주성분: 첫째, 둘째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축
# 
# * ...

# **주성분과 사영**

# * 특잇값 분해(SVD) 기법을 이용하면 쉽게 해결됨.

# * 특잇값 분해: m x n 모양을 가지며, 평균값이 0인 데이터셋 $X$가 주어졌을 때 
#     아래 조건을 만족시키는 세 개의 행렬 
#     $U$, $\Sigma$, $V$가 존재.
#     - $U$: m x m 행렬
#     - $\Sigma$: m x n 모양의 대각행렬(diagonal matrix). 
#     - $V$: n x n 행렬. 윗첨자 $T$는 전치행렬을 의미함.
# 
#     $$
#     X = U\, \Sigma \, V^{\!T}
#     $$
# 
# 

# * 주성분 벡터는 행렬 $V$의 열에 해당하며, 따라서
#     $d$차원으로의 사영은 아래와 같이 계산됨:
#     
#     $$
#     X\, (V\text{[: ,  :d]})
#     $$

# **사이킷런의 `PCA` 모델**

# * 사이킷런의 PCA 모델 제공
#     * SVD 기법 활용

# * 예제: 데이터셋의 차원을 2로 줄이기
# 
#     ```python
#     from sklearn.decomposition import PCA
# 
#     pca = PCA(n_components = 2)
#     X2D = pca.fit_transform(X)
#     ```

# ### 설명 분산 비율

# * `explained_variance_ration_` 속성 변수: 각 주성분에 대한 원 데이터셋의 분산 비율 저장

# * 예제: 아래 사영 그림에서 설명된 3차원 데이터셋의 경우.
#     * $z_1$ 축: 75.8%
#     * $z_2$ 축: 15.2%

# ```python
# >>> pca.explained_variance_ratio_
# array([0.7578477 , 0.15186921])
# ```

# <table>
#     <tr>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-1.png" width="400"/> </td>
#         <td></td>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-02-2.png" width="400"/> </td>
#     </tr>
# </table>

# **적절한 차원**

# * 적절한 차원: 밝혀진 분산 비율의 합이 95% 정도 되도록 하는 주성분들로 구성

# * 데이터 시각화 목적의 경우: 2개 또는 3개

# **설명 분산 비율 활용**

# * 설명 분산 비율의 합과 차원 사이의 그래프 활용

# * 설명 분산의 비율의 합의 증가가 완만하게 변하는 지점(elbow)에 주시할 것.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-10.png" width="400"/></div>

# **(MNIST 활용 예제) 압축을 위한 PCA**

# * PCA를 MNIST 데이터셋의 차원축소를 위해 사용할 수 있음.
# 
# * MINST 데이터셋의 주성분 분석을 통해 95% 정도의 분산을 유지하려면 154개 정도의 주성분만 사용해도 됨.
# 
# * 아래 코드: 154개 주성분 사용하여 차원축소하기
# 
#     ```python
#     pca = PCA(n_components = 154)
#     X_reduced = pca.fit_transform(X_train)
#     ```

# ### 재구성 오차

# * 차원축소 결과:
#     * 784차원을 154 차원으로 줄임.
#     * 유실된 정보: 5%
#     * 크기: 원본 데이터셋 크기의 20%

# * 원본과의 비교: 정보손실 크지 않음 확인 가능

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-11.png" width="400"/></div>

# ### 랜덤 PCA

# * 주성분 선택을 위해 사용되는 SVD 알고리즘을 확률적으로 작동하도록 만드는 기법
# 
# * 보다 빠르게 지정된 개수의 주성분에 대한 근삿값을 찾아줌.
# 
# * $d$가 $n$ 보다 많이 작으면 기본 SVD 보다 훨씬 빠름.
#     - 기존의 특잇값 분해 알고리즘의 시간 복잡도: $O(m \times n^2) + O(n^3)$
#     - 랜덤 특잇값 분해 알고리즘의 시간 복잡도: $O(m \times d^2) + O(d^3)$
# 
# 
# * 아래 코드: `svd_solver` 옵션을 `"randomized"`로 설정
# 
#     ```python
#     rnd_pca = PCA(n_components = 154, svd_solver="randomized")
#     X_reduced = rnd_pca.fit_transform(X_train)
#     ```

# ### 점진적 PCA

# * 훈련세트를 미니배치로 나눈 후 IPCA(점진적 PCA)에 하나씩 주입 가능

# * 온라인 학습에 적용 가능

# * `partial_fit()` 활용에 주의할 것.
# 
#     ```python
#     from sklearn.decomposition import IncrementalPCA
# 
#     n_batches = 100
#     inc_pca = IncrementalPCA(n_components=154)
#     for X_batch in np.array_split(X_train, n_batches):
#         inc_pca.partial_fit(X_batch)
# 
#     X_reduced = inc_pca.transform(X_train)
#     ```

# **넘파이의 `memmap()` 클래스 활용**

# * 바이너리 파일로 저장된 (매우 큰) 데이터셋을 마치 메모리에 들어있는 것처럼 취급할 수 있는 도구 제공

# * 이를 이용하여 미니배치/온라인 학습 가능
# 
#     ```python
#     X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
#     inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
#     inc_pca.fit(X_mm)
#     ```

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

# **아이디어**

# * 대표적인 다양체 학습 기법
# 
# * 롤케이크 데이터셋의 경우처럼 전체적으론 비선형인 다양체이지만 국소적으로는 데이터가 선형적으로 연관되어 있음.
# 
# * 국소적 관계가 가장 잘 보존되는 훈련 세트의 저차원 표현 찾을 수 있음.
# 
# * 사영이 아닌 다양체 학습에 의존

# **예제: 롤케이크**

# ```python
# X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
# lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
# X_unrolled = lle.fit_transform(X_swiss)
# ```

# <table>
#     <tr>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-04.png" width="300"/> </td>
#         <td></td>
#         <td> <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch08/homl08-14.png" width="370"/> </td>
#     </tr>
# </table>

# **예제: MNIST 데이터셋 시각화**

# * 다양한 차원축소 기법을 이용한 MNIST 데이터셋 시각화 가능
# 
# * 참조: [사이킷런 활용 손글씨 데이터셋 시각화](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py)
