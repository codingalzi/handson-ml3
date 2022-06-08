#!/usr/bin/env python
# coding: utf-8

# (ch:unsupervisedLearning)=
# # 비지도 학습

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 비지도 학브](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_unsupervised_learning.ipynb)에서 확인할 수 있다.

# **주요 내용**
# 
# 비지도 학습은 레이블이 없는 데이터를 학습하는 기법이며,
# 주로 아래 분야에서 활용된다.
# 
# - 군집화: 비슷한 샘플끼리의 군집을 형성하는 것이며,
#     아래 용도에 활용된다.
# 
#     * 데이터 분석
#     * 고객분류
#     * 추천 시스템
#     * 검색 엔진
#     * 이미지 분할
#     * 차원 축소
#     * 준지도 학습
# 
# - 이상치 탐지: 정상 테이터와 이상치를 구분하는 데에 활용된다.
# 
#     * 생산라인에서 결함제품 탐지
#     * 새로운 트렌드 찾기
# 
# - 데이터 밀도 추정: 데이터셋의 확률밀도를 추정한다.
# 
#     * 이상치 분류: 밀도가 낮은 지역에 위치한 샘플
#     * 데이터 분석
#     * 데이터 시각화
#     
# 여기서는 군집화 관련 다양한 알고리즘과 모델을 살펴본다.
# 
# - 군집화
# - k-평균
# - DBSCAN
# - 가우시안 혼합

# ## 군집화

# **군집과 군집화**
# 
# **군집**<font size='2'>cluster</font>은 유사한 대상들의 모음을 가리킨다.
# 예를 들어, 산이나 공원에서 볼 수 있는 이름은 모르지만 동일 품종의 꽃으로 이루어진 군집 등을
# 생각하면 된다.
# **군집화**<font size='2'>clustering</font>는 대상들을 나누어 군집을 
# 형성하는 것을 말한다. 

# **분류 대 군집화**
# 
# 각 샘플에 하나의 그룹을 할당한다는 점에서 유사하다.
# 하지만 군집화는 군집이 미리 레이블(타깃)로 지정되지 않고 예측기 스스로 적절한 군집을
# 찾아내야 한다는 점에서 다르다.
# 
# 아래 왼쪽 그림은 붓꽃의 꽃잎 길이와 너비를 특성으로 사용해서 품종을 분류한 결과를
# 보여주지만, 왼쪽 그림은 군집화의 결과를 보여준다. 
# 분류는 세 개의 품종을 매우 잘 분류하지만 군집은 세토사 군집과 나머지 군집으로 구분할 뿐이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-01.png" width="600"/></div>

# 반면에 **가우시안 혼합 모델**<font size='2'>Gaussian Mixture Model</font>(GMM)을 
# 꽃잎의 길이와 너비 뿐만 아니라
# 꽃받침의 길이와 너비 특성까지 적용항 붓꽃 데이터셋에 대해 적용하면
# 세 개의 군집을 매우 정확하게 생성한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-02.png" width="450"/></div>

# 군집에 대한 보편적 정의는 없다. 사용되는 알고리즘에 따라 다른 형식의 군집을 생성한다.
# 
# * K-평균: 센트로이드(중심)라는 특정 샘플을 중심으로 모인 샘플들의 집합
# * DBSCAN: 밀집된 샘플들의 연속으로 이루어진 집합
# * 가우시안 혼합: 특정 가우시안 분포를 따르는 샘플들의 집합

# ## K-평균

# 각 군집의 중심인 센트로이드<font size='2'>centroid</font>을 찾고 
# 각 샘플에 대해 가장 가까운 센트로이드를 중심으로 군집을 형성하는 기법이다.

# **예제: 사이킷런의 `KMeans` 모델** 
# 
# 아래 그림은 다섯 개의 샘플 덩어리로 이루어진 데이터셋을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-03.png" width="450"/></div>

# 위 데이터셋에 대해 다섯 개의 군집을 형성하는 K-평균 알고리즘은 다음과 같이 적용한다.
# 군집 수를 몇 개로 지정하는가는 미리 알 수 없다. 
# 나중에 몇 개의 군집이 적절한가를 판단하는 여러 방식을 살펴볼 것이다. 

# ```python
# k = 5
# kmeans = KMeans(n_clusters=k, random_state=42)
# y_pred = kmeans.fit_predict(X)
# ```

# `predict()` 함수의 반환값은 0, 1, 2, 3, 4 등의 인덱스이다.
# 하지만 이는 임의로 지정된 군집의 인덱스를 가리키며, 클래스 분류와는 아무 상관이 없다.

# **보로노이 다이어그램**

# 평면을 특정 점(센트로이드)까지의 거리가 가장 가까운 점의 집합으로 분할한 그림이다. 
# 경계 부분의 일부 샘플을 제외하고 기본적으로 군집을 잘 구성한다.

# <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-04.png" width="450"/>

# ### K-평균 알고리즘

# 먼저 k 개의 센트로이드를 무작위로 선택한 후 수렴할 때까지 다음 과정 반복한다.
# 
# * 각 샘플을 가장 가까운 센트로이드에 할당
# * 군집별로 샘플의 평균을 계산하여 새로운 센트로이드 지정

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-05.png" width="700"/></div>

# **K-평균 알고리즘의 단점**
# 
# 군집의 크기가 서로 많이 다르면 잘 작동하지 않는다.
# 이유는 샘플과 센트로이드까지의 거리만 고려되기 때문이다.
# 또한 임의로 선택된 초기 센트로이드에 따라 매우 다른 군집이 생성될 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-06.png" width="750"/></div>

# **K-평균 모델 평가 기준: 관성**

# **관성**<font size='2'>intertia</font>은 각 샘플과 가장 가까운 센트로이드와의 거리의 제곱의 합이며,
# 각 군집이 센트로이드에 얼마나 가까이 모여있는가를 측정한다.
# 따라서 관성이 작을 수록 군집이 잘 구성되었다고 평가한다.
# 
# 훈련된 KMeans 모델의 경우 `inertia_` 속성에 관성 값이 저징되며,
# `score()` 메서드가 관성의 음숫값을 반환한다. 
# 이유는 점수(score)는 높을 수록 좋은 모델을 나타내도록 해야 하기 때문이다. 
# KMeans 모델은 훈련 과정 중에 다양한 초기화 과정을 실험하고 그 중에 가장 
# 좋은 군집 모델을 선택한다. 

# **K-평균++ 초기화 알고리즘**

# 센트로이드를 무작위로 초기화하는 대신 특정 확률분포를 이용하여 선택하여
# 센트로이드들 사이의 거리를 가능한 크게 선택하는 알고리즘이며,
# KMeans 모델의 기본 알고리즘으로 사용된다.

# **미니배치 k-평균**

# 미니배치를 사용해서 센트로이드를 조금씩 이동하는 k-평균 알고리즘이다. 
# 사이킷런의 `MiniBatchMeans` 모델이 지원한다. 
# 
# ```python
# minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10,
#                                    random_state=42)
# minibatch_kmeans.fit(X_memmap)
# ```

# 군집수가 많아질 수록 k-평균보다 서너 배 정도 빠르게 훈련되지만, 성능은 조금 낮다.
# 하지만 아래 왼쪽 그림에서 보면 군집수 k 가 커져도 성능 차이가 유지되지만
# 성능 자체가 좋아지므로 두 모델의 상대적 성능 차이는 점점 벌어짐을 의미한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-07.png" width="700"/></div>

# ### 최적의 군집수 찾기

# 군집수가 적절하지 않으면 좋지 않은 모델로 수렴할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-08.png" width="700"/></div>

# **방법 1: 관성과 군집수**

# 군집수 k가 증가할 수록 관성은 기본적으로 줄어들기에 관성만으로 모델을 평가할 수는 없다.
# 예를 들어 관성이 더 이상 획기적으로 줄어들지 않는 지점을 선택할 수는 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-09.png" width="600"/></div>

# 하지만 아래 그림에서 보듯이 반드시 좋은 모델을 보장하지는 않는다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-10.png" width="400"/></div>

# **방법 2: 실루엣 점수와 군집수**

# **실루엣 점수**<font size='2'>silhouette score</font> 샘플별 실루엣 계수의 평균값이다. 
# 샘플의 **실루엣 계수**<font size='2'>silhouette coefficient</font>는 
# 다음 식으로 계산된다.
# 
# $$\frac{b - a}{\max(a, b)}$$
# 
# $a$는 동일 군집 내의 다른 샘플과의 거리의 평균값이며,
# $b$ 가장 가까운 타 군집에 속하는 샘플들과의 거리의 평균값이다.
# 실루엣 계수는 -1과 1사이의 값이며, 다음 특성을 보여준다.
# 
# * 1에 가까운 값: 적절한 군집에 포함됨.
# * 0에 가까운 값: 군집 경계에 위치
# * -1에 가까운 값: 잘못된 군집에 포함됨

# 실루엣 점수가 높은 모델을 선택할 수 있다.
# 아래 그림에 따르면 `k=4`도 좋은 선택이 될 수 있다.
# 하지만 여기서는 `k=5` 가 가장 좋은 선택이다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-11.png" width="600"/></div>

# **방법 3: 실루엣 다이어그램과 군집수**

# **실루엣 다이어그램**은 군집별로 실루엣 계수들의 모아 놓은 그래프다.
# 군집별로 실루엣 계수를 내림차순으로 정렬하면 칼날 모양이 형성된다.
# 
# * 칼날 두께: 군집에 포함된 샘플 수
# * 칼날 길이: 군집에 포함된 각 샘플의 실루엣 계수
# * 빨강 파선: 실루엣 점수, 즉 실루엣 계수의 평균값이다. 
# 
# 좋은 군집 모델은 대부분의 칼날이 빨간 파선보다 길어야 하며,
# 칼날의 두께가 서로 비슷해야 한다. 
# 즉, 군집별 크기가 비슷해야 좋은 모델이다.
# 이런 기준으로 볼 때 `k=5` 가 가장 좋은 모델이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-12.png" width="700"/></div>

# ### K-평균의 한계

# * 최적의 모델을 구하기 위해 여러 번 학습해야 함.
# * 군집수를 미리 지정해야 함.
# * 군집의 크기나 밀집도가 다르거나, 원형이 아닐 경우 잘 작동하지 않음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-13.png" width="600"/></div>

# ### 군집화 활용

# **활용 예제 1: 이미지 색상 분할**

# **색상 분할**은 유사 색상으로 이루어진 군집으로 분할하는 것을 의미한다.
# 아래 그림은 무당벌레가 포함된 이미지를 대상으로 색상 수를 다르게 하면서
# 색상 분할을 시도한 결과를 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-14.png" width="600"/></div>

# **활용 예제 2: 차원축소**

# 샘플에 대해 각 센트로이드부터의 거리로 이루어진 어레이를 생성하면
# `n` 차원의 데이터셋을 `k` 차원의 데이터셋으로 변환한다. 
# 
# 이렇게 새로운 특성을 생성하여 다른 목적의 모델 훈련에 활용한다.
# 예를 들어, {ref}`ch:end2end` 에서 다룬 캘리포니아 주택 데이터셋의
# 위도, 경도 정보를 이용하여 가까운 구역으로 이루어진 군집을 생성하는 모델
# `ClusterSimilarity` 에 KMeans 모델이 활용되었다.

# **활용 예제 3: 준지도 학습**

# 레이블이 있는 데이터가 적고, 레이블이 없는 데이터가 많을 때 활용한다.

# :::{prf:example} 미니 MNIST 데이터셋
# 
# 미니 MNist 데이터셋은 1,797 개의 8x8 크기의 손글씨 이미지로 구성된다.
# :::

# 예를 들어, 미니 MNIST 데이터셋을 50개의 군집으로 나눈 후 각 군집에서 
# 센트로이드에 가장 가까운 샘플 50개를 대표 이미지로 선정한다.
# 선정된 50개 샘플만을 이용하여 분류 모델을 훈련해도 84.9%의 정확도가 달성된다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-15.png" width="500"/></div>

# **활용 예제 4: 레이블 전파**

# 대표 이미지의 레이블을 해당 군집의 모든 샘플로 전파한다.
# 하지만 동일한 군집에 속하지만 
# 센트로이드에서 샘플에 동일한 레이블을 전파하면 
# 전파된 레이블의 정확도가 매우 낮을 수 있다.
# 
# 따라서 센트로이드에 가장 멀리 떨어진 1%의 데이터를 이상치로 취급하여
# 제외시킨 다음에 레이블 전파를 진행하면
# 보다 좋은 성능이 분류 모델을 얻게 된다.
# 
# `sklearn.semi_supervised` 패키지는 다양한 레이블 전파 모델을 제공한다. 
# 
# - `LabelSpreading`
# - `LabelPropagation`
# - `SelfTrainingClassifier`

# ## DBSCAN

# * 연속적인 밀집 지역을 하나의 군집으로 설정.

# **사이킷런의 DBSCAN 모델**
# 
# * 두 개의 하이퍼파라미터 사용
#     * `eps`: $\varepsilon$-이웃 범위
#         * 주어진 기준값 $\varepsilon$ 반경 내에 위치한 샘플
#     * `min_samples`: $\varepsilon$ 반경 내에 위치하는 이웃의 수

# **핵심샘플과 군집**
# 
# * 핵심샘플: $\varepsilon$ 반경 내에 자신을 포함해서 `min-samples`개의 이웃을 갖는 샘플
# 
# * 군집: 핵심샘플로 이루어진 이웃들로 구성된 그룹

# **이상치**
# 
# * 핵심샘플이 아니면서 동시에 핵심샘플의 이웃도 아닌 샘플.

# **예제:** 반달모양 데이터 활용

# ---
# ```python
# from sklearn.cluster import DBSCAN
# 
# dbscan = DBSCAN(eps=0.05, min_samples=5)
# dbscan.fit(X)
# ```
# ---

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-16.png" width="600"/></div>

# **DBSCAN과 예측**
# 
# * `predict()` 메서드 지원하지 않음.
# 
# * 이유: `KNeighborsClassifier` 등 보다 좋은 성능의 분류 알고리즘 활용 가능.
# 
# * 아래 코드: 핵심샘플 대상 훈련.

# ---
# ```python
# from sklearn.neighbors import KNeighborsClassifier
# 
# knn = KNeighborsClassifier(n_neighbors=50)
# knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
# ```
# ---

# * 이후 새로운 샘플에 대한 예측 가능
# * 아래 그림은 새로운 4개의 샘플에 대한 예측을 보여줌.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-17.png" width="450"/></div>

# **이상치 판단**
# 
# * 위 예제에서, 두 군집으로부터 일정거리 이상 떨어진 샘플을 이상치로 간주 가능.
# 
# * 예를 들어, 양편 끝쪽에 위치한 두 개의 샘플이 이상치로 간주될 수 있음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-17a.png" width="450"/></div>

# **DBSCAN의 장단점**
# 
# * 매우 간단하면서 매우 강력한 알고리즘.
#     * 하이퍼파라미터: 단 2개
# 
# * 군집의 모양과 개수에 상관없음.
# 
# * 이상치에 안정적임.
# 
# * 군깁 간의 밀집도가 크게 다르면 모든 군집 파악 불가능.

# **계산복잡도**
# 
# * 시간복잡도: 약 $O(m\, \log m)$. 단, $m$은 샘플 수
# 
# * 공간복잡도: 사이킷런의 DBSCAN 모델은 $O(m^2)$의 메모리 요구.
#     * `eps`가 커질 경우.

# **기타 군집 알고리즘**

# * 응집 군집(병합 군집, agglomerative clustering)
# * BIRCH
# * 평균-이동
# * 유사도 전파
# * 스펙트럼 군집

# ## 가우시안 혼합 모델

# * 데이터셋이 여러 개의 혼합된 가우시안 분포를 따르는 샘플들로 구성되었다고 가정.
# 
# * 가우시안 분포 = 정규분포

# **정규분포 소개**
# 
# * 종 모양의 확률밀도함수를 갖는 확률분포

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-18.png" width="400"/></div>

# **군집**
# 
# * 하나의 가우시안 분포에서 생생된 모든 샘플들의 그룹
# * 일반적으로 타원형 모양.

# **예제**
# 
# * 아래 그림에서처럼 일반적으로 모양, 크기, 밀집도, 방향이 다름.
# * 따라서 각 샘플이 어떤 정규분포를 따르는지를 파악하는 게 핵심.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-13.png" width="600"/></div>

# **GMM 활용**
# 
# * 위 데이터셋에 `GaussianMixture` 모델 적용
# 
# * `n_components`: 군집수 지정
# 
# * `n_init`: 모델 학습 반복 횟수. 
#     * 파라미터(평균값, 공분산 등)를 무작위로 추정한 후 수렴할 때까지 학습시킴.

# ---
# ```python
# from sklearn.mixture import GaussianMixture
# 
# gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
# gm.fit(X)
# ```
# ---

# * 아래 그림은 학습된 모델을 보여줌.
#     * 군집 평균, 결정 경계, 밀도 등고선

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-19.png" width="500"/></div>

# **GMM 모델 규제**
# 
# * 특성수가 크거나, 군집수가 많거나, 샘플이 적은 경우 최적 모델 학습 어려움.
# * 공분산(covariance)에 규제를 가해서 학습을 도와줄 수 있음.
#     * `covariance_type` 설정.

# **covariance_type 옵션값**

# * full
#     * 아무런 제한 없음.
#     * 기본값임.

# * spherical
#     * 군집이 원형이라 가정. 
#     * 지름(분산)은 다를 수 있음.

# * diag
#     * 어떤 타원형도 가능.
#     * 단. 타원의 축이 좌표축과 평행하다고 가정.

# * tied
#     * 모든 군집의 동일 모양, 동일 크기, 동일 방향을 갖는다고 가정.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-20.png" width="600"/></div>

# ### 가우시안 혼합 모델 활용: 이상치 탐지

# * 밀도가 임곗값보다 낮은 지역에 있는 샘플을 이상치로 간주 가능.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-21.png" width="500"/></div>

# ### 가우션 혼합모델 군집수 지정

# * k-평균에서 사용했던 관성 또는 실루엣 점수 사용 불가.
#     * 군집이 타원형일 때 값이 일정하지 않기 때문.

# * 대신에 __이론적 정보 기준__ 을 최소화 하는 모델 선택 가능.

# **이론적 정보 기준**

# * BIC: Bayesian information criterion
# 
#     $$ \log(m)\, p - 2 \log (\hat L)$$

# * AIC: Akaike information criterion
# 
#     $$ 2\, p - 2 \log (\hat L)$$

# * $m$: 샘플 수
# * $p$: 모델이 학습해야 할 파라미터 수
# * $\hat L$: 모델의 가능도 함수의 최댓값

# * 학습해야 할 파라미터가 많을 수록 벌칙이 가해짐.
# * 데이터에 잘 학습하는 모델일 수록 보상을 더해줌.

# **군집수와 정보조건**

# * 아래 그림은 군집수 $k$와 AIC, BIC의 관계를 보여줌.
# * $k=3$이 최적으로 보임.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-22.png" width="600"/></div>

# ### 베이즈 가우시안 혼합 모델

# * 베이즈 확률통계론 활용

# **BayesianGaussianMixture 모델**

# * 최적의 군집수를 자동으로 찾아줌.
# * 단, 최적의 군집수보다 큰 수를 `n_components`에 전달해야 함.
#     * 즉, 군집에 대한 최소한의 정보를 알고 있다고 가정.
# * 자동으로 불필요한 군집 제거

# ---
# ```python
# from sklearn.mixture import BayesianGaussianMixture
# 
# bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
# bgm.fit(X)
# ```
# ---

# * 결과는 군집수 3개를 사용한 이전 결과와 거의 동일.
# * 군집수 확인 가능

# ```python
# >>> np.round(bgm.weights_, 2)
# array([0.4 , 0.21, 0.4 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])
# ```

# **사전 믿음**

# * 군집수가 어느 정도일까를 나타내는 지수
# * `weight_concentration_prior` 하이퍼파라미터
#     * `n_components`에 설정된 군집수에 대한 규제로 사용됨.
#     * 작은 값이면 특정 군집의 가중치를 0에 가깝게 만들어 군집수를 줄이도록 함.
#     * 즉, 큰 값일 수록 `n_components`에 설정된 군집수가 유지되도록 함.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-24.png" width="600"/></div>

# **가우시안 혼합 모델의 장단점**
# 
# * 타원형 군집에 잘 작동.
# 
# * 하지만 다른 모양을 가진 데이터셋에서는 성능 좋지 않음.
# 
# * 예제: 달모양 데이터에 적용하는 경우
#     * 억지로 타원을 찾으려 시도함.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch09/homl09-23.png" width="700"/></div>

# ### 이상치 탐지와 특이치 탐지를 위한 다른 알고리즘

# * PCA
# * Fast-MCD
# * 아이솔레이션 포레스트
# * LOF
# * one-class SVM
