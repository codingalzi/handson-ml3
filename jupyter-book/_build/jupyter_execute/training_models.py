#!/usr/bin/env python
# coding: utf-8

# (ch:trainingModels)=
# # 모델 훈련

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 모델 훈련](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_training_models.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**
# 
# * 선형 회귀 모델 구현
#     * 선형대수 활용
#     * 경사하강법 활용
# * 경사하강법 종류
#     * 배치 경사하강법
#     * 미니배치 경사하강법
#     * 확률적 경사하강법(SGD)
# * 다항 회귀: 비선형 회귀 모델
# * 학습 곡선: 과소, 과대 적합 감지
# * 모델 규제: 과대 적합 방지
# * 로지스틱 회귀와 소프트맥스 회귀: 분류 모델

# **목표**
# 
# 모델 훈련의 기본 작동 과정과 원리를 살펴보며,
# 이를 통해 다음 사항들에 대한 이해를 넓힌다.
# 
# - 적정 모델 선택
# - 적정 훈련 알고리즘 선택
# - 적정 하이퍼파라미터 선택
# - 디버깅과 오차 분석
# - 신경망 구현 및 훈련 과정 이해

# ## 선형 회귀

# **선형 회귀 예제: 1인당 GDP와 삶의 만족도**

# {numref}`%s 절 <sec:model_based_learning>`에서 1인당 GDP와 삶의 만족도 사이의 
# 관계를 다음 1차 함수로 표현할 수 있었다.
# 
# $$(\text{삶의만족도}) = \theta_0 + \theta_1\cdot (\text{1인당GDP})$$
# 
# 즉, 1인당 GDP가 주어지면 위 함수를 이용하여 삶의 만족도를 예측하였다.
# 주어진 1인당 GDP를 **입력 특성**<font size="2">input feature</font> $x$, 
# 예측된 삶의 만족도를 **예측값** $\hat y$ 라 하면 다음 식으로 변환된다.
# 
# $$\hat y = \theta_0 + \theta_1\cdot x_1$$
# 
# 절편 $\theta_0$ 와 기울기 $\theta_1$ 은 (선형) 모델의 **파라미터**<font size="2">weight parameter</font>이다.
# 머신러닝에서는 절편은 **편향**<font size="2">bias</font>, 
# 기울기는 **가중치**<font size="2">weight</font> 라 부른다.
# 
# 따라서 1인당 GDP와 삶의 만족도 사이의 선형 관계를 모델로 구현하려면
# 적절한 하나의 편향과 하나의 가중치, 즉 총 2개의 파라미터를 결정해야 한다.

# **선형 회귀 예제: 캘리포니아 주택 가격 예측**

# 반면에 {numref}`%s 장 <ch:end2end>`의 캘리포니아 주택 가격 예측 선형 회귀 모델은
# 24개의 입력 특성을 사용하는 다음 함수를 이용한다.
# 
# $$\hat y = \theta_0 + \theta_1\cdot x_1 + \cdots + \theta_{24}\cdot x_{24}$$
# 
# * $\hat y$: 예측값
# * $x_i$: 구역의 $i$ 번째 특성값(위도, 경도, 중간소득, 가구당 인원 등등등)
# * $\theta_0$: 편향
# * $\theta_i$: $i$ 번째 특성에 대한 (가중치) 파라미터, 단 $i > 0$.
# 
# 따라서 캘리포니아의 구역별 중간 주택 가격을 예측하는 선형 회귀 모델을 구하려면 
# 적절한 하나의 편향과 24개의 가중치,
# 즉 총 25개의 파라미터를 결정해야 한다.

# **선형 회귀 함수**

# 이를 일반화하면 다음과 같다.
# 
# $$\hat y = \theta_0 + \theta_1\cdot x_1 + \cdots + \theta_{n}\cdot x_{n}$$
# 
# * $\hat y$: 예측값
# * $n$: 특성 수
# * $x_i$: 구역의 $i$ 번째 특성값
# * $\theta_0$: 편향
# * $\theta_j$: $j$ 번째 특성에 대한 (가중치) 파라미터(단, $1 \le j \le n$)
# 
# 일반적으로 선형 회귀 모델을 구현하려면
# 한 개의 편향과 $n$ 개의 가중치, 즉 총 $(1+n)$ 개의 파라미터를 결정해야 한다.

# **벡터 표기법**
# 
# 예측값을 벡터의 **내적**<font size="2">inner product</font>으로 표현할 수 있다.
# 
# $$
# \hat y
# = h_\theta (\mathbf{x})
# = \mathbf{\theta} \cdot \mathbf{x}
# $$
# 
# * $h_\theta(\cdot)$: 예측 함수, 즉 모델의 `predict()` 메서드.
# * $\mathbf{x} = (1, x_1, \dots, x_n)$
# * $\mathbf{\theta} = (\theta_0, \theta_1, \dots, \theta_n)$

# **2D 어레이 표기법**
# 
# 머신러닝에서는 입력 벡터와 파라미터 벡터를 일반적으로 아래 모양의 행렬로 나타낸다.
# 
# $$
# \mathbf{x}=
# \begin{bmatrix}
# 1 \\
# x_1 \\
# \vdots \\
# x_n
# \end{bmatrix},
# \qquad
# \mathbf{\theta}=
# \begin{bmatrix}
# \theta_0\\
# \theta_1 \\
# \vdots \\
# \theta_n
# \end{bmatrix}
# $$
# 
# 따라서 예측값은 다음과 같이 행렬 연산으로 표기된다.
# 단, $A^T$ 는 행렬 $A$의 전치행렬을 가리킨다.
# 
# $$
# \hat y
# = h_\theta (\mathbf{x})
# = \mathbf{\theta}^{T} \mathbf{x}
# $$

# **선형 회귀 모델의 행렬 연산 표기법**

# $\mathbf{X}$가 전체 입력 데이터셋을 가리키는 (m, 1+n) 모양의 2D 어레이, 즉 행렬이라 하자.
# - $m$: 입력 데이터셋의 크기.
# - $n$: 특성 수
# 
# 그러면 $\mathbf{X}$ 는 다음과 같이 표현된다.
# 단, $\mathbf{x}_j^{(i)}$ 는 $i$-번째 입력 샘플의 $j$-번째 특성값을 가리킨다.
# 
# $$
# \mathbf{X}= 
# \begin{bmatrix} 
# [1, \mathbf{x}_1^{(1)}, \dots, \mathbf{x}_n^{(1)}] \\
# \vdots \\
# [1, \mathbf{x}_1^{(m)}, \dots, \mathbf{x}_n^{(m)}] \\
# \end{bmatrix}
# $$

# 결론적으로 모든 입력값에 대한 예측값을 하나의 행렬식으로 표현하면 다음과 같다.

# $$
# \begin{bmatrix}
# \hat y_1 \\
# \vdots \\
# \hat y_m
# \end{bmatrix}
# = 
# \begin{bmatrix} 
# [1, \mathbf{x}_1^{(1)}, \dots, \mathbf{x}_n^{(1)}] \\
# \vdots \\
# [1, \mathbf{x}_1^{(m)}, \dots, \mathbf{x}_n^{(m)}] \\
# \end{bmatrix}
# \,\, 
# \begin{bmatrix}
# \theta_0\\
# \theta_1 \\
# \vdots \\
# \theta_n
# \end{bmatrix}
# $$

# 간략하게 줄이면 다음과 같다.

# $$
# \hat{\mathbf y} = \mathbf{X}\, \mathbf{\theta}
# $$

# 위 식에 사용된 기호들의 의미와 어레이 모양은 다음과 같다.
# 
# | 데이터 | 어레이 기호           |     어레이 모양(shape) | 
# |:-------------:|:-------------:|:---------------:|
# | 예측값 | $\hat{\mathbf y}$  | $(m, 1)$ |
# | 입력 데이터셋 | $\mathbf X$   | $(m, 1+n)$     |
# | 파라미터 | $\mathbf{\theta}$      | $(1+n, 1)$ |

# **비용함수: 평균 제곱 오차(MSE)**

# 회귀 모델은 훈련 중에 **평균 제곱 오차**<font size="2">mean squared error</font>(MSE)를 이용하여
# 성능을 평가한다.

# $$
# \mathrm{MSE}(\mathbf{\theta}) := \mathrm{MSE}(\mathbf X, h_{\mathbf{\theta}}) = 
# \frac 1 m \sum_{i=1}^{m} \big(\mathbf{\theta}^{T}\, \mathbf{x}^{(i)} - y^{(i)}\big)^2
# $$

# 최종 목표는 훈련셋이 주어졌을 때 $\mathrm{MSE}(\mathbf{\theta})$가 최소가 되도록 하는 
# $\mathbf{\theta}$를 찾는 것이다.
# 
# * 방식 1: 정규방정식 또는 특이값 분해(SVD) 활용
#     * 드물지만 수학적으로 비용함수를 최소화하는 $\mathbf{\theta}$ 값을 직접 계산할 수 있는 경우 활용
#     * 계산복잡도가 $O(n^2)$ 이상인 행렬 연산을 수행해야 함. 
#     * 따라서 특성 수($n$)이 큰 경우 메모리 관리 및 시간복잡도 문제때문에 비효율적임.
# 
# * 방식 2: 경사하강법
#     * 특성 수가 매우 크거나 훈련 샘플이 너무 많아 메모리에 한꺼번에 담을 수 없을 때 적합
#     * 일반적으로 선형 회귀 모델 훈련에 적용되는 기법

# **정규 방정식**
# 
# 비용함수를 최소화 하는 $\theta$를 
# 정규 방정식<font size="2">normal equation</font>을 이용하여 
# 아래와 같이 바로 계산할 수 있다.
# 단, $\mathbf{X}^T\, \mathbf{X}$ 의 역행렬이 존재해야 한다.
# 
# $$
# \hat{\mathbf{\theta}} = 
# (\mathbf{X}^T\, \mathbf{X})^{-1}\, \mathbf{X}^T\, \mathbf{y}
# $$

# **SVD(특잇값 분해) 활용**
# 
# 그런데 행렬 연산과 역행렬 계산은 계산 복잡도가 $O(n^{2.4})$ 이상이며
# 항상 역행렬 계산이 가능한 것도 아니다.
# 반면에, **특잇값 분해**를 활용하여 얻어지는 
# **무어-펜로즈(Moore-Penrose) 유사 역행렬** $\mathbf{X}^+$은 항상 존재하며
# 계산 복잡도가 $O(n^2)$ 로 보다 빠른 계산을 지원한다.
# 또한 다음이 성립한다.
# 
# $$
# \hat{\mathbf{\theta}} = 
# \mathbf{X}^+\, \mathbf{y}
# $$

# ## 경사 하강법

# 훈련 세트를 이용한 훈련 과정 중에 가중치 파라미터를 조금씩 반복적으로 조정한다. 
# 이때 비용 함수의 크기를 줄이는 방향으로 조정한다.

# 경사 하강법 이해를 위해 다음 개념들을 충분히 이해하고 있어야 한다.

# **최적 학습 모델**
# 
# 비용 함수를 최소화하는 또는 효용 함수를 최대화하는 파라미터를 사용하는 모델이며,
# 최종적으로 훈련시킬 대상이다.

# **파라미터**
# 
# 선형 회귀 모델에 사용되는 편향과 가중치 파라미터처럼 모델 훈련중에 학습되는 파라미터를 가리킨다.

# **비용 함수**
# 
# 평균 제곱 오차(MSE)처럼 모델이 얼마나 나쁜가를 측정하는 함수다.

# **전역 최솟값**
# 
# 비용 함수의 최솟값이다. 

# **비용 함수의 그레이디언트**
# 
# MSE를 비용함수로 사용하는 경우 $\textrm{MSE}(\mathbf{\theta})$ 함수의 $\mathbf{\mathbf{\theta}}$ 에 대한 그레이디언트를 사용한다.
# 그레이디언트 벡터는 방향과 크기에 대한 정보를 제공하며, 
# 그레이디언트가 가리키는 방향의 __반대 방향__으로 움직이면
# 빠르게 전역 최솟값에 접근한다.
# 
# $$
# \nabla_\mathbf{\theta} \textrm{MSE}(\mathbf{\theta}) =
# \begin{bmatrix}
#     \frac{\partial}{\partial \mathbf{\theta}_0} \textrm{MSE}(\mathbf{\theta}) \\
#     \frac{\partial}{\partial \mathbf{\theta}_1} \textrm{MSE}(\mathbf{\theta}) \\
#     \vdots \\
#     \frac{\partial}{\partial \mathbf{\theta}_n} \textrm{MSE}(\mathbf{\theta})
# \end{bmatrix}
# $$
# 
# **참고:** 산에서 가장 경사가 급한 길을 따를 때 가장 빠르게 하산하는 원리와 동일하다.

# **학습률($\eta$)**
# 
# 훈련 과정에서의 비용함수의 파라미터($\mathbf{\theta}$)를 조정할 때 사용하는 조정 비율이다.

# **허용오차**
# 
# 비용함수의 그레이디언트 벡터의 크기가 허용오차보다 작아지면 모델의 훈련을 종료시키는 데에 
# 허용오차<font size="2">tolerance</font>를 사용한다.

# ### 선형 회귀 모델 파라미터 조정

# 선형회귀 모델 파라미터를 조정하는 과정을 이용하여 경사 하강법의 기본 아이디어를 설명한다.
# 
# 먼저 $\mathrm{MSE}(\mathbf{\theta})$ 는 $\mathbf{\theta}$ 에 대한 2차 함수임에 주의한다.
# 여기서는 $\mathbf{\theta}$ 가 하나의 파라미터로 구성되었다고 가정한다.
# 따라서 $\mathrm{MSE}(\mathbf{\theta})$의 그래프는 포물선이 된다.
# 
# $$
# \mathrm{MSE}(\mathbf{\theta}) =
# \frac 1 m \sum_{i=1}^{m} \big(\mathbf{\theta}^{T}\, \mathbf{x}^{(i)} - y^{(i)}\big)^2
# $$

# 경사 하강법은 다음 과정으로 이루어진다. 
# 
# 1. $\mathbf{\theta}$를 임의의 값으로 지정한 후 훈련을 시작한다.
# 
# 1. 아래 단계를 $\nabla_\theta \textrm{MSE}(\theta)$ 의 크기가 허용오차보다 작아질 때까지 반복한다.
#     1. 지정된 수의 훈련 샘플을 이용한 학습.
#     2. $\mathrm{MSE}(\mathbf{\theta})$ 계산.
#     3. 이전 $\mathbf{\theta}$에서 $\nabla_\mathbf{\theta} \textrm{MSE}(\mathbf{\theta})$ 와
#         학습률 $\eta$를 곱한 값 빼기.
# 
#         $$\theta^{(\text{new})} = \theta^{(\text{old})}\, -\, \eta\cdot \nabla_\theta \textrm{MSE}(\theta^{(\text{old})})$$    

# 아래 그림이 경사 하강법을 잘 설명한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-01.png" width="500"/></div>

# 학습률이 너무 작거나 크면 비용 함수의 전역 최솟값에 수렴하지 않을 수 있다.
# 
# - 학습률이 너무 작은 경우: 비용 함수가 전역 최소값에 너무 느리게 수렴.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-02.png" width="500"/></div>

# * 학습률이 너무 큰 경우: 비용 함수가 수렴하지 않음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-03.png" width="500"/></div>

# 선형 회귀의 경우 학습률이 너무 크거나 너무 작지 않으면 적정한 시간 내에 비용 함수의
# 전역 최솟값에 수렴하는 것이 보장된다.
# 하지만 선형 회귀가 아닌 경우에는 시작점에 따라 지역 최솟값에 수렴하거나 정체될 수 있음을
# 아래 그림이 잘 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-04.png" width="500"/></div>

# **특성 스케일링의 중요성**

# 특성들의 스켈일을 통일시키면 보다 빠른 학습이 이루어지는 이유를 
# 아래 그림이 설명한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-04a.png" width="500"/></div>

# ### 경사 하강법 종류

# 모델을 지정할 때 지정하는 배치 크기, 에포크 수, 허용오차에 따라 
# 세 종류로 나뉜다.
# 
# * 에포크<font size="2">epoch</font>: 입력 데이터셋에 포함된 모든 데이터를 대상으로 예측값을 계산하는 과정.
# 
# * 허용오차<font size="2">tolerance</font>: 비용함수의 그레이디언트 벡터의 크기가 허용오차보다 작아지면 훈련 종료
# 
# * 배치 크기<font size="2">batch size</font>: 파라미터를 업데이트하기 위해, 즉 그레이디언트 벡터를 계산하기 위해 사용되는 훈련 샘플 수. 
# 
# **참고:** 지정된 배치 크기의 샘플에 대해 예측을 한 후에 경사하강법을 이용하여 파라미터를 조정하는 단계를
# 스텝<font size="2">step</font>라 하며, 다음이 성립힌다.
# 
#     스텝 크기 = (훈련 샘플 수) / (배치 크기)
# 
# 예를 들어, 훈련 세트의 크기가 1,000이고 배치 크기가 10이면, 에포크 당 100번의 스텝이 실행된다.

# #### 배치 경사 하강법

# 에포크마다 그레이디언트를 계산하여 파라미터를 조정한다.
# 즉, 배치의 크기가 전체 입력 데이터셋의 크기와 같고 따라서 스텝의 크기는 1이다.

# * 에포크와 허용오차
# 
#     * 에포크 수는 크게 설정한 후 허용오차를 지정하여 학습 시간 제한 필요.
#         이유는 포물선의 최솟점에 가까워질 수록 그레이디언트 벡터의 크기가 0에 수렴하기 때문임.
# 
#     * 허용오차와 에포크 수는 서로 반비례의 관계임. 즉, 오차를 1/10로 줄이려면 에포크 수를 10배 늘려야함.

# * 단점
# 
#     * 훈련 세트가 크면 그레이디언트를 계산하는 데에 많은 시간 필요
#     * 아주 많은 데이터를 저장해야 하는 메모리 문제도 발생 가능

# * __주의사항__
# 
#     * 사이킷런은 배치 경사 하강법을 활용한 선형 회귀 지원하지 않음.
#         (책 176쪽, 표 4-1에서 사이킷런의 SGDRegressor가 배치 경사 하강법을 지원한다고 __잘못__ 명시됨.)

# #### 학습율과 경사 하강법의 관계

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-04b.png" width="700"/></div>

# ### 4.2.2 확률적 경사 하강법

# **확률적 경사 하강법**
# 
# 배치 크기가 1이다. 
# 즉, 하나의 훈련 샘플에 대해 예측을 하면 그 결과를 이용하여 그레이디언트를 계산하고 파라미터를 조정한다.

# * 장점
# 
#     * 매우 큰 훈련 세트를 다룰 수 있음.
#         예를 들어, 외부 메모리(out-of-core) 학습을 활용할 수 있음
#     * 학습 과정이 매우 빠르며 파라미터 조정이 불안정 할 수 있기 때문에 지역 최솟값에 상대적으로 덜 민감

# * 단점: 학습 과정에서 파라미터의 동요가 심해서 경우에 따라 전역 최솟값에 수렴하지 못하고 계속해서 발산할 가능성도 높음

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-04c.png" width="400"/></div>

# 처음 20 단계 동안의 SGD 학습 내용: 모델이 수렴하지 못함을 확인할 수 있음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-04d.png" width="600"/></div>

# #### 학습 스케줄
# 
# * 요동치는 파라미터를 제어하기 위해 학습률을 학습 과정 동안 천천히 줄어들게 만들 수 있음

# * 주의사항
#     * 학습률이 너무 빨리 줄어들면, 지역 최솟값에 갇힐 수 있음
#     * 학습률이 너무 느리게 줄어들면 전역 최솟값에 제대로 수렴하지 못하고 맴돌 수 있음
#     

# * 학습 스케줄(learning schedule)
#     * 훈련이 지속될 수록 학습률을 조금씩 줄이는 기법
#     * 에포크, 훈련 샘플 수, 학습되는 샘플의 인덱스에 따른 학습률 지정

# #### 사이킷런의 `SGDRegressor`
# 
# * 경사 하강법 사용
# 
# * 사용되는 하이퍼파라미터
#   * `max_iter=1000`: 에포크 수 제한
#   * `tol=1e-3`: 하나의 에포크가 지날 때마다 0.001보다 적게 손실이 줄어들 때까지 훈련.
#   * `eta0=0.1`: 학습 스케줄 함수에 사용되는 매개 변수. 일종의 학습률.
#   * `penalty=l2`: 규제 사용 여부 결정 (추후 설명)

# ### 4.2.3 미니배치 경사 하강법

# **미니배치 경사 하강법**
# 
# 배치 크기가 2에서 수백 사이로 정해지며, 
# 최적 배치 크기는 경우에 따라 다르다.

# * 장점
# 
#     * 배치 크기를 어느 정도 크게 하면 확률적 경사 하강법(SGD) 보다 파라미터의 움직임이 덜 불규칙적이 됨
#     * 반면에 배치 경사 하강법보다 빠르게 학습
#     * 학습 스케줄 잘 활용하면 최솟값에 수렴함.

# * 단점
# 
#     * SGD에 비해 지역 최솟값에 수렴할 위험도가 보다 커짐.

# ### 경사 하강법 비교

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-05.png" width="600"/></div>

# ### 선형 회귀 알고리즘 비교
# 
# 
# | 알고리즘   | 많은 샘플 수 | 외부 메모리 학습 | 많은 특성 수 | 하이퍼 파라미터 수 | 스케일 조정 | 사이킷런 지원 |
# |:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
# | 정규방정식  | 빠름       | 지원 안됨      |  느림        | 0          | 불필요    | 지원 없음      |
# | SVD      | 빠름       | 지원 안됨      |  느림        | 0          | 불필요     | LinearRegression     |
# | 배치 GD   | 느림       | 지원 안됨      |  빠름        | 2          | 필요      | LogisticRegression      |
# | SGD      | 빠름       | 지원          |  빠름        | >= 2       | 필요      | SGDRegressor |
# | 미니배치 GD | 빠름       | 지원         |  빠름        | >=2        | 필요      | 지원 없음      |

# ## 4.3 다항 회귀

# * 다항 회귀(polynomial regression)란?
#     * 선형 회귀를 이용하여 비선형 데이터를 학습하는 기법
#     * 즉, 비선형 데이터를 학습하는 데 선형 모델 사용을 가능하게 함.

# * 기본 아이디어
#     * 특성들의 조합 활용
#     * 특성 변수들의 다항식을 조합 특성으로 추가

# ### 선형 회귀 vs. 다항 회귀

# #### 선형 회귀: 1차 선형 모델
# 
# $$\hat y = \theta_0 + \theta_1\, x_1$$

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-06.png" width="600"/></div>

# #### 다항 회귀: 2차 다항식 모델
# 
# $$\hat y = \theta_0 + \theta_1\, x_1 + \theta_2\, x_1^{2}$$

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-07.png" width="600"/></div>

# #### 사이킷런의 `PolynomialFeatures` 변환기

# * 주어진 특성들의 거듭제곱과 특성들 사이의 곱셈을 실행하여 특성을 추가하는 기능 제공

# * `degree=d`: 몇 차 다항식을 활용할지 지정하는 하이퍼파라미터
# 
#     * 이전 예제: $d=2$으로 지정하여  $x_1^2$에 대한 특성 변수가 추가됨.

# * 예제: $n=2, d=3$인 경우에 $(x_1+x_2)^2$과 $(x_1+x_2)^3$의 항목에 해당하는 7개 특성 추가
# 
# $$x_1^2,\,\, x_1 x_2,\,\, x_2^2,\,\, x_1^3,\,\, x_1^2 x_2,\,\, x_1 x_2^2,\,\, x_2^3$$

# ## 4.4 학습 곡선

# ### 과소적합/과대적합 판정
# 
# * 예제: 선형 모델, 2차 다항 회귀 모델, 300차 다항 회귀 모델 비교
# 
# * 다항 회귀 모델의 차수에 따라 훈련된 모델이 훈련 세트에 과소 또는 과대 적합할 수 있음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-08.png" width="600"/></div>

# ### 교차 검증 vs. 학습 곡선
# 
# * 교차 검증(2장)
#     * 과소적합: 훈련 세트와 교차 검증 점수 모두 낮은 경우
#     * 과대적합: 훈련 세트에 대한 검증은 우수하지만 교차 검증 점수가 낮은 경우

# * 학습 곡선 살피기
#     * 학습 곡선: 훈련 세트와 검증 세트에 대한 모델 성능을 비교하는 그래프
#     * 학습 곡선의 모양에 따라 과소적합/과대적합 판정 가능

# ### 과소적합 모델의 학습 곡선 특징

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-09.png" width="600"/></div>

# * 훈련 데이터(빨강)에 대한 성능
#     * 훈련 세트가 커지면서 RMSE(평균 제곱근 오차)가 커짐
#     * 훈련 세트가 어느 정도 커지면 더 이상 RMSE가 변하지 않음

# * 검증 데이터(파랑)에 대한 성능
#     * 검증 세트에 대한 성능이 훈련 세트에 대한 성능과 거의 비슷해짐

# 
# ### 과대적합 모델의 학습 곡선 특징

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-10.png" width="600"/></div>

# * 훈련 데이터(빨강)에 대한 성능: 훈련 데이터에 대한 평균 제곱근 오차가 매우 낮음.

# * 검증 데이터(파랑)에 대한 성능: 훈련 데이터에 대한 성능과 차이가 크게 벌어짐.

# * 과대적합 모델 개선법: 훈련 데이터 추가

# ### 편향 vs 분산

# * 편향(bias)
#     - 실제로는 2차원 모델인데 1차원 모델을 사용하는 경우처럼 잘못된 가정으로 인해 발생.
#     - 과소적합 발생 가능성 높음.

# * 분산(variance)
#     - 모델이 훈련 데이터에 민감하게 반응하는 정도
#     - 고차 다항 회귀 모델의 경우 분산이 높아질 수 있음.
#     - 과대적합 발생 가능성 높음.

# * 편향과 분산의 트레이드 오프
#     - 복잡한 모델일 수록 편향을 줄어들지만 분산을 커짐.

# ### 모델 일반화 오차

# * 훈련 후에 새로운 데이터 대한 예측에서 발생하는 오차를 가리키며 세 종류의 오차가 있음.

# - 편향

# - 분산

# - 줄일 수 없는 오차
#     - 데이터 자체가 갖고 있는 잡음(noise) 때문에 발생.
#     - 잡음을 제거해야 오차를 줄일 수 있음.

# ## 4.5 규제를 사용하는 선형 모델

# ### 자유도와 규제

# * 자유도(degree of freedom): 학습 모델 결정에 영향을 주는 요소(특성)들의 수
#     * 단순 선형 회귀의 경우: 특성 수
#     * 다항 선형 회귀 경우: 차수

# * 규제(regularization): 자유도 제한
#     * 단순 선형 회귀 모델에 대한 규제: 가중치 역할 제한
#     * 다항 선형 회귀 모델에 대한 규제: 차수 줄이기

# ### 가중치를 규제하는 선형 회귀 모델

# * 릿지 회귀

# * 라쏘 회귀

# * 엘라스틱넷

# ### 규제 적용 주의사항

# 규제항은 훈련 과정에만 사용된다. 테스트 과정에는 다른 기준으로 성능을 평가한다.
# 
# * 훈련 과정: 비용 최소화 목표

# * 테스트 과정: 최종 목표에 따른 성능 평가
#     * 예제: 분류기의 경우 재현율/정밀도 기준으로 성능 평가

# ### 4.5.1 릿지 회귀

# * 비용함수
# 
# $$J(\theta) = \textrm{MSE}(\theta) + \alpha \, \frac{1}{2} \sum_{i=1}^{n}\theta_i^2$$

# * $\alpha$(알파): 규제 강도 지정. 
#     $\alpha=0$이면 규제가 전혀 없는 기본 선형 회귀

# * $\alpha$가 커질 수록 가중치의 역할이 줄어듦. 
#     비용을 줄이기 위해 가중치를 작게 유지하는 방향으로 학습

# * $\theta_0$은 규제하지 않음

# * 주의사항: 특성 스케일링 전처리를 해야 성능이 좋아짐.

# ### 4.5.2 라쏘 회귀

# * 비용함수
# 
# $$J(\theta) = \textrm{MSE}(\theta) + \alpha \, \sum_{i=1}^{n}\mid\theta_i\mid$$

# * $\alpha$(알파): 규제 강도 지정.
#     $\alpha=0$이면 규제가 전혀 없는 기본 선형 회귀

# * $\theta_i$: 덜 중요한 특성을 무시하기 위해 $\mid\theta_i\mid$가 0에 수렴하도록 학습 유도.

# * $\theta_0$은 규제하지 않음

# #### 라쏘 회귀 대 릿지 회귀 비교

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/lasso_vs_ridge_plot.png" width="600"/></div>

# ### 4.5.3 엘라스틱넷

# * 비용함수
# 
# $$J(\theta) = \textrm{MSE}(\theta) + r\, \alpha \, \sum_{i=1}^{n}\mid\theta_i\mid + \,\frac{1-r}{2}\, \alpha\, \sum_{i=1}^{n}\theta_i^2$$

# * 릿지 회귀와 라쏘 회귀를 절충한 모델

# * 혼합 비율 $r$을 이용하여 릿지 규제와 라쏘 규제를 적절하게 조절

# ### 규제 사용 방법

# * 대부분의 경우 약간이라도 규제 사용 추천

# * 릿지 규제가 기본

# * 유용한 속성이 많지 않다고 판단되는 경우 
#     * 라쏘 규제나 엘라스틱넷 활용 추천
#     * 불필요한 속성의 가중치를 0으로 만들기 때문

# * 특성 수가 훈련 샘플 수보다 크거나 특성 몇 개가 강하게 연관되어 있는 경우
#     * 라쏘 규제는 적절치 않음.
#     * 엘라스틱넷 추천

# ### 4.5.4 조기 종료

# * 모델의 훈련 세트에 대한 과대 적합 방지를 위해 훈련을 적절한 시기에 중단시키기.

# * 조기 종료: 검증 데이터에 대한 손실이 줄어 들다가 다시 커지는 순간 훈련 종료

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-11.png" width="600"/></div>

# * 확률적 경사 하강법 등의 경우 손실 곡선의 진동 발생. 
#     검증 손실이 한동안 최솟값보다 높게 유지될 때 훈련 멈춤. 최소 검증 손실 모델 확인.

# ## 4.6 로지스틱 회귀

# 회귀 모델을 분류 모델로 활용할 수 있다. 

# * 이진 분류: 로지스틱 회귀

# * 다중 클래스 분류: 소프트맥스 회귀

# ### 4.6.1 확률 추정

# * 시그모이드 함수
# 
# $$\sigma(t) = \frac{1}{1 + e^{-t}}$$

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-12.png" width="600"/></div>

# * 로지스틱 회귀 모델에서 샘플 $\mathbf x$가 양성 클래스에 속할 확률
# 
# $$\hat p = h_\theta (\mathbf x)
# = \sigma(\theta_0 + \theta_1\, x_1 + \cdots + \theta_n\, x_n)$$

# #### 예측값
# 
# $$
# \hat y = 
# \begin{cases}
# 0 & \text{if}\,\, \hat p < 0.5 \\
# 1 & \text{if}\,\, \hat p \ge 0.5
# \end{cases}
# $$

# * 양성 클래스인 경우: 
# 
# $$\theta_0 + \theta_1\, x_1 + \cdots + \theta_n\, x_n \ge 0$$

# * 음성 클래스인 경우: 
# 
# $$\theta_0 + \theta_1\, x_1 + \cdots + \theta_n\, x_n < 0$$

# ### 4.6.2 훈련과 비용함수

# * 비용함수: 로그 손실(log loss) 함수 사용
# 
# $$
# J(\theta) = 
# - \frac{1}{m}\, \sum_{i=1}^{m}\, [y^{(i)}\, \log(\,\hat p^{(i)}\,) + (1-y^{(i)})\, \log(\,1 - \hat p^{(i)}\,)]
# $$

# * 모델 훈련: 위 비용함수에 대해 경사 하강법 적용

# #### 로그 손실 함수 이해

# * 틀린 예측을 하면 손실값이 많이 커짐

# $$
# - [y\, \log(\,\hat p\,) + (1-y)\, \log(\,1 - \hat p\,)]
# $$

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-12-10a.png" width="700"/></div>

# #### 로그 손실 함수의 편도 함수

# $$
# \dfrac{\partial}{\partial \theta_j} \text{J}(\boldsymbol{\theta}) = \dfrac{1}{m}\sum\limits_{i=1}^{m}\left(\mathbf{\sigma(\boldsymbol{\theta}}^T \mathbf{x}^{(i)}) - y^{(i)}\right)\, x_j^{(i)}
# $$

# * 편도 함수가 선형 회귀의 경우와 매우 비슷한 것에 대한 확률론적 근거가 있음.

# * __참고:__ [앤드류 응(Andrew Ng) 교수의 Stanford CS229](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

# ### 4.6.3 결정 경계

# #### 예제: 붓꽃 데이터셋

# * 꽃받침(sepal)과 꽃입(petal)과 관련된 4개의 특성 사용
#     * 꽃받침 길이
#     * 꽃받침 너비
#     * 꽃잎 길이
#     * 꽃잎 너비

# * 타깃: 세 개의 품종
#     * 0: Iris-Setosa(세토사)
#     * 1: Iris-Versicolor(버시컬러)
#     * 2: Iris-Virginica(버지니카)

# #### 꽃잎의 너비를 기준으로 Iris-Virginica 여부 판정하기
# 
# * 결정경계: 약 1.6cm

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-14.png" width="700"/></div>

# #### 꽃잎의 너비와 길이를 기준으로 Iris-Virginica 여부 판정하기
# 
# * 결정경계: 검정 점선

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-15.png" width="700"/></div>

# ### 로지스틱 회귀 규제하기

# * 하이퍼파라미터 `penalty`와 `C` 이용

# * `penalty`
#     * `l1`, `l2`, `elasticnet` 세 개중에 하나 사용.
#     * 기본은 `l2`, 즉, $\ell_2$ 규제를 사용하는 릿지 규제.
#     * `elasticnet`을 선택한 경우 `l1_ration` 옵션 값을 함께 지정.

# * `C`
#     * 릿지 또는 라쏘 규제 정도를 지정하는 $\alpha$의 역수에 해당. 
#     * 따라서 0에 가까울 수록 강한 규제 의미.

# ### 4.6.4 소프트맥스(softmax) 회귀

# * 로지스틱 회귀 모델을 일반화하여 다중 클래스 분류를 지원하도록 한 회귀 모델

# * **다항 로지스틱 회귀** 라고도 불림

# * 주의사항: 소프트맥스 회귀는 다중 출력 분류 지원 못함. 
#     예를 들어, 하나의 사진에서 여러 사람의 얼굴 인식 불가능.

# #### 소프트맥스 회귀 학습 아이디어

# * 샘플 $\mathbf x$가 주어졌을 때 각각의 분류 클래스 $k$ 에 대한 점수 $s_k(\mathbf x)$ 계산.
#     즉, `k*(n+1)` 개의 파라미터를 학습시켜야 함.
# 
# $$
# s_k(\mathbf x) = \theta_0^{(k)} + \theta_1^{(k)}\, x_1 + \cdots + \theta_n^{(k)}\, x_n
# $$    

# * __소프트맥스 함수__를 이용하여 각 클래스 $k$에 속할 확률 $\hat p_k$ 계산
# 
# $$
# \hat p_k = 
# \frac{\exp(s_k(\mathbf x))}{\sum_{j=1}^{K}\exp(s_j(\mathbf x))}
# $$

# * 추정 확률이 가장 높은 클래스 선택
# 
# $$
# \hat y = 
# \mathrm{argmax}_k s_k(\mathbf x)
# $$

# ### 소프트맥스 회귀 비용함수

# * 각 분류 클래스 $k$에 대한 적절한 가중치 벡터 $\theta_k$를 학습해 나가야 함.

# * 비용함수: 크로스 엔트로피 비용 함수 사용
# 
# $$
# J(\Theta) = 
# - \frac{1}{m}\, \sum_{i=1}^{m}\sum_{k=1}^{K} y^{(i)}_k\, \log(\hat{p}_k^{(i)})
# $$

# * 위 비용함수에 대해 경사 하강법 적용

# * $K=2$이면 로지스틱 회귀의 로그 손실 함수와 정확하게 일치.

# * 주어진 샘플의 타깃 클래스를 제대로 예측할 경우 높은 확률값 계산

# * 크로스 엔트로피 개념은 정보 이론에서 유래함. 자세한 설명은 생략.

# ### 다중 클래스 분류 예제

# * 사이킷런의 `LogisticRegression` 예측기 활용
#     * `multi_class=multinomial`로 지정
#     * `solver=lbfgs`: 다중 클래스 분류 사용할 때 반드시 지정

# * 붓꽃 꽃잎의 너비와 길이를 기준으로 품종 분류
#     * 결정경계: 배경색으로 구분
#     * 곡선: Iris-Versicolor 클래스에 속할 확률

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-16.png" width="700"/></div>

# In[ ]:




