#!/usr/bin/env python
# coding: utf-8

# (ch:decisionTrees)=
# # 결정트리

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 결정트리](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_decision_trees.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**
# 
# * 결정트리 훈련과 활용
# * CART 알고리즘
# * 지니 불순도 vs. 엔트로피
# * 결정트리 규제
# * 결정트리 회귀 모델
# * 결정트리 단점

# **목표**
# 
# 분류와 회귀 모두 사용 가능하며
# 랜덤 포레스트의 기본 요소로 사용되는 
# 결정트리에 대해 알아본다. 

# ## 결정트리 훈련과 활용

# ### 결정트리 훈련

# 아래 코드는 붓꽃 데이터셋을 대상으로 사이킷런의 `DecisionTreeClassifier` 모델을 훈련시킨다. 
# 특성은 꽃잎의 길이와 너비만을 사용하여 세 개의 품종으로 분류하는 다중 클래스 모델을 훈련시킨다. 
# `max_depth`는 결정트리의 최대 깊이 지정하는 하이퍼파라미터이며 허용되는 최대 가지치수 수를 지정하며,
# 여기서는 2를 사용한다. 
# 
# ```python
# >>> tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# >>> tree_clf.fit(X, y)
# ```

# :::{admonition} 결정트리 모델과 데이터 전처리
# :class: info
# 
# 결정트리는 특별한 경우가 아니면 데이터 전처리를 거의 요구하지 않지만, 잠시 뒤에 전처리가 필요한 경우를 살펴볼 것이다.
# :::

# ### 결정트리 시각화

# 사이킷런의 `export_graphviz()` 함수를 이용하여 학습된 결정트리를 그래프로 시각화한다. 
# 또한 pdf, png 등 많은 종류의 파일로 변환이 가능하다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-01.png" width="400"/></div>

# 결정트리의 각 노드에 포함된 속성은 다음과 같다.
# 
# * `gini`: 해당 노드의 지니 불순도 측정값.
# * `samples`: 해당 노드에 포함된 샘플 수
# * `value`: 해당 노드에 포함된 샘플들의 실제 클래스별 개수. 타깃 정보 활용.
# * `class`: 각 클래스별 비율을 계산하여 가장 높은 비율에 해당하는 클래스 선정. 
#     동일한 비율이면 낮은 인덱스 선정

# **지니 불순도**
# 
# 각 노드의 지니 불순도는 다음처럼 계산된다. 
# 아래 수식에서 $G_i$ 는 $i$-번째 노드의 지니 불순도를 가리킨다.
# 
# $$G_i = 1 - \sum_{k=1}^{K} (p_{i,k})^2$$
# 
# 단, $p_{i,k}$는 $i$ 번째 노드에 있는 훈련 샘플 중 클래스 $k$에 속한 샘플의 비율이며,
# $K$는 클래스의 총 개수이다.
# 예를 들어, 깊이 2의 왼편 노드 $G_4$ 의 지니 불순도는 0.168로 계산된다.
# 
# $$G_4 = 1 - (0/54)^2 - (49/54)^2 - (5/54)^2 = 0.168$$

# :::{admonition} `graphviz` 패키지
# :class: info
# 
# 아래 결정트리 그래프를 그리려면 다음 두 패키지를 컴퓨터 설치해야 한다.
# 
# - 파이썬 graphviz 패키지: 
# 
#     ```
#     pip install graphviz
#     ```
#     
# - 우분투 graphviz 패키지: 
# 
#     ```
#     sudo apt-get update
#     sudo apt-get -y install graphviz
#     ```
# :::

# ### 클래스 예측

# 트리의 노드<font size='2'>node</font>는 가지 분할이 시작되는 지점이며 세 종류로 나뉜다.
# 
# * 노드<font size='2'>node</font>: 가지 분할이 시작되는 지점이며 부모 노드와 자식노드를 가질 수 있다. 
# * 루트<font size='2'>root</font>: 맨 상단에 위치한 노드이며, 따라서 부모 노드를 갖지 않는다.
# * 리프<font size='2'>leaf</font>: 더 이상의 가지분할이 발생하지 않는 노드이며, 따라서 자식 노드를 갖지 않는다.

# **결정트리 예측**
# 
# 꽃잎 길이와 너비가 각각 5cm, 1.5cm 인 샘플의 클래스는 아래 과정을 거처 
# 주어진 데이터가 속한 리프 노드의 속성을 확인하여 예측한다.
# 
# * 루트에서 시작한다.
# 
# * 분할 1단계: 꽃잎 길이가 2.45cm 보다 크기에 오른편 자식 노드로 이동한다.
# 
# * 분할 2단계: 꽃잎 너비가 1.75cm 이하이기에 왼편 자식 노드로 이동한다.
#     해당 노드가 리프 노드이고 버시컬러의 비율이 가장 높기에 버시컬러 품종으로 예측한다.

# **결정경계**
# 
# `max_depth=3`으로 지정해서 학습된 결정트리의 결정경계를 그래프로 그리면 다음과 같다.
# 
# * 1차 분할 기준: 꽃잎 길이 2.45cm
# * 2차 분할 기준: 꽃잎 너비 1.75cm
# * 3차 분할 기준: 꽃잎 길이 4.85cm와 4.95cm

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-02a.png" width="500"/></div>

# graphviz를 이용한 결과는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-02c.png" width="630"/></div>

# ### 클래스 확률 추정

# 주어진 샘플이 어떤 클래스에 속할 확률은 다음과 같이 계산된다.
# 
# - 먼저, 해당 샘플이 포함되는 리프 노드를 확인한다.
# - 해당 리프 노드에서 각 클래스별 비율을 계산하여 각 클래스에 속할 확률로 사용한다.
# 
# 예를 들어, 꽃잎 길이와 너비가 각각 5cm, 1.5cm인 붓꽃은 위 결정트리에서 
# 깊이 2의 왼편 리프 노드에 포함된다.
# 해당 리프 노드 포함된 샘플들의 클래스별 비율은 다음과 같다.
#     
# $$(0/54, 49/54, 5/54) = (0, 0.907, 0.093) \qquad\qquad (\text{세토사}, \text{버시컬러}, \text{버지니카})$$
#    
# 따라서 버시컬러에 속할 확률이 90.7%로 가장 높다. 
# 참고로 동일한 노드에 속한 샘플에 대한 추정 확률은 언제나 동일하다.
# 
# 사이킷런의 결정트리 모델은 `predict_proba()` 메서드가 지정된 샘플의 클래스별 추정 확률을 계산한다.
# 
# ```python
# >>> tree_clf.predict_proba([[5, 1.5]]).round(3)
# array([[0.   , 0.907, 0.093]])
# ```
# 
# 반면에 `predict()` 메서드는 품종 클래스를 예측하며, 가장 높은 추정 확률을 갖는 품종으로 지정한다.
# 
# ```python
# >>> tree_clf.predict_proba([[5, 1.5]]).round(3)
# array([1])
# ```

# ## CART 훈련 알고리즘

# **CART<font size='2'>Classification and Regression Tree<font> 분류 알고리즘**

# 각 노드에서 아래 비용함수를 최소화 하는 특성 $k$와 해당 특성의 임곗값 $t_k$을 사용한다.
# 
# - $m$, $m_\text{left}$, $m_\text{right}$: 각각 부모와 왼쪽, 오른쪽 자식 노드에 속한 샘플 개수
# - $G_\text{left}$, $G_\text{right}$: 각각 왼쪽, 오른쪽 자식 노드의 지니 불순도
# 
# $$
# J(k, t_k) = \frac{m_\text{left}}{m}\, G_\text{left} + \frac{m_\text{right}}{m}\, G_\text{right}
# $$

# 즉, 지니 불순도가 낮은 두 개의 부분집합으로 분할되도록 학습된다.
# 분할은 `max_depth` 등 규제에 의해 조절되거나 더 이상 불순도를 줄이는 분할이 불가능할 때까지 진행된다.

# :::{admonition} 탐욕 알고리즘
# :class: info
# 
# $J(k, t_k)$ 를 가장 작게 하는 $k$ 와 $t_k$ 를 찾는 알고리즘은 탐욕 알고리즘이다.
# 이유는 대상으로 삼은 노드를 기준으로 지니 불순도가 가장 낮은, 
# 즉 가장 순수한(pure) 두 개의 부분집합으로 분할하지만
# 이후의 분할에 대해서는 생각하지 않기 때문이다.
# 하지만 탐욕적 기법은 일반적으로 적절한 성능의 해를 찾아준다. 
# :::

# ### CART 알고리즘의 시간복잡도

# 최적의 결정트리를 찾는 문제는 $O(\exp(m))$ 복잡도를 가지면,
# 이런 문제는 NP-완전<font size='2'>NP-complete</font>이다.
# 따라서 매우 작은 훈련 세트에 대해서도 제대로 적용하기 어렵다. 

# 반면에 CART 알고리즘의 시간 복잡도는 다음과 같다. 
# $n, m$은 각각 특성 개수와 샘플 개수를 나타낸다. 
# 
# * 각 노드에서 분류하는 데 걸리는 시간: $O(n\cdot m\cdot \log_2(m))$
# * 결정트리를 완성하는 데 걸리는 시간: $O(n\cdot m^2\cdot \log_2(m))$ 

# 또한 학습된 결정트리가 예측에 필요한 시간은 $O(\log_2 m)$ 로 매우 빠르다.
# 이유는 결정트리 모델은 균형 이진탐색트리<font size='2'>balanced binary search tree</font>에 
# 가깝고 따라서 가장 깊은 경로가 $\log_2(m)$ 정도이기 때문이다.
# 실제로 각 노드에서 하나의 특성만 분류기준으로 사용되기에 특성 수와 무관하게 이진트리의 경로를 추적할 수 있다.

# ### 지니 불순도 vs. 엔트로피

# `DecisionTreeClassifier`의 `criterion` 하이퍼파라미터의 값을 `'entropy'` 로 지정하면
# 지니 불순도 대신에 샘플들의 __무질서__ 정도를 측정하는 엔트로피<font size='2'>entropy</font>가 
# 노드를 분할하는 기준으로 사용된다.
# 
# $i$-번째 노드의 엔트로피 $H_i$ 는 다음과 같이 계산된다.
# 
# $$H_i = -\sum_{\substack{k=1\\p_{i,k}\neq 0}}^{K} p_{i,k} \log_2(p_{i,k})$$
#     
# 지니 불순도를 사용할 때와 비교해서 큰 차이가 나지는 않는다. 
# 만약 차이가 난다면 엔트로피 방식이 결정 트리를 보다 좌우 균형이 잡히도록
# 자식 노드로 분할한다. 
# 하지만 기본적으로 별 차이가 없고 지니 불순도 방식이 보다 빠르게 훈련되기에 기본값으로 지정되었다.

# :::{admonition} 엔트로피 방식이 보다 균형 잡힌 이진탐색트리를 만드는 이유
# :class: info
# 
# 특정 $k$ 에 대해 $p_{i,k}$ 가 $0$ 에 매우 가까우면 $-\log(p_{i,k})$ 가 매우 커진다.
# 이는 엔트로피 증가로 이어지기 때문에 결국 비용함수 $J(k, t_k)$ 가 증가한다.
# 따라서 $p_{i, k}$ 가 매우 작게되는 경우는 피하도록 학습하게 되고
# 결국 보다 균형 잡힌 두 개의 부분집합으로 분할하는 방향으로 유도된다.
# :::

# ### 규제 하이퍼파라미터

# **비파라미터 모델 vs. 파라미터 모델**
# 
# 결정트리 모델은 데이터에 대한 어떤 가정도 하지 않는다. 
# 예를 들어, 노드를 분할할 때 어떤 제한도 가해지지 않으며,
# 따라서 학습되어야 하는 파라미터의 개수를 미리 알 수 없다. 
# 이유는 노드를 분할 할 때마다 새로운 파라미터가 학습되기 때문이다. 
# 이런 모델을 **비파라미터 모델**<font size='2'>nonparametric model</font>이라 부른다.
# 비파라미터 모델의 자유도가 제한되지 않기에 과대적합될 가능성이 높다.
# 
# 반면에 선형 모델 등 지금까지 살펴 본 모든 모델은 학습되는 모델에
# 필요한 파라미터 수가 훈련 시작 전에 규정되며, 이런 이유로 과대적합 가능성이 상대적으로 작아진다.
# 이런 모델을 **파라미터 모델**<font size='2'>parametric model</font>이라 한다. 

# **사이킷런  `DecisionTreeClassifier` 규제 하이퍼파라미터**
# 
# 앞서 살펴 본 `max_depth` 이외에 다양한 하이퍼파라미터를 이용해 
# 모델의 과대적합 위험도를 줄이는 규제로 사용한다.
# 가장 많이 사용되는 규제는 다음과 같다. 
# `min_` 을 접두사로 갖는 하이퍼파라미터는 값을 키우는 경우, 
# `max_` 를 접두사로 갖는 하이퍼파라미터는 값을 감소시키는 경우 규제가 강해진다.
# 
# | 하이퍼파라미터 | 기능 |
# |:---|:---|
# | `max_depth` | 결정트리의 높이 제한 |
# | `min_samples_split` | 노드 분할해 필요한 최소 샘플 개수 |
# | `min_samples_leaf` | 리프에 포함된 최소 샘플 개수 |
# | `min_weight_fraction_leaf` | 샘플 가중치 합의 최솟값 | 
# | `max_leaf_nodes` | 최대 리프 개수 |
# | `max_features` | 분할에 사용되는 특성 개수 |

# :::{admonition} 샘플의 가중치
# :class: note
# 
# 샘플의 가중치가 지정되지 않은 경우 모든 샘플의 가중치가 1이라고 가정한다.
# 이런 경우 `min_weight_fraction_leaf`는 `min_samples_leaf`와 동일한 역할을 수행한다.
# :::

# :::{prf:example} `moons` 데이터셋과 결정트리
# :label: exp:moons_decision_tree
# 
# 아래 두 그래프는 moons 데이터셋에 두 개의 결정트리 모델을 적용한 결과를 보여준다. 
# 왼쪽 그래프가 규제 없이 훈련된 결정트리인데 과대적합되었음을 바로 확인할 수 있다.
# 반면에 오른쪽 그래프는 `min_samples_leaf=5` 를 사용하여 모든 리프가 최소 5개 이상의 
# 데이터를 포함하도록 규제하였다.
# 결과적으로 일반화 성능이 보다 좋은 결정트리가 생성되었다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-03.png" width="600"/></div>
# :::

# ## (결정트리) 회귀

# ### 사이킷런의 `DecisionTreeRegressor` 예측기 활용

# * 결정트리 알고리즘 아이디어를 거의 그대로 이용하여 회귀 문제에 적용 가능
# 
#     ```python
#     tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
#     tree_reg.fit(X, y)
#     ```

# * 예제: 잡음이 포함된 2차 함수 형태의 데이터셋
#     * 왼편: `max_depth=2`
#     * 오른편: `max_depth=3`

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-04.png" width="600"/></div>

# * 왼편 그림 설명
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-05.png" width="600"/></div>

# * 각 노드에 포함된 속성
#   * `samples`: 해당 노드에 속한 훈련 샘플 수
#   * `value`: 해당 노드에 속한 훈련 샘플의 평균 타깃값
#   * `mse`: 해당 노드에 속한 훈련 샘플의 평균제곱오차(MSE)
#     * 오차 기준은 `value` 사용.

# ### 회귀용 CART 알고리즘과 비용함수

# * 분류의 경우처럼 탐욕적으로 아래 비용함수를 최소화 하는 특성 $k$와 해당 특성의 임곗값 $t_k$을 결정함:

# $$
# \begin{align*}
# J(k, t_k) &= \frac{m_\text{left}}{m}\, \text{MSE}_\text{left} + \frac{m_\text{right}}{m}\, \text{MSE}_\text{right} \\[2ex]
# \text{MSE}_\text{node} &= \sum_{i\in \text{node}} (\hat y_{node} - y^{(i)})^2\\[1ex]
# \hat y_\text{node} &= \frac{1}{m_\text{node}} \sum_{i\in\text{node}} y^{(i)}
# \end{align*}
# $$

# * $\text{MSE}_\text{left}$($\text{MSE}_\text{right}$):
#     지정된 특성 $k$와 특성 임곗값 $t_k$로 구분된 왼편(오른편) 부분집합의 평균제곱오차
#     * 해당 노드에 속한 샘플들의 평균 타깃값 기준
#     * $m_\text{left}$/$m_\text{right}$: 해당 노드에 속하는 샘플 수
#     * $y^{(i)}$: 샘플 $i$에 대한 실제 타깃

# ### 규제

# * 분류의 경우처럼 규제가 없으면 과대적합 발생할 수 있음.
# 
# * 왼편: 규제가 없는 경우. 과대적합 발생
#     
# * 오른편: `min_samples_leaf=10`

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-06.png" width="600"/></div>

# ## (결정트리) 불안정성

# ### 단점 1: 훈련 세트 회전 민감도

# * 결정트리 알고리즘은 성능이 매우 우수하지만 기본적으로 주어진 훈련 세트에 민감하게 반응함.

# * 결정트리는 항상 축에 수직인 분할을 사용. 따라서 조금만 회전을 가해도 결정 경계가 많이 달라짐

# * 예제: 오른편 그래프: 왼편 그래프를 45도 회전시킨 훈련 세트 학습

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-07.png" width="600"/></div>

# * PCA 기법 등을 사용하여 훈련 샘플 회전시킨 후 학습 가능. (8장 참조)

# ### 단점 2: 훈련 세트 변화 민감도

# * 훈련 데이터의 작은 변화에도 매우 민감함.

# * 예제: 붓꽃 데이터에서 하나의 샘플을 제거한 후 학습시킬 때 매우 다르게 학습할 수 있음.
#     * 왼편 그래프: 모든 샘플 대상 훈련
#     * 오른편 그래프: 가장 넓은 버시컬러 샘플 제거 후 훈련

# <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-08d.png"/>

# <img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch06/homl06-08e.png"/>

# * 많은 트리에서 만든 예측값의 평균을 활용 추천(7장 램덤포레스트 모델 참조)
