#!/usr/bin/env python
# coding: utf-8

# (ch:ensemble)=
# # 앙상블 학습과 랜덤 포레스트

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 앙상블 학습과 랜덤 포레스트](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_ensemble_learning_random_forests.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**

# (1) 편향과 분산의 트레이드오프
# 
# 앙상블 학습의 핵심은 **편향**<font size='2'>bias</font>과 
# **분산**<font size='2'>variance</font>을 줄인 모델을 구현하는 것이다.
# 
# * 편향: 예측값과 정답이 떨어져 있는 정도를 나타낸다.
#     정답에 대한 잘못된 가정으로부터 유발되며
#     편향이 크면 과소적합이 발생한다.
# 
# * 분산: 입력 샘플의 작은 변동에 반응하는 정도를 나타낸다.
#     정답에 대한 너무 복잡한 모델을 설정하는 경우 분산이 커지며,
#     분산이 크면 과대적합이 발생한다.
# 
# 그런데 편향과 분산을 동시에 줄일 수 없다.
# 이유는 편향과 분산은 서로 트레이드오프 관계를 갖기 때문이다. 
# 예를 들어 회귀 모델의 평균 제곱 오차(MSE)는 편향의 제곱과 분산의 합으로 근사되는데,
# 회귀 모델의 복잡도에 따른 편향, 분산, 평균 제곱 오차 사이의 관계를 
# 그래프로 나타내면 보통 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/bagging_boosting02.png" width="600"/></div>
# 
# <[위키백과: 편향-분산 트레이드오프](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)>

# :::{admonition} 평균 제곱 오차, 편향, 분산의 관계
# :class: info
# 
# [Bias, Variance, and MSE of Estimators](http://theanalysisofdata.com/notes/estimators1.pdf) 에서
# 평균 제곱 오차, 분산, 편향 사이의 다음 수학적 관계를 잘 설명한다.
# 
# $$
# \text{평균제곱오차} \approx \text{편향}^2 + \text{분산}
# $$
# :::

# (2) 앙상블 학습
# 
# **앙상블 학습**<font size='2'>ensemble learning</font>은 
# 모델 여러 개를 이용한 훈련과 예측을 진행하는 모델을 구현할 때 사용한다.
# 결과적으로 분산 또는 편향을 줄이기 사용되며 대표적으로 
# **배깅**<font size='2'>bagging</font> 기법과
# **부스팅**<font size='2'>boosting</font> 기법이 
# 주로 사용된다.
# 
# - 배깅 기법: 독립적으로 학습된 예측기 여러 개의 예측값들의 평균값을 예측값으로 
#     사용하여 분산이 줄어든 모델을 구현한다.
# 
# - 부스팅 기법: 예측기 여러 개를 순차적으로 쌓아 올려 예측값의 편향를 줄이는 
#     모델을 구현한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/bagging_boosting01.png" width="500"/></div>

# ## 투표식 분류기

# 동일한 훈련 세트에 대해 여러 종류의 분류 모델을 이용한 앙상블 학습을 진행한 후에 
# 직접 또는 간접 투표를 통해 예측값을 결정한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-01.png" width="500"/></div>

# **직접 투표**
# 
# 앙상블 학습에 사용된 예측기들의 예측값들 중에서 다수결 방식으로 예측하면
# 각각의 예측기보다 좋은 성능의 모델을 얻는다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-02.png" width="500"/></div>

# **간접 투표**
# 
# 앙상블 학습에 사용된 예측기들의 예측한 확률값들의 평균값으로 예측값 결정한다.
# 이를 위해서는 모든 예측기가 `predict_proba()` 메서드처럼 확률을 예측하는 기능을 지원해야 한다.
# 높은 확률에 보다 높은 비중을 두기 때문에 직접 투표 방식보다 성능이 좀 더 좋은 경향이 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-04.png" width="500"/></div>
# 
# <그림출처: [kaggle](https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning)>

# **사이킷런의 투표식 분류기: `VotingClassifier`, `VotingRegressor`**
# 
# * `voting='hard'` 또는 `voting='soft'`: 직접 또는 간접 투표 방식 지정 하이퍼파라미터.
#     기본값은 `'hard'`.
# * 주의: `SVC` 모델 지정할 때 `probability=True` 사용해야 `predict_proba()` 메서드 지원됨.
# 
# ```python
# voting_clf = VotingClassifier(
#     estimators=[
#         ('lr', LogisticRegression(random_state=42)),
#         ('rf', RandomForestClassifier(random_state=42)),
#         ('svc', SVC(random_state=42))
#     ]
# )
# ```

# :::{admonition} 투표식 분류 성능 향상의 확률적 근거
# :class: info
# 
# 이항분포의 누적 분포 함수<font size='2'>cumulative distribution function</font>(cdf)를 
# 이용하여 앙상블 학습의 성능이 향상되는 이유를 설명할 수 있다.
# 
# ```python
# from scipy.stats import binom
# 
# def ensemble_win_proba(n, p):
#     """
#     p: 예측기 하나의 성능.
#     n: 앙상블 크기, 즉 예측기 개수.
#     반환값: 다수결을 따를 때 성공할 확률. 이항 분포의 누적분포함수 활용.
#     """
#     return 1 - binom.cdf(int(n*0.4999), n, p)
# ```
# 
# 적중률 51% 모델 1,000개의 다수결을 따르면 74.7% 정도의 적중률 나옴.
# 
# ```python
# >>> ensemble_win_proba(1000, 0.51)
# 0.7467502275561786
# ```
# 
# 적중률 51% 모델 10,000개의 다수결을 따르면 97.8% 정도의 적중률 나옴.
# 
# ```python
# >>> ensemble_win_proba(10000, 0.51)
# 0.9777976478701533
# ```
# 
# 적중률 80% 모델 10개의 다수결을 따르면 100%에 가까운 성능이 가능함.
# 
# ```python
# >>> ensemble_win_proba(10, 0.8)
# 0.9936306176
# ```
# 
# 위 결과는 앙상블 학습에 포함된 각각의 모델이 서로 독립인 것을 전제로한 결과이다.
# 만약에 훈련에 동일한 데이터를 사용하면 모델 사이의 독립성이 완전히 보장되지 않으며, 
# 경우에 따라 오히려 성능이 하락할 수 있다.
# 모델들의 독립성을 높이기 위해 매우 다른 알고리즘을 사용하는 다른 종류의
# 모델을 사용할 수도 있다.
# :::

# ## 배깅과 페이스팅

# 배깅 기법은 여러 개의 동일 모델을 훈련 세트의 다양한 부분집합을
# 대상으로 학습시키는 방식이다. 
# 부분집합을 임의로 선택할 때의 중복 허용 여부에 따라 앙상블 학습 방식이 달라진다.
# 
# - **배깅**<font size='2'>bagging</font>: 중복 허용 샘플링(부분집합 선택)
# - **페이스팅**<font size='2'>pasting</font>: 중복 미허용 샘플링(부분집합 선택)

# :::{admonition} 배깅과 부트스트랩
# :class: info
# 
# 배깅은 bootstrap aggregation의 줄임말이며,
# 부트스트랩<font size='2'>bootstrap</font>은 전문 통계 용어로 중복허용 리샘플링을 가리킨다.
# :::

# 아래 그림은 배깅 기법으로 하나의 훈련셋으로 네 개의 동일 예측기를 사용하는 것을 보여준다.
# 각 예측기의 훈련셋이 다르게, 하지만 중복을 허용하는 방식으로 지정되는 것을 그림에서도 확인할 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-05.png" width="500"/></div>

# **예측값**
# 
# 배깅 또는 페이스팅 모델의 예측값은 분류 모델은 예측값의 최빈값을 선택하며,
# 회귀 모델은 예측값들의 평균값을 사용한다.

# **병렬 훈련 및 예측**
# 
# 배깅/페이스팅 모델의 훈련과 예측은 다른 CPU 또는 심지어 다른 컴퓨터 서버를 이용하여 각 모델을 훈련 또는 예측을 하게 만든 후 병합하여 하나의 예측값을 생성하도록 할 수 있다.

# **편향과 분산**
# 
# 개별 예측기의 경우에 비해 앙상블 모델의 편향은 조금 커지거나 거의 비슷하지만 분산은 줄어든다.
# 이유는 배깅이 표본 샘플링의 다양성을 보다 많이 추가하기 때문이다.
# 배깅 방식이 페이스팅 방식보다 과대적합의 위험성일 줄어주기에 기본으로 사용된다.
# 보다 자세한 설명은 
# [Single estimator versus bagging: bias-variance decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py) 을 참고한다.

# ### 사이킷런의 배깅과 페이스팅

# 사이킷런은 `BaggingClassifier` 분류 모델과 `BaggingRegressor` 회귀 모델을 지원하며
# 사용법은 다음과 같다.
# 
# - `n_estimators=500` 개의 `DecisionTreeClassifier` 모델을 이용항 앙상블 학습.
# - `max_samples=100` 개의 훈련 샘플 사용.
# - 배깅 방식. 페이스팅 방식을 사용하려면 `bootstrap=False` 로 지정.
# - `n_jobs=-1` 하이퍼파라미터를 이용하여 사용할 CPU 수 지정. 기본값 -1은 전부 사용을 의미함.
# - 기본적으로 간전 투표 방식 사용. 하지만 기본 예측기가 `predict_proba()` 메서드를 지원하지 않으면
#     직접 투표 방식 사용. 결정트리는 `predict_proba()` 메서드를 지원하기에 간접 투표 방식을 사용함.

# ```python
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                             max_samples=100, random_state=42)
# ```

# 아래 두 이미지는 한 개의 결정트리 모델과 500개의 결정트리 모델의 
# 반달(moons) 데이터셋에 대한 훈련 결과의 차이를 명확하게 보여준다.
# 배깅을 사용한 오른쪽 결정트리 모델의 일반화 성능이 훨씬 좋음을 알 수 있다.
# 왼쪽 하나의 결정트리 모델과 비교해서 편향(오류 숫자)은 좀 더 커졌지만
# 분산(결정 경계의 불규칙성)은 훨씬 덜하다.
# 하지만 편향이 커졌다는 의미는 각 결정트리들 사이의 연관성이 약해졌음을 의미한다.
# 즉, 각 결정트리의 독립성이 커졌고, 따라서 학습 모델의 일반화 성능이 좋아졌다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-06.png" width="600"/></div>

# ### oob 평가

# 배깅 기법을 적용하면 모델 훈련에 선택되지 않은 훈련 샘플이 평균적으로 전체 훈련셋의 37% 정도를 차지한다.
# 이런 샘플을 oob(out-of-bag) 샘플이라 부른다.
# oob 평가는 각 샘플에 대해 해당 샘플을 훈련에 사용하지 않은 예측기들로 이루어진 앙상블 모델의 예측값을 이용하여
# 전체 앙상블 모델의 성능을 검증하는 것이다.

# `BaggingClassifier` 의 경우 `oob_score=True` 하이퍼파라미터를 사용하면
# oob 평가를 자동으로 실행한다. 
# 평가 결과는 `oob_score_` 속성에 저정되며, 테스트 성능과 비슷하게 나온다.

# ```python
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                             oob_score=True, random_state=42)
# ```

# 각 샘플에 대한 oob 예측값, 즉 해당 샘플을 훈련에 사용하지 않은 예측기들로 이루어진 앙상블 모델의 예측값은 
# `oob_decision_function_()` 메서드가 계산한다. 
# 예를 들어, 처음 세 개 훈련 샘플에 대한 oob 예측값은 다음과 같다. 
# 결정트리 모델이 `predict_proba()` 메서드를 지원하기에 양성, 음성 여부를 확률로 계산한다.

# ```python
# >>> bag_clf.oob_decision_function_[:3]
# array([[0.32352941, 0.67647059],
#        [0.3375    , 0.6625    ],
#        [1.        , 0.        ]])
# ```

# ## 랜덤 패치와 랜덤 서브스페이스

# 이미지 데이터의 경우처럼 특성 수가 매우 많은 경우 특성에 대해 중복선택 옵션을 지정할 수 있다.
# 이를 통해 더 다양한 예측기를 만들며, 편향이 커지지만 분산은 낮아진다.

# * `max_features` 하이퍼파라미터: 
#     학습에 사용할 특성 수 지정. 기본값은 1.0, 즉 전체 특성 모두 사용.
#     정수를 지정하면 지정된 수 만큼의 특성 사용.
#     0과 1 사이의 부동소수점이면 지정된 비율 만큼의 특성 사용.
# 
# * `bootstrap_features` 하이퍼파라미터: 
#     학습에 사용할 특성을 선택할 때 중복 허용 여부 지정. 
#     기본값은 False. 즉, 중복 허용하지 않음.

# **랜덤 패치 기법**
# 
# 훈련 샘플과 훈련 특성 모두를 대상으로 중복을 허용하며 임의의 샘플 수와 임의의 특성 수만큼을 샘플링해서 학습하는 기법이다.

# **랜덤 서브스페이스 기법**
# 
# 전체 훈련 세트를 학습 대상으로 삼지만 훈련 특성은 임의의 특성 수만큼 샘플링해서 학습하는 기법이다.
# 
# - 샘플에 대해: `bootstrap=False`이고 `max_samples=1.0`
# - 특성에 대해: `bootstrap_features=True` 또는 `max_features` 는 1.0 보다 작게.

# ## 랜덤 포레스트

# **랜덤 포레스트**<font size='2'>random forest</font>는
# 배깅 기법을 결정트리의 앙상블에 특화시킨 모델이다.
# 배깅 기법 대신에 페이스팅 기법을 옵션으로 사용할 수도 있으며,
# `RandomForestClassifier` 는 분류 용도로, ` RandomForestRegressor` 는 회귀 용도로 사용한다.
# 
# 아래 두 모델은 기본적으로 동일하며, 사용된 하이퍼파라미터는 다음과 같다.
# 
# - `n_estimators=500`: 500 개의 결정트리 사용
# - `max_leaf_nodes=16`: 리프 노드 최대 16개
# - `n_jobs=-1`: 모든 CPU 사용

# `RandomForestClassifier` 모델

# ```python
# rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
#                                  n_jobs=-1, random_state=42)
# ```

# `BaggingClassifier` 모델

# ```python
# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
#     n_estimators=500, n_jobs=-1, random_state=42)
# ```

# 배깅 모델에 사용된 `DecisionTreeClassifier`의 `max_features="sqrt"` 하이퍼파라미터 인자는 
# 노드 분할에 사용되는 특성의 수를 전체 특성 개수 $n$의 제곱근 값인 $\sqrt{n}$ 으로 제한한다는 의미다.
# 더 나아가 특성 선택이 무작위로 이루어진다.
# 이를 통해 보다 다양한 결정트리를 사용하게 되며, 결과적으로 편향은 좀 더 높지만 보다 좋은 성능의 
# 앙상블 모델이 학습된다.

# `RandomForestClassifier` 모델의 하아퍼파라미터는 `BaggingClassifier`와 `DecisionTreeClassifier`의 옵션을 거의 모두 동일하게 사용한다.

# ### 엑스트라 트리

# 랜덤 포레스트는 $\sqrt{n}$ 개의 특성을 무작위로 선택하지만 선택된 특성의 임곗값은 모든 특성값에 
# 대해 확인한다.
# 그런데 `DecisionTreeClassifier` 모델의 `splitter="random"` 하이퍼파라미터 인자를 사용하면 
# 임곗값도 무작위로 몇 개 선택해서 그중에 최선의 임곗값을 찾는데,
# 그런 결정트리로 구성된 앙상블 학습 모델을 
# **엑스트라 트리**<font size='2'>Extra-Tree</font>라 한다. 
# 참고로 엑스트라 트리는 **Extremely Randomized Tree** 의 줄임말이다.

# 엑스트라 트리는 일반적인 램덤포레스트보다 속도가 훨씬 빠르고,
# 보다 높은 편향을 갖지만 분산은 상대적으로 낮다.
# 
# 아래 코드는 사이킷런의 엑스트라 모델을 선언한다. 
# 하이퍼파라미터는 `bootstrap=False` 를  사용하는 것 이외에는 랜덤포레스트의 경우와 하나만 빼고 동일하다.
# `bootstrap=False` 를 사용하는 이유는 특성과 임곗값을 무작위로 선택하기에 각
# 결정트리의 훈련에 사용될 훈련 샘플들까지 중복을 허용해서 모델의 다양성을 굳이
# 보다 더 키울 필요는 없는 것으로 이해된다.

# ```python
# extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, 
#                                  n_jobs=-1, random_state=42)
# ```

# 랜덤 포레스트와 엑스트르 트리 두 모델의 성능은 기본적으로 비슷한 것으로 알려졌다.

# ### 특성 중요도

# 어떤 특성의 중요도는 해당 특성을 사용한 마디가 평균적으로 불순도를 얼마나 감소시키는지를 측정한 값이다.
# 즉, 불순도를 많이 줄이면 그만큼 중요도가 커진다. 

# `RandomForestClassifier` 모델은 훈련할 때마다 자동으로 모든 특성에 대해 
# 상대적 특성 중요도를 계산하여 `feature_importances_` 속성에 저장한다.
# 즉, 모든 특성 중요도의 합은 1이다.
# 이렇듯 랜덤 포레스트 모델을 이용하여 특성의 상대적 중요도를 파악한 다음에 보다 
# 중요한 특성을 선택해서 활용할 수 있다.

# **예제: 붓꽃 데이터셋**

# 붓꽃 데이터셋의 경우 특성별 상대적 중요도는 다음과 같이 꽃잎의 길이와 너비가 매우 중요하며,
# 꽃받침의 길이와 너비 정보는 상대적으로 훨씬 덜 중요하다.
# 지금까지 붓꽃 데이터셋을 사용할 때 꽃잎의 길이와 너비 두 개의 특성만을 사용한 이유가 여기에 있다.

# | 특성 | 상대적 중요도 |
# | :--- | ---: |
# | 꽃받침 길이 | 0.11 |
# | 곷받침 너비 | 0.02 |
# | 꽃잎 길이 | 0.44 |
# | 곷잎 너비 | 0.42 |

# **예제: MNIST**

# MNIST 데이터셋의 경우 특성으로 사용된 모든 픽셀의 중요도를 그래프로 그리면 다음과 같다.
# 숫자가 일반적으로 중앙에 위치하였기에 중앙에 위치한 픽셀의 중요도가 보다 높게 나온다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-07.png" width="400"/></div>

# ## 부스팅

# 약한 성능의 예측기 여러 개를 이용하여 보다 강한 성능의 예측기를 학습시기는 기법이
# **부스팅**<font size='2'>boosting</font>이다.
# 여러 개의 예측기를 선형적으로 훈련시키면서 약점을 보완해 나가는 알고리즘이 일반적으로 
# 사용되며 대표적으로 다음 두 기법이 사용된다.
# 
# - 에이다부스트<font size='2'>AdaBoost</font>
# - 그레이디언트 부스팅<font size='2'>Gradient Boosting</font>
# 
# 두 기법은 순차적으로 이전 예측기의 결과를 바탕으로 예측 성능을 조금씩 높혀 간다.
# 예측 모델의 편향을 줄여나간다.
# 하지만 순차적으로 학습하기에 배깅/페이스팅 방식과는 달리 훈련을 동시에 진행할 수 없다.
# 따라서 훈련 시간이 훨씬 오래 걸릴 수 있고 따라서 훈련셋 또는 특성 수가 너무 커지면
# 적용이 어려울 수 있다.

# ### 에이다부스트<font size='2'>AdaBoost</font>

# 하나의 모델을 훈련 시킨 후 틀리계 예측된 샘플을 보다 강조하면서 해당 모델을 다시 훈련시킨다.
# 아래 그림은 모델이 제대로 학습하지 못한, 즉 과소적합했던 샘플에 대한 가중치를 
# 키우는 방식으로 모델을 다시 훈련시키는 과정을 반복하는 것을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-08.png" width="500"/></div>

# **샘플 가중치**
# 
# 훈련중에 특정 샘플을 보다 강조하도록 유도하는 값을 샘플 가중치라 하며,
# 모든 훈련 샘플에 대해 지정할 수 있다.
# 지정하지 않으며 모든 훈련 샘플에 대한 가중치는 동일한 값으로 처리된다.
# 
# 사이킷런 모델의 `fit()` 메서드는 `sample_weight` 옵션인자를 이용하여
# 각 훈련 샘플에 대한 가중치를 지정할 수 있다.
# 
# 참고: [SVC의 `fit()` 메서드 정의](https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/svm/_base.py#L119)

# **에이다부스트 알고리즘 작동 과정**
# 
# * moons 데이터셋에 rbf 커널을 사용하는 SVC 모델을 5번 연속 새로 생성하는 
#     방식으로 학습한 결과를 보여줌.
# 
# * 새로운 예측기의 `fit()` 메서드는 이전 예측기의 경우와 다른 `sample_weight` 옵션값을 사용함.
# 
# * 새로운 예측기는 이전의 예측기의 예측값이 틀린 샘플을 보다 강조하도록 유도됨.
# 
# * 왼편과 오른편은 학습률만 다름.
#     * __주의사항:__ `learnign_rate`는 기존에 설명한 학습률과 다른 의미이며, 각 예측기의 기여도 조절에 사용됨.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-09.png" width="600"/></div>

# **사이키런의 에이다부스트**

# * 분류 모델: `AdaBoostClassifier` 
# * 회귀 모델: `AdaBoostRegressor`

# **예제: 에이다부스트 + 결정트리**

# * `AdaBoostClassifier` 의 기본 모델임.

# ```python
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
#                              n_estimators=200, algorithm="SAMME.R", 
#                              learning_rate=0.5, random_state=42)
# ```

# * 훈련 세트: moons 데이터셋
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-10.png" width="400"/></div>

# ### 그레이디언트 부스팅

# * 이전 학습기에 의한 오차를 보정하도록 새로운 예측기를 순차적으로 추가하는 아이디어는 에이다부스트와 동일

# * 샘플의 가중치를 수정하는 대신 이전 예측기가 만든 __잔차__(residual error)에 대해 새로운 예측기를 학습시킴

# * 잔차(residual error): 예측값과 실제값 사이의 오차

# **사이킷런 그레이디언트 부스팅 모델**

# * 분류 모델: `GradientBoostingClassifier`
#     * `RandomForestClassifier`와 비슷한 하이퍼파라미터를 제공

# * 회귀 모델: `GradientBoostingRegressor`
#     * `RandomForestRegressor`와 비슷한 하이퍼파라미터를 제공

# **그레이디언트 부스티드 회귀 나무(GBRT) 예제: 그레이디언트 부스팅 (회귀)+ 결정트리**
# 
# * 2차 다항식 데이터셋에 결정트리 3개를 적용한 효과와 동일하게 작동

# ```python
# gbrt = GradientBoostingRegressor(max_depth=2, 
#                                  n_estimators=3, 
#                                  learning_rate=1.0)
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-11.png" width="500"/></div>

# **`learning_rate`(학습률)**

# * `learnign_rate`는 기존에 설명한 학습률과 다른 의미의 학습률. 
#     * 각 결정트리의 기여도 조절에 사용

# * 수축(shrinkage) 규제: 학습률을 낮게 정하면 많은 수의 결정트리 필요하지만 성능 좋아짐.

# * 이전 결정트리에서 학습된 값을 전달할 때 사용되는 비율
#     * 1.0이면 그대로 전달
#     * 1.0보다 작으면 해당 비율 만큼 조금만 전달

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-12a.png" width="500"/></div>

# **최적의 결정트리 수 확인법**

# * 조기종료 기법 활용

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-12.png" width="500"/></div>

# **확률적 그레이디언트 부스팅**

# * 각 결정트리가 훈련에 사용할 훈련 샘플의 비율을 지정하여 학습: `subsample=0.25` 등 비율 지정

# * 훈련 속도 빨라짐.

# * 편향 높아지지만, 분산 낮아짐.

# **XGBoost**

# * Extreme Gradient Boosting의 줄임말.

# * 빠른 속도, 확장성, 이식성 뛰어남.

# * 조기종료 등 다양한 기능 제공.
# 
#     ```python
#     import xgboost
#     xgb_reg = xgboost.XGBRegressor(random_state=42)
#     xgb_reg.fit(X_train, y_train,
#                 eval_set=[(X_val, y_val)], 
#                 early_stopping_rounds=2)
#     ```

# ## 스태킹

# * 배깅방식의 응용으로 볼 수 있는 기법

# * 다수결을 이용하는 대신 여러 예측값을 훈련 데이터로 활용하는 예측기를 훈련시키는 기법

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-13.png" width="400"/></div>

# ### 스태킹 모델 훈련법

# - 책에서는 스태킹 기법을 소개만하고 코드 구현은 연습문제 9번에서 설명한다.  

# - 여기서는 사이킷런 0.22부터 지원하는 스태킹 모델을 활용하여 코드구현을 설명한다.

# - 참조: [Stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)

# **1층 훈련**

# * 먼저 훈련 세트를 훈련세트1과 훈련세트2로 이등분한다.

# * 하나의 훈련세트1의 전체 샘플을 이용하여 주어진 예측기들을 각자 독립적으로 훈련시킨다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-14.png" width="400"/></div>

# **2층 훈련**

# * 훈련세트2의 모든 샘플에 대해 훈련된 예측기별로 예측값을 생성한다.

# * 예측값들로 이루어진 훈련세트를 이용하여 믹서기 모델(블렌더)을 훈련시킨다.
#     - 2층 훈련에 사용되는 샘플의 차원은 1층에 사용되는 예측기 개수이다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-15.png" width="400"/></div>

# **사이킷런의 `StackingRegressor` 모델 활용법**

# ```python
# estimators = [('ridge', RidgeCV()),
#               ('lasso', LassoCV(random_state=42)),
#               ('knr', KNeighborsRegressor(n_neighbors=20,
#                                           metric='euclidean'))]
# 
# final_estimator = GradientBoostingRegressor(n_estimators=25, subsample=0.5, 
#                         min_samples_leaf=25, max_features=1, random_state=42)
# 
# reg = StackingRegressor(estimators=estimators,
#                         final_estimator=final_estimator)
# ```

# **사이킷런의 `StackingClassifier` 모델 활용법**

# ```python
# estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#               ('svr', make_pipeline(StandardScaler(),
#                                     LinearSVC(random_state=42)))]
# 
# clf = StackingClassifier(estimators=estimators, 
#                          final_estimator=LogisticRegression())
# ```

# **스태킹 모델의 예측값**

# * 레이어를 차례대로 실행해서 믹서기(블렌더)가 예측한 값을 예측값으로 지정한다.

# * 훈련된 스태킹 모델의 편향과 분산이 훈련에 사용된 모델들에 비해 모두 감소한다.

# ### 다층 스태킹

# * 2층에서 여러 개의 믹서기(블렌더)를 사용하고, 
#     그위 3층에 새로운 믹서기를 추가하는 방식으로 다층 스태킹을 훈련시킬 수 있다.

# * 다층 스태킹의 훈련 방식은 2층 스태킹의 훈련 방식을 반복하면 된다.

# **예제: 3층 스태킹 모델 훈련과정**

# - 훈련세트를 세 개의 부분 훈련세트로 나눈다.
# - 훈련세트1은 1층 예측기 훈련에 사용한다.
# - 훈련세트2은 2층 믹서기 훈련에 사용한다.
# - 훈련세트3은 3층 믹서기 훈련에 사용한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-17.png" width="400"/></div>

# ## 연습문제

# 1. [AugoGluon](https://auto.gluon.ai/stable/index.html) 활용하기
