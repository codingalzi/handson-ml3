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
# **분산**<font size='2'>variance</font>을 최소화한 모델을 구현하는 것이다.
# 
# * 편향: 예측값과 정답이 떨어져 있는 정도를 나타낸다.
#     정답에 대한 잘못된 가정으로부터 유발되며
#     편향이 크면 과소적합이 발생한다.
# 
# * 분산: 입력 샘플의 작은 변동에 반응하는 정도를 나타낸다.
#     일반적으로 모델을 복잡하게 설정할 수록 분산이 커지며,
#     따라서 과대적합이 발생한다.
# 
# 그런데 편향과 분산을 동시에 줄일 수 없다.
# 이유는 편향과 분산은 서로 트레이드오프 관계를 갖기 때문이다. 
# 예를 들어 회귀 모델의 평균 제곱 오차(MSE)는 
# 편향을 제곱한 값과 분산의 합으로 근사되는데,
# 회귀 모델의 복잡도에 따른 편향, 분산, 평균 제곱 오차 사이의 관계를 
# 그래프로 나타내면 보통 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/bagging_boosting02.png" width="500"/></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff">위키백과: 편향-분산 트레이드오프</a>&gt;</div></p>

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
# 여러 개 모델을 훈련시킨 결과를 이용하여 기법이며,
# 대표적으로 
# **배깅**<font size='2'>bagging</font> 기법과
# **부스팅**<font size='2'>boosting</font> 기법이 있다.
# 
# - 배깅 기법: 여러 개의 예측기를 (가능한한) 독립적으로 학습시킨 후
#     모든 예측기들의 예측값들의 평균값을 최종 모델의 예측값으로 사용한다.
#     분산이 보다 줄어든 모델을 구현한다.
# 
# - 부스팅 기법: 여러 개의 예측기를 순차적으로 훈련시킨 결과를 예측값으로 사용한다.
#     보다 적은 편향를 갖는 모델을 구현한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/bagging_boosting01.png" width="550"/></div>

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
# 이를 위해서는 사용되는 모든 분류기가 `predict_proba()` 메서드처럼 확률을 예측하는 기능을 지원해야 한다.
# 높은 확률의 비중을 크게 잡기 때문에 직접 투표 방식보다 일반적으로 성능이 좀 더 좋다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-04.png" width="500"/></div>
# 
# <p><div style="text-align: center">&lt;그림 출처: <a href="https://www.kaggle.com/fengdanye/machine-learning-6-basic-ensemble-learning">Machine Learning 6 Basic Ensemble Learning</a>&gt;</div></p>

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
# 누적 분포 함수는 주어진 확률 변수가 특정 값보다 같거나 작은 값을 가질 확률을 계산한다.
# 
# 아래 코드에서 `binom.cdf(int(n*0.4999), n, p)`는 `p`의 확률로
# 문제를 맞추는 예측기 `n` 개를 이용하여 다수결에 따라 예측을 했을 때
# 성공할 확률을 계산한다.
# 
# ```python
# from scipy.stats import binom
# 
# def ensemble_win_proba(n, p):
#     """
#     p: 예측기 하나의 성능.
#     n: 앙상블 크기, 즉 예측기 개수.
#     반환값: 다수결을 따를 때 성공할 확률. 이항 분포의 누적 분포수 활용.
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

# 배깅 기법은 하나의 훈련 세트의 다양한 부분집합을 이용하여 
# 동일한 모델 여러 개를 학습시키는 방식이다. 
# 부분집합을 임의로 선택할 때의 중복 허용 여부에 따라 앙상블 학습 방식이 달라진다.
# 
# - **배깅**<font size='2'>bagging</font>: 중복을 허용하며 부분집합 샘플링(부분집합 선택)
# - **페이스팅**<font size='2'>pasting</font>: 중복을 허용하지 않으면서 부분집합 샘플링(부분집합 선택)

# :::{admonition} 배깅과 부트스트랩
# :class: info
# 
# 배깅은 bootstrap aggregation의 줄임말이며,
# 부트스트랩<font size='2'>bootstrap</font>은 전문 통계 용어로 중복 허용 리샘플링을 가리킨다.
# :::

# 아래 그림은 하나의 훈련셋으로 동일한 예측기 네 개를 훈련시키는 내용을 보여준다.
# 훈련셋으로 사용되는 각각의 부분집합이 중복을 허용하는 방식, 즉 
# 배깅 방식으로 지정되는 것을 그림이 잘 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-05.png" width="500"/></div>

# **예측값**
# 
# 배깅 또는 페이스팅 방식으로 훈련된 모델의 예측값은,
# 분류기인 경우엔 최빈 예측값<font size='2'>mode</font>을,
# 회귀인 경우엔 예측값들의 평균값<font size='2'>mean</font>을 사용한다.

# **병렬 훈련 및 예측**
# 
# 배깅/페이스팅 기법은 각 모델의 훈련과 예측을 병렬로 다룰 수 있다.
# 즉, 다른 CPU 또는 심지어 다른 컴퓨터 서버를 이용하여 각 모델을 훈련 또는 예측을 하게 만든 후 
# 그 결과를 병합하여 하나의 예측값을 생성할 수 있다.

# **편향과 분산**
# 
# 개별 예측기의 경우에 비해 배깅 방식으로 학습된 앙상블 모델의 편향은 비슷하거나 조금 커지는
# 반면에 분산은 줄어든다.
# 분산이 줄어드는 이유는 배깅 방식이 표본 샘플링의 다양성을 키우기 때문이다.
# 또한 배깅 방식이 페이스팅 방식보다 과대적합의 위험성을 잘 줄어주며,
# 따라서 보다 선호된다.
# 보다 자세한 설명은 
# [Single estimator versus bagging: bias-variance decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py) 을 참고한다.

# ### 사이킷런의 배깅과 페이스팅

# 사이킷런은 분류 모델인 `BaggingClassifier`와 회귀 모델인 `BaggingRegressor`을 지원한다.
# 아래 코드에서 사용된 분류 모델의 하이퍼파라미터는 다음과 같다.
# 
# - `n_estimators=500`: 500 개의 `DecisionTreeClassifier` 모델을 이용항 앙상블 학습.
# - `max_samples=100`: 각각의 모델을 100 개의 훈련 샘플을 이용하여 훈련.
# 
# 이외에 아래 옵션이 기본으로 사용된다.
# 
# - `n_jobs=None`:  하이퍼파라미터를 이용하여 사용할 CPU 수 지정. 
#     `None`은 1을 의미함. -1로 지정하면 모든 CPU를 사용함.
# - `bootstrap=True`: 배깅 방식. 페이스팅 방식을 사용하려면 `bootstrap=False` 로 지정.
# - `oob_score=False`: oob 평가 진행 여부. `bootstrap=True`인 경우에만 설정 가능.
# 
# ```python
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                             max_samples=100, random_state=42)
# ```

# 모델의 예측값은 기본적으로 간전 투표 방식을 사용한다.
# 하지만 기본 예측기가 `predict_proba()` 메서드를 지원하지 않으면
# 직접 투표 방식을 사용한다. 
# 위 코드에서는 결정트리가 `predict_proba()` 메서드를 지원하기에 간접 투표 방식을 사용하게 된다.

# 아래 두 그림은 한 개의 결정트리 모델의 훈련 결과와 500개의 결정트리 모델을 
# 배깅 기법으로 훈련시킨 결과의 차이를 보여준다.
# 훈련셋으로 초승달 데이터셋<font size='2'>moons dataset</font>이 사용되었다.
# 
# 왼쪽 그림은 규제를 전혀 사용하지 않아 훈련셋에 과대적합된 결정트리 모델을 보여준다.
# 반면에 오른쪽 그림은 규제를 전혀 사용하지 않은 결정트리를 이용했음에도 불구하고
# 배깅 기법을 적용하면 보다 높은 일반화 성능의 모델을 얻을 수 있음을 잘 보여준다.
# 하나의 결정트리 모델과 비교해서 편향(오류 숫자)은 좀 더 커졌지만
# 분산(결정 경계의 불규칙성)은 훨씬 덜하다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-06.png" width="600"/></div>

# ### oob 평가

# 배깅 기법을 적용하면 하나의 모델 훈련에 선택되지 않은 훈련 샘플이 평균적으로 
# 전체 훈련셋의 37% 정도를 차지한다.
# 이런 샘플을 oob(out-of-bag) 샘플이라 부른다.
# oob 평가는 각각의 샘플에 대해 해당 샘플을 훈련에 사용하지 않은 
# 모델들의 예측값을 이용하여 앙상블 학습 모델의 검증하는 기법이다.
# 
# 6 개의 훈련 샘플로 구성된 훈련셋 대해 5개의 결정트리 모델을 배깅기법으로
# 적용할 때 예를 들어 아래 표와 같은 경우가 발생할 수 있다.
# 표에 사용된 정수는 중복으로 뽑힌 횟수를 가리키며,
# 각 샘플은 위치 인덱스로 구분한다.
# 
# 
# | | 훈련 샘플(총 6개) | OOB 평가 샘플 |
# | :---: | :---: | :---: |
# | 결정트리1 | 1, 1, 0, 2, 1, 1 | 2번 |
# | 결정트리2 | 3, 0, 1, 0, 2, 0 | 1번, 3번, 5번 |
# | 결정트리3 | 0, 1, 3, 1, 0, 1 | 0번, 4번 |
# | 결정트리4 | 0, 0, 2, 0, 2, 2 | 0번, 1번, 3번 |
# | 결정트리5 | 2, 0, 0, 1, 3, 0 | 1번, 2번, 5번 |
# 
# 그러면 각 샘플을 이용한 앙상블 학습에 사용된 모델은 다음과 같다.
# 
# - 0번 샘플: 결정트리3, 결정트리4
# - 1번 샘플: 결정트리2, 결정트리4, 결정트리5
# - 2번 샘플: 결정트리1, 결정트리5
# - 3번 샘플: 결정트리2, 결정트리4
# - 4번 샘플: 결정트리3
# - 5번 샘플: 결정트리2, 결정트리5

# **예제: `BaggingClassifier`를 이용한 oob 평가**

# `BaggingClassifier` 의 경우 `oob_score=True` 하이퍼파라미터를 사용하면
# oob 평가를 자동으로 실행한다. 
# 평가 결과는 `oob_score_` 속성에 저정되며, 테스트 성능과 비슷하게 나온다.

# ```python
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                             oob_score=True, random_state=42)
# ```

# 각 샘플에 대한 oob 예측값, 즉 해당 샘플을 훈련에 사용하지 않은 예측기들로 이루어진 앙상블 모델의 예측값은 
# `oob_decision_function_` 속성에 저장된다. 
# 예를 들어, 훈련셋의 맨 앞에 위치한 3 개의 훈련 샘플에 대한 oob 예측값은 다음과 같다. 
# 결정트리 모델이 `predict_proba()` 메서드를 지원하기에 양성, 음성 여부를 확률로 계산한다.

# ```python
# >>> bag_clf.oob_decision_function_[:3]
# array([[0.32352941, 0.67647059],
#        [0.3375    , 0.6625    ],
#        [1.        , 0.        ]])
# ```

# 예를 들어, 첫째 훈련 샘플에 대한 oob 평가는 양성일 확률을 67.6% 정도로 평가한다.

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

# :::{prf:example} 붓꽃 데이터셋
# :label: exp-minist-feature-importance
# 
# 붓꽃 데이터셋의 경우 특성별 상대적 중요도는 다음과 같이 꽃잎의 길이와 너비가 매우 중요하며,
# 꽃받침의 길이와 너비 정보는 상대적으로 훨씬 덜 중요하다.
# 지금까지 붓꽃 데이터셋을 사용할 때 꽃잎의 길이와 너비 두 개의 특성만을 사용한 이유가 여기에 있다.
# 
# | 특성 | 상대적 중요도 |
# | :--- | ---: |
# | 꽃받침 길이 | 0.11 |
# | 곷받침 너비 | 0.02 |
# | 꽃잎 길이 | 0.44 |
# | 곷잎 너비 | 0.42 |
# :::

# :::{prf:example} MNIST
# :label: exp-MNIST-feature-importance
# 
# MNIST 데이터셋의 경우 특성으로 사용된 모든 픽셀의 중요도를 그래프로 그리면 다음과 같다.
# 숫자가 일반적으로 중앙에 위치하였기에 중앙에 위치한 픽셀의 중요도가 보다 높게 나온다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-07.png" width="400"/></div>
# :::

# ## 부스팅

# 약한 성능의 예측기 여러 개를 이용하여 보다 강한 성능의 예측기를 학습시기는 기법이
# **부스팅**<font size='2'>boosting</font>이다.
# 일반적으로 여러 개의 예측기를 선형적으로 훈련시키면서 약점을 보완해 나가는 알고리즘을 사용하며
# 대표적으로 다음 두 기법이 사용된다.
# 
# - 에이다부스트<font size='2'>AdaBoost</font>
# - 그레이디언트 부스팅<font size='2'>Gradient Boosting</font>
# 
# 두 기법은 순차적으로 이전 예측기의 결과를 바탕으로 예측 성능을 조금씩 높혀 간다.
# 즉, 예측 모델의 편향을 줄여나간다.
# 하지만 순차적으로 학습하기에 배깅/페이스팅 방식과는 달리 훈련을 동시에 진행할 수 없다는 단점을 갖는다.
# 훈련 시간이 훨씬 오래 걸릴 수 있고 따라서 훈련셋 또는 특성 수가 너무 커지면
# 적용이 어려울 수 있다.

# ### 에이다부스트

# **에이다부스트**<font size='2'>AdaBoost</font> 기법은 
# 하나의 모델을 훈련 시킨 후 **잘못 예측된 샘플을 보다 강조**하면서 해당 모델을 다시 훈련시킨다.
# 아래 그림은 모델이 제대로 학습하지 못한, 즉 과소적합했던 **샘플에 대한 가중치**를 
# 키우는 방식으로 모델을 다시 훈련시키는 과정을 반복하는 것을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-08.png" width="500"/></div>

# **샘플 가중치**
# 
# 훈련중에 특정 샘플을 보다 강조하도록 유도하는 값을 **샘플 가중치**라 하며,
# 모든 훈련 샘플에 대해 지정할 수 있다.
# 지정하지 않으며 모든 훈련 샘플에 대한 가중치는 동일한 값으로 처리된다.
# 특정 훈련 샘플에 대한 샘플 가중치를 크게 주면 해당 샘플이 그만큼 더 훈련셋에 많이 포함된 것처럼
# 처리되어, 해당 샘플의 중요도를 키우게 된다.

# :::{admonition} `fit()` 메서드와 샘플 가중치
# :class: info
# 
# 사이킷런 모델의 `fit()` 메서드는 `sample_weight` 옵션인자를 이용하여
# 각 훈련 샘플에 대한 가중치를 지정할 수 있다.
# :::

# **에이다부스트 알고리즘 작동 과정**
# 
# 아래 두 개의 그래프는 
# 초승달 데이터셋에 rbf 커널을 사용하는 SVC 모델을 5번 연속 새로 훈련시킨 결과를 보여준다.
# 새로 훈련시킬 때마다 이전에 제대로 학습하지 못한 샘플에 대한 샘플 가중치를 키워
# 해당 샘플에 보다 집중하여 훈련하도록 유도된다.
# 
# 왼쪽과 오른쪽은 새 모델을 훈련하기 시작할 때 적용할 샘플 가중치의 적용 정도를 
# 지정하는 학습률(`learnign_rate`)을 달리한 결과를 보여준다.
# 
# - 왼쪽 그래프: 학습률 1
# - 오른쪽 그래프: 학습률 0.5
# 
# 왼쪽 그래프는 학습률을 너무 크게 잡으면 각 모델의 변화가 커질 수 있음을 보여준다.
# 이유는 기존의 모델을 새로운 모델로 업데이트할 때 기존 모델로부터 계산된 샘플 가중치를
# 계산된 그대로 또는 그 이상으로 각 샘플에 적용하면 그만큼 모델의 변화가 커져서
# 하나의 좋은 모델로 수렴하지 못할 수 있기 때문이다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-09.png" width="600"/></div>

# :::{admonition} 학습률
# :class: info
# 
# 여기서 말하는 학습률은 {ref}`경사하강법의 학습률 <sec:gradient-descent>`과 
# 다른 개념이지만 모델을 업데이트할 때 
# 샘플 가중치의 적용 정도를 조절한다는 의미에서 비슷한 의미로 사용된다.
# :::

# **사이킷런의 에이다부스트 모델**
# 
# 사이키런에서 제공하는 에이다부스트 모델은 두 개다.
# 
# * 분류 모델: `AdaBoostClassifier` 
# * 회귀 모델: `AdaBoostRegressor`

# 아래 코드는 결정트리 분류 모델에 에이다부스트 기법을 적용한다.
# 
# * `n_estimators=30`: 30번 부스팅 반복
# * `learning_rate=0.5`: 학습률 0.5
# 
# ```python
# ada_clf = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=1), n_estimators=30,
#     learning_rate=0.5, random_state=42)
# ```
# 
# 초승달 데이터셋을 대상으로 훈련시킨 결과는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-10.png" width="400"/></div>

# ### 그레이디언트 부스팅

# **그레이디언트 부스팅**<font size='2'>Gradient Boosting</font> 기법 또한 
# 이전 모델의 예측에 오차가 있다면 그 오차를 보정하는 새로운 예측기를 새롭게 훈련시킨다. 
# 에이다부스트 기법이 샘플의 가중치를 조정하는 반면에
# 그레이디언트 부스팅 기법은 이전 예측기에 의해 생성된 **잔차**(residual error)에 대해 
# 새로운 예측기를 학습시킨다. 

# :::{admonition} 잔차
# :class: info
# 
# 잔차<font size='2'>residual error</font>는 예측값과 실제값 사이의 오차를 가리킨다.
# :::

# **사이킷런 그레이디언트 부스팅 모델**
# 
# 사이키런에서 제공하는 그레이디언트 부스팅 모델은 두 개다.
# 
# * 분류 모델: `GradientBoostingClassifier`
# * 회귀 모델: `GradientBoostingRegressor`
# 
# 두 모델 모두 결정트리 모델을 연속적으로 훈련시킨다.

# **예제: GBRT**
# 
# 아래 그래프는 2차 다항식 모양의 훈련 데이터셋에 결정트리 모델을 3번 연속 적용하면서
# 생성한 예측값의 변화과정을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-11.png" width="700"/></div>

# 위 그래프는 아래 `GradientBoostingRegressor` 모델을 훈련할 때 실제로 학습되는 과정을 잘 보여준다.

# ```python
# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
#                                  learning_rate=1.0, random_state=42)
# ```

# :::{admonition} GBRT
# :class: info
# 
# GBRT는 Gradient Boosted Regression Trees, 즉 '그레이디언트 부스팅 회귀 나무'의 줄임말이다. 
# :::

# **학습률과 축소 규제**

# 학습률(`learnign_rate`)는 그레이디언트 부스팅 기법으로 훈련된 모델의 
# 예측값을 정할 때 훈련 과정에 사용된 각 모델의 예측값이 기여하는 정도를 결정한다.
# 
# 학습률이 0.1 등처럼 작게 잡으면 보다 많은 수의 모델을 훈련시켜야 하지만 
# 그만큼 일반화 성능이 좋은 모델이 훈련된다.
# 이런 방식으로 훈련 과정을 규제하는 기법을 **축소 규제**<font size='2'>shrinkage regularization</font>다. 
# 즉, 훈련에 사용되는 각 모델의 기여도를 어느 정도로 축소할지 결정하는 방식으로
# 모델의 훈련을 규제한다는 의미다. 
# 
# 아래 두 그래프는 학습률이 1인 경우(왼쪽)와 0.05인 경우(오른쪽)의 차이를 보여준다.
# 학습률이 1인 경우 모델 훈련을 세 번만 반복해서 과소적합이 발생했지만, 
# 학습률이 0.05인 경우 모델 훈련을 92번 반복해서 적절한 모델을 훈련시켰다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-12a.png" width="700"/></div>

# **조기 종료**

# 훈련 모델의 수를 너무 크게 잡으면 과대적합의 위험성은 커지게 된다.
# 따라서 훈련되는 모델의 적절한 수를 알아내는 일이 중요하다.
# 이를 위해 그리드 탐색 기법, 랜덤 탐색 기법 등을 사용할 수 있다.
# 
# 하지만 `GradientBoostingRegressor` 모델의 `n_iter_no_change` 하이퍼파라미터를 지정하면
# 간단하게 {ref}`조기 종료 <sec:early-stopping>` 기법을 적용할 수 있다.
# 
# 아래 코드는 `n_iter_no_change=10` 을 사용하여 모델이 연속적으로 10번 제대로 개선되지 못하는 경우
# 훈련을 종료시킨다.
# 원래 500 번 연속 결정트리를 훈련시켜야 하지만 실제로는 훨씬 일찍 훈련을 종료한다. 
# 
# ```python
# gbrt_best = GradientBoostingRegressor(
#     max_depth=2, learning_rate=0.05, n_estimators=500,
#     n_iter_no_change=10, random_state=42)
# ```

# **확률적 그레이디언트 부스팅**
# 
# `subsample` 하이퍼파라리미터를 이용하여 각 결정트리가 훈련에 사용할 훈련 샘플의 비율을 지정한다.
# 예를 들어 `subsample=0.25` 로 설정하면 각 결정트리 모델은 전체 훈련셋의 25% 정도만
# 이용해서 훈련한다. 훈련 샘플은 매번 무작위로 선택된다.
# 이 방식을 사용하면 훈련 속도가 빨라지며, 편향은 높아지지만, 모델의 다양성이 많아지기에 분산은 낮아진다.

# ### 히스토그램 그레이디언트 부스팅

# 대용량 데이터셋을 이용하여 훈련해야 하는 경우 
# **히스토그램 그레이디언트 부스팅**<font size='2'>Histogram-based Gradient Boosing</font>(HGB)
# 기법이 추천된다.
# 이 기법은 훈련 샘플의 특성값을 최대 255개의 구간으로 분류한다.
# 즉, 샘플의 특성이 최대 255개의 값 중에 하나라는 의미다.
# 원래는 훈련 샘플 수만큼 다른 특성값이 존재한다.
# 
# 이렇게 하면 결정트리의 CART 알고리즘이 적용될 때 최적의 임곗값을 결정할 때
# 확인해야 하는 경우의 수가 매우 작아지기에 수 백배 이상 빠르게 학습된다.
# 또한 특성값이 모두 정수이기에 메모리도 보다 효율적으로 사용한다. 
# 실제로 하나의 결정트리 모델의 훈련 시간 복잡도는 원래 $O(n\times m \times \log(m))$ 이지만
# 히스토그램 방식을 사용하면 $O(n\times m)$ 로 줄어든다.
# 이유는 특성값을 정렬할 필요가 없어지기 때문이다.
# 물론 모델의 정확도는 떨어진다. 하지만 경우에 따라 과대적합을 방지하는 규제 역할도 수행한다.

# **사이킷런의 히스토그램 그레이디언트 부스팅 모델**
# 
# * `HistGradientBoostingRegressor`: 회귀 모델
# * `HistGradientBoostingClassifier`: 분류 모델
# 
# `GradientBoostingRegressor`, `GradientBoostingClassifier` 등과 유사하게 작동한다.

# **장점**
# 
# 언급된 두 모델은 범주 특성을 다룰 수 있으며, 결측치도 처리할 수 있다.
# 결측치는 255개의 구간 이외에 특별한 구간에 들어가는 것으로 간주된다.

# **예제**

# 아래 코드는 {ref}`캘리포니아 주택가격 데이터셋 <ch:end2end>`을 이용하여
# 히스토그램 그레이디언트 부스팅 모델을 적용하는 것을 보여준다.
# 
# - `(OrdinalEncoder(), ["ocean_proximity"])` : 해안 근접도 특성값으로 사용된 5개를 0에서 4사이의 정수로 변환하는 변환기 지정.
# - `categorical_features=[0]` : 범주형 특성의 위치 지정
# - 캘리포니아 주택가격 데이터셋에 결측치 존재하지만 전처리로 다루지 않음.
# - 스케일링, 원-핫-인코딩 등도 필요하지 않음.

# ```python
# hgb_reg = make_pipeline(
#     make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
#                             remainder="passthrough"),
#     HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
# )
# 
# hgb_reg.fit(housing, housing_labels)
# ```

# :::{admonition} 기타 그레이디언트 부스팅 모델
# :class: tip
# 
# 다음 모델들이 제3자 라이브러리로 제공되지만 사이킷런의 모델과 유사하게 작동한다.
# 
# - XGBoost
# - CatBoost
# - LightGBM
# 
# 또한 
# [텐서플로우<font size='2'>Tensorflow</font>의 랜덤 포레스트와 GBRT 관련 다양한 최신 모델](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras)도 있다.
# :::

# ## 스태킹

# 배깅방식의 응용으로 볼 수 있는 기법이다.
# 다수결을 이용하는 대신 여러 예측값을 훈련 데이터로 활용하는 예측기를 훈련시킨다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-13.png" width="400"/></div>

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
# 
# 레이어를 차례대로 실행해서 믹서기(블렌더)가 예측한 값을 예측값으로 지정한다.
# 훈련된 스태킹 모델의 편향과 분산이 훈련에 사용된 모델들에 비해 모두 감소한다.

# **다층 스태킹**

# 2층에서 여러 개의 믹서기(블렌더)를 사용하고, 
# 그위 3층에 새로운 믹서기를 추가하는 방식으로 다층 스태킹을 훈련시킬 수 있다.
# 다층 스태킹의 훈련 방식은 2층 스태킹의 훈련 방식을 반복하면 된다.

# **예제: 3층 스태킹 모델 훈련과정**

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch07/homl07-17.png" width="400"/></div>

# ## 연습문제

# 참고: [(실습) 앙상블 학습과 랜덤 포레스트](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_ensemble_learning_random_forests.ipynb)
