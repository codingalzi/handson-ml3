#!/usr/bin/env python
# coding: utf-8

# (ch:end2end)=
# # 머신러닝 프로젝트 처음부터 끝까지

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 머신러닝 프로젝트 처음부터 끝까지](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_end2end_ml_project.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**
# 
# 주택 가격을 예측하는 다양한 **회귀 모델**<font size="2">regression model</font>의
# 훈련 과정을 이용하여 머신러닝 시스템의 전체 훈련 과정을 상세히 살펴본다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-01d.png" width="600"></div>

# ## 실전 데이터 활용

# 다양하고 수많은 실전 데이터를 모아놓은 데이터 저장소를
# 머신러닝 공부에 잘 활용할 수 있어야 한다. 
# 여기서 사용하는 데이터는 1990년 미국 캘리포니아 주에서 수집한 인구조사 자료이며,
# 데이터의 원본은 다양한 공개 저장소에서 다운로드할 수 있다.
# 
# 가장 유명한 데이터 저장소는 다음과 같다.
# 
# * [OpenML](https://www.openml.org/)
# * [캐글(Kaggle) 데이터셋](http://www.kaggle.com/datasets)
# * [페이퍼스 위드 코드](https://paperswithcode.com/)
# * [UC 얼바인(UC Irvine) 대학교 머신러닝 저장소](http://archive.ics.uci.edu/ml)
# * [아마존 AWS 데이터셋](https://registry.opendata.aws)
# * [텐서플로우 데이터셋](https://www.tensorflow.org/datasets)

# ## 큰 그림 그리기

# 머신러닝으로 해결하고자 하는 문제를 파악하기 위해
# 주어진 데이터에 대한 기초적인 정보,
# 문제 정의, 
# 문제 해결법 등을 구상해야 한다. 

# ### 데이터 정보 확인

# 1990년도에 시행된 미국 캘리포니아 주의 20,640개 구역별 인구조사 데이터는
# 경도, 위도, 중간 주택 연도, 방의 총 개수, 침실 총 개수, 인구, 가구 수, 중간 소득, 중간 주택 가격, 해안 근접도
# 등을 포함한다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/LA-USA01.png" width="600"></div>

# ### 학습 모델 확인

# 구역별 중간 주택 가격을 예측하는 시스템에 활용될
# 회귀 모델을 훈련시킨다.
# 훈련시킬 모델의 특성은 다음과 같다.
# 
# * 지도 학습: 구역별 '중간 주택 가격'을 레이블(타깃)로 지정한다.
# 
# * 회귀: 가격을 예측한다. 보다 세분화하면 다중 회귀이지 단변량 회귀 모델이다.
#   * 다중 회귀<font size="2">multiple regression</font>: 구역별로 여러 특성을 주택 가격 예측에 사용한다.
#   * 단변량 회귀<font size="2">univariate regression</font>: 구역별로 한 종류의 값만 예측한다.
# 
# * 배치 학습: 빠르게 변하는 데이터에 적응할 필요가 없으며, 데이터셋의 크기도 충분히 작다.

# :::{admonition} 다변량 회귀
# :class: info
# 
# 다변량 회귀<font size="2">multivariate regression</font>는 여러 종류의 값을 동시에 예측한다.
# :::

# ### 회귀 모델 성능 측정 지표 선택

# 훈련중인 회귀 모델의 성능을 평가하는 지표로 
# 예측값과 타깃 사이의 오차를 활용하는 아래 두 방식 중 하나를 사용한다.
# 
# * 평균 제곱근 오차(RMSE)
# * 평균 절대 오차(MAE)

# **평균 제곱근 오차(RMSE)**
# 
# 평균 제곱근 오차<font size="2">Root Mean Square Error</font>는
# 예측값과 타깃 사이의 오차의 제곱의 평균값이다. 
# 수학에서는 **유클리디안 노름** 또는 **$\ell_2$ 노름**으로 불린다.

# $$\text{RMSE}(\mathbf X, h) = \sqrt{\frac 1 m \sum_{i=1}^{m} (h(\mathbf x^{(i)}) - y^{(i)})^2}$$

# 위 수식에 사용된 기호의 의미는 다음과 같다.
# 
# * $\mathbf X$: 모델 성능 평가에 사용되는 데이터셋 전체 샘플들의 특성값들로 구성된 행렬, 레이블(타겟) 제외.
# * $m$: $\mathbf X$의 행의 수. 즉, 훈련 데이터셋 크기.
# * $\mathbf x^{(i)}$: $i$ 번째 샘플의 전체 특성값 벡터. 레이블(타겟) 제외.
# * $y^{(i)}$: $i$ 번째 샘플의 레이블(타깃)
# * $h$: 예측 함수
# * $h(\mathbf x^{(i)})$: $i$번째 샘플에 대한 예측 값. $\hat y^{(i)}$로 표기되기도 함.

# :::{prf:example} 훈련셋과 2D 어레이
# :label: 2d-array
# 
# 모델 훈련에 사용되는 훈련셋에
# $m$ 개의 샘플이 포함되어 있고 각각의 샘플이 $n$ 개의 특성을 갖는다면
# 훈련셋은 $(m, n)$ 모양의 numpy의 2D 어레이로 지정된다.
# 
# 예를 들어, $m = 5$, $n = 4$ 이면 훈련셋 $\mathbf X$는 다음과 같이
# 표현된다.
# 
# ```python
# array([[-118.29, 33.91, 1416, 38372],
#        [-114.30, 34.92, 2316, 41442],
#        [-120.38, 35.21, 3444, 29303],
#        [-122.33, 32.95, 2433, 24639],
#        [-139.31, 33.33, 1873, 50736]])
# ```
# 
# 각각의 $\mathbf{x}^{(i)}$는 $i$ 번째 행에 해당한다. 
# 예를 들어 $\mathbf{x}^{(1)}$은 첫째 행의 1D 어레이를 가리킨다. 
# 
# ```python
# array([-118.29, 33.91, 1416, 38372])
# ```
# 
# 단변량 회귀에서 $y^{(i)}$ 는 보통 부동소수점을 가리키며, 
# 다변량 회귀에서는 $\mathbf{x}^{(i)}$ 처럼 
# 여러 개의 타깃 값으로 구성된 1D 어레이로 표현된다.
# :::

# **평균 절대 오차(MAE)**
# 
# 평균 절대 오차<font size="2">Mean Absolute Error</font>는
# 맨해튼 노름 또는 $\ell_1$ 노름으로도 불리며
# 예측값과 타깃 사이의 오차의 평균값이다.
# 
# $$\text{MAE}(\mathbf X, h) = \frac 1 m \sum_{i=1}^{m} \mid h(\mathbf x^{(i)}) - y^{(i)} \mid$$
# 
# 훈련 데이터셋에 이상치가 많이 포함된 경우 주로 사용되지만,
# 그렇지 않다면 일반적으로 RMSE가 선호된다.

# ## 데이터 다운로드 및 적재

# 캐리포니아 주택가격 데이터셋은 매우 유명하여 많은 공개 저장소에서 다운로드할 수 있다.
# 여기서는 저자가 자신의 깃허브에 압축파일로 저장한 파일을 다운로드해서 사용한다. 

# ### 데이터셋 기본 정보 확인

# pandas의 데이터프레임으로 데이터셋을 적재하여 기본적인 데이터 구조를 훑어볼 수 있다.

# **`head()` 메서드 활용**
# 
# 데이터프레임 객체의 처음 5개 샘플을 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-05.png" width="600"></div>

# **`info()` 메서드 활용**
# 
# 데이터셋의 정보를 요약해서 보여준다.
# 
# * 구역 수: 20,640개. 한 구역의 인구는 600에서 3,000명 사이.
# * 구역별로 경도, 위도, 중간 주택 연도, 해안 근접도 등 총 10개의 조사 항목
#     * '해안 근접도'는 범주형 특성이고 나머지는 수치형 특성.
# * '방의 총 개수'의 경우 207개의 null 값, 즉 결측치 존재.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-05a.png" width="450"></div>

# **범주형 특성 탐색**
# 
# '해안 근접도'는 5개의 범주로 구분된다.
# 
# | 특성값 | 설명 |
# | --- | --- |
# | <1H OCEAN | 해안에서 1시간 이내 |
# | INLAND | 내륙 |
# | NEAR OCEAN | 해안 근처 |
# | NEAR BAY | 샌프란시스코의 Bay Area 구역 |
# | ISLAND | 섬  |

# **수치형 특성 탐색**
# 
# `describe()` 메서드는 수치형 특성들의 정보를 요약해서 보여준다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/housing-describe.png"></div>

# `hist()` 메서드는 수치형 특성별 히스토그램을 그린다.
# 히스토그램을 통해 각 특성별 데이터셋의 다양한 정보를 확인할 수 있다.
# 
# - 각 특성마다 사용되는 단위와 스케일(척도)가 다르다.
# - 일부 특성은 한쪽으로 치우쳐저 있다.
# - 일부 특성은 값을 제한한 것으로 보인다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/feature-histogram.png" width="600px"></div>

# ### 테스트셋 만들기

# 모델 학습 시작 이전에 준비된 데이터셋을 훈련셋과 테스트셋으로 구분해야 한다.
# 테스트셋은 훈련 과정중에 전혀 사용되지 않으며 보통 전체 데이터셋의 20% 정도 이하로 
# 전체 데이터셋의 크기에 따라 적절히 조절한다.
# 
# 테스트셋에 대한 정보는 절대로 모델 훈련에 이용하지 않아야 한다.
# 그렇지 않으면 미래에 실전에서 사용되는 데이터를 미리 안다고 가정하고 모델을 훈련시키는
# 것과 동일하게 되며, 이런 방식은 매우 잘못된 모델을 훈련시킬 위험을 키운다.
# 
# 데이터셋을 훈련셋과 데이터셋으로 구분할 때 보통 계층적 샘플링을 사용한다.

# **계층적 샘플링**
# 
# 각 계층별로 적절한 샘플을 추측하는 기법이다. 
# 이유는 계층별로 충분한 크기의 샘플이 포함되도록 지정해야 학습 과정에서 편향이 발생하지 않는다.
# 예를 들어, 특정 소득 구간에 포함된 샘플이 과하게 적거나 많으면 해당 계층의 중요도가 과대 혹은 과소 평가될 수 있다.
# 
# 캘리포니아 데이터셋의 중간 소득을 대상으로하는 히스토그램을 보면
# 대부분 구역의 중간 소득이 1.5~6.0, 즉 15,000 달러에서 60,000 달러 사이인 것을 알 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-08.png" width="400"></div>

# 소득 구간을 아래 숫자들을 기준으로 5개로 구분한 다음에 계층적 샘플링을 이용하여
# 훈련셋과 테스트셋을 구분하면 무작위 샘플링 방식과는 분명히 다르게
# 계층별 샘플의 비율을 거의 동일하게 유지한다.
# 
# | 구간 | 범위 |
# | :---: | :--- |
# | 1 | 0 ~ 1.5 |
# | 2 | 1.5 ~ 3.0 |
# | 3 | 3.0 ~ 4.5 |
# | 4 | 4.5 ~ 6.0 |
# | 5 | 6.0 ~  |

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-07.png" width="500"></div>

# ## 데이터 탐색과 시각화

# 테스트셋을 제외한 훈련셋에 대해서 시각화를 이용하여 데이터셋을 탐색한다.

# ### 지리적 데이터 시각화

# 경도와 위도 정보를 이용하여 구역을 산포도로 나타내면 인구의 밀집 정도를 확인할 수 있다. 
# 예를 들어, 샌프란시스코의 베이 에어리어, LA, 샌디에고 등 유명 대도시의 인구 밀도가 높다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-09.png" width="500"></div>

# 유명 대도시의 인구 밀도가 높은 특정 구역의 주택 가격이 높다는 일반적인 사실 또한 산포도록 확인할 수 있다.
# 산포도를 그릴 때 해당 구역의 중간 주택 가격을 색상으로, 
# 인구밀도는 원의 크기로 활용한 결과는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-11.png" width="500"></div>

# ### 상관관계 조사

# 중간 주택 가격 특성과 다른 특성 사이의 선형 상관관계를 나타내는 상관계수는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-12.png" width="600"></div>

# **상관계수의 특징**
# 
# 상관계수는 -1에서 1 사이의 값으로 표현된다.
# 
# * 1에 가까울 수록: 강한 양의 선형 상관관계
# * -1에 가까울 수록: 강한 음의 선형 상관관계
# * 0에 가까울 수록: 매우 약한 선형 상관관계

# :::{admonition} 상관계수와 상관관계
# :class: warning
# 
# 상관계수가 0이라는 것은 선형 관계가 없다는 의미이지 서로 아무런 상관관계가 없다는 의미는 아니다.
# 또한 선형계수가 1이라 하더라도 산점도에서 데이터의 선형관계를 보여주는 직선의 기울기는 아무런 상관이 없다.
# 아래 그림은 상관계수의 이런 특성을 잘 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-14.png" width="400"></div>
# 
# <그림 출처: [위키백과](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)>
# :::

# 상관계수를 통해 중간 주택 가격과 중간 소득의 상관계수가 0.68로 가장 높다는 사실을 확인한다.
# 즉, 중간 소득이 올라가면 중간 주택 가격도 상승하는 경향이 있다.
# 하지만 점들이 너무 넓게 퍼져 있어서 완벽한 선형관계와는 거리 멀다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-13.png" width="400"></div>

# 위 산점도는 다음 사항들에 주의할 것을 잘 보여준다.
# 
# * 50만 달러에서 보이는 수평선은 가격을 제한한 결과로 보여진다.
# * 45만, 35만, 28만, 그 아래 정도에서도 수평선이 존재하는데 이유는 알려지지 않았다.
# * 이러한 이상한 모델이 형태를 학습하지 못하도록 해당 구역을 제거하는 것이 
#     일반적으로 좋다. 하지만 여기서는 그대로 두고 사용한다.

# 경우에 따라 기존의 특성을 조합해서 새로운 특성을 활용할 수도 있다.
# 예를 들어 구역별 방의 총 개수와 침실의 총 개수 대신 아래 특성이 보다 유용해 보인다.
# 
# * 가구당 방 개수(`rooms_for_house`)
# * 방 하나당 침실 개수(`bedrooms_ratio`)
# * 가구당 인원(`people_perhouse`)
# 
# 실제로 세 특성을 새로 추가한 다음에 상관계수를 확인하면 
# 방 하나당 침실 개수와 중간 주택 가격 사이의 선형 연관성이 
# 특성 조합에 사용된 다른 특성들에 비해 높게 나타난다.

# ## 머신러닝 알고리즘 훈련용 데이터 준비

# 머신러닝 모델 훈련에 사용되는 알고리즘을 이용하려면
# 적재된 데이터셋을 적절하게 준비해야 한다.
# 즉, 데이터 정제와 전처리 과정을 수행해서
# 바로 모델 훈련에 사용될 수 있도록 해야 한다. 
# 정제와 전처리 모든 과정은 __파이프라인__<font size="2">pipeline</font>으로
# 자동화해서 언제든지 재활용할 수 있도록 해야 한다.

# 먼저 앞서 계층별로 구분된 훈련셋 `strat_train_set` 을 
# 입력 데이터셋 과 
# 타깃 데이터셋으로 또다시 구분한다. 
# 
# * 입력 데이터셋: 중간 주택 가격 특성이 제거된 훈련셋 
# 
#     ```python
#     housing = strat_train_set.drop("median_house_value", axis=1)
#     ```
# * 타깃 데이터셋: 중간 주택 가격 특성으로만 구성된 훈련셋 
# 
#     ```python
#     housing_labels = strat_train_set["median_house_value"].copy()
#     ```

# 데이터 준비는 기본적으로 입력 데이터셋만을 대상으로 **정제**<font size="2">cleaning</font>와 
# **전처리**<font size="2">preprocessing</font> 단계로 실행된다. 
# 타깃 데이터셋은 결측치가 없는 경우라면 일반적으로 정제와 전처리 대상이 아니지만
# 경우에 따라 변환이 요구될 수 있다.
# 예를 들어, 타깃 데이터셋의 히스토그램이 지나치게 한쪽으로 치우치는 모양을 띠면
# 로그 함수를 적용하여 값을 변환하는 것이 권장된다.

# :::{admonition} 테스트셋 전처리
# :class: info
# 
# 테스트셋에 대한 전처리와 구분은 모든 훈련이 완성된 후에 
# 훈련됨 모델의 성능을 측정할 때 
# 기존에 완성된 파이트라인을 이용하면 된다.
# :::

# ### 데이터 정제와 전처리

# 데이터 정제는 결측치 처리, 이상치 및 노이즈 데이터 제거 등을 의미한다.
# 캘리포니아 주택 가격 데이터셋은 구역별 전체 방 개수(`total_rooms`) 특성에서 결측치가 일부 포함되어 있지만 
# 이상치 또는 노이즈 데이터는 없다고 가정한다. 

# 전처리는 수치형 특성과 범주형 특성을 나누어 수행한다. 
# 
# * 수치형 특성에 대한 전처리
#     * 특성 스케일링
#     * 특성 조합
# 
# * 범주형 특성 전처리 과정
#     * 원-핫-인코딩<font size="2">one-hot-encoding</font>

# 데이터 정제와 전처리의 모든 과정은 데이터셋에 포함된 샘플을 
# 한꺼번에 지정된 방식으로 변환한다.
# 따라서 모든 변환 과정을 자동화는
# __파이프라인__<font size="2">pipeline</font> 기법을 활용해야 한다.

# **사이킷런 API 활용**

# 사이킷런<font size="2">Scikit-Learn</font>에서 제공하는
# 머신러닝 관련 API를 활용하여 데이터 준비 과정을 자동화하는 파이프라인을 쉽게 구현할 수 있다.
# 파이프라인 구성이 간단한 이유는 사이킷런의 API를 합성하는 기능일 기본으로 지원하기 때문이다.
# 이를 이해하려면 먼저 사이킷런이 제공하는 API의 유형을 구분할 줄 알아야 한다.
# 
# 사이킷런의 API는 크게 세 종류의 클래스로 나뉜다.

# * 추정기<font size="2">estimator</font>
#     * 인자로 주어진 데이터셋 객체 관련된 특정 값 계산
#     * `fit()` 메서드: 계산된 특정 값으로 업데이트된 데이터 객체 자신 반환

# * 변환기<font size="2">transformer</font>
#     * `fit()` 메서드 이외에 `fit()` 가 계산한 값을 이용하여 데이터셋을 변환하는 `transform()` 메서드 지원.
#     * `fit()` 메서드와 `transform()` 메서드를 연속해서 호출하는 `fit_transform()` 메서드도 지원.

# * 예측기<font size="2">predictor</font>
#     * `fit()` 메서드 이외에 `fit()` 가 계산한 값을 이용하여 
#         타깃을 예측하는 `predict()` 메서드 지원.
#     * `predict()` 메서드가 예측한 값의 성능을 측정하는 `score()` 메서드 지원.
#     * 일부 예측기는 추정치의 신뢰도를 평가하는 기능도 제공

# 사이킷런의 API는 또한 적절한 
# **하이퍼파라미터**<font size="2">hyperparameter</font>로 초기화되어 있으며
# 추정 및 변화 관정에 필요한 모든 파라미터를 저장하며 효율적으로 관리한다.

# :::{admonition} 하이퍼파라미터 대 파라미터
# :class: info
# 
# 사이킷런 API의 하이퍼파라미터는 해당 객체를 생성할 때 사용되는 값을 가리킨다.
# 즉, API 객체를 생성하기 위해 해당 API 클래스의 생성자인 
# `__init__()` 메서드를 호출할 때 사용하는 인자를 가리킨다.
# 
# 반면에 파라미터는 지정된 API 객체의 `fit()` 메서드가 실행되는 과정에 
# 주어진 입력 데이터셋 관련해서 계산하는 값을 가리킨다.
# 추정기, 변환기, 예측기는 각각의 역할에 맞는 파라미터를 계산한다.
# :::

# ### 데이터 정제

# 입력 데이터셋의 `total_bedrooms` 특성에 207개 구역에 대한 값이 null 값으로 채워져 있다.
# 즉, 일부 구역에 대한 전체 방의 개수 정보가 누락되었다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/null-value01.png" width="800"></div>

# 머신러닝 모델은 결측치가 있는 데이터셋을 잘 활용하지 못한다.
# 따라서 아래 옵션 중 하나를 선택해서 데이터를 정제해야 한다.
# 
# * 옵션 1: 해당 구역 제거
# * 옵션 2: 해당 특성 삭제
# * 옵션 3: 평균값, 중앙값, 0, 주변에 위치한 값 등 특정 값으로 채우기. 

# 여기서는 중앙값으로 채우는 옵션 3 방식을 사이킷런의 `SimpleImputer` 변환기를 이용하여 적용한다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/null-value02.png" width="800"></div>

# ### 범주형 특성 다루기: 원-핫 인코딩

# 해안 근접도(`ocean_proximity`)는 수가 아닌 아닌 5 개의 범주를 나타내는 텍스트를 값으로 사용한다.
# 그런데 머신러닝 모델은 일반적으로 텍스트 데이터를 처리하지 못한다. 

# 가장 단순한 해결책으로 5 개의 범주를 정수로 변환할 수 있다.
# 
# | 범주 | 숫자 |
# |---|---|
# | <1H OCEAN | 0 |
# | INLAND | 1 |
# | ISLAND | 2 |
# | NEAR BAY | 3 |
# | NEAR OCEAN | 4 |
# 
# 하지만 이 방식은 수의 크기 특성을 모델이 활용할 수 있기에 위험하다. 
# 예를 들어 바닷가 근처(`NEAR OCEAN`)에 위치한 주택이 가장 비쌀 것으로 모델이 학습할 수 있다.

# 범주형 특성을 수치화하는 가장 일반적인 방식은 
# **원-핫 인코딩**<font size="2">one-hot encoding</font>이다.
# 원-핫 인코딩은 수치화된 범주들 사이의 크기 비교를 피하기 위해 더미(dummy) 특성을 활용한다.
# 
# 원-핫 인코딩을 적용하면 해안 근접도 특성을 다섯 개의 범주 전부를 
# 새로운 특성으로 대체한다.
# 다섯 개의 특성에 사용되는 값은 다음 방식으로 지정한다.
# 
# * 해당 카테고리의 특성값: 1
# * 나머지 카테고리의 특성값: 0
# 
# 예를 들어, `INLAND`를 해안 근접도 특성값으로 간던 샘플은 다음 모양의 특성값을 갖게 된다.
# 
# `[0, 1, 0, 0, 0]`

# 사이킷런의 `OneHotEncoder` 변환기가 원-핫-인코딩을 지원한다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-16.png" width="600"></div>

# ### 특성 스케일링

# 머신러닝 알고리즘은 입력 데이터셋의 특성값들의 
# **스케일**<font size="2">scale</font>(척도)이 다르면 제대로 작동하지 않는다.
# 따라서 모든 특성의 척도를 통일하는 **스케일링**<font size="2">scaling</font> 전처리가 요구된다.

# :::{admonition} 타깃 데이터셋과 스케일링
# :class: warning
# 
# 타깃(레이블) 데이터셋에 대한 스케일링은 굳이 필요하지 않다.
# 다만 타깃 데이터셋의 분포가 한쪽으로 치우친 경우 로그 함수를 적용할 필요가 있을 수는 있다.
# 로그 함수 적용은 스케일링처럼 특성값들을 0과 1 근처의 값으로 몰아가지는 않는 대신에 
# 데이터의 분포가 보다 좌우 대칭을 갖도록 변화시킨다.
# :::

# 스케일링은 보통 아래 두 가지 방식을 사용한다. 
# 
# - min-max 스케일링
# - 표준화

# **min-max 스케일링**
# 
# **정규화**(normalization)라고도 불리며
# 아래 식을 이용하여 모든 특성값을 0에서 1 사이의 값으로 변환한다.
# 단, $max$ 와 $min$ 은 각각 특성값들의 최댓값과 최솟값을 가리킨다. 
# 
# $$
# \frac{x-min}{max-min}
# $$
# 
# min-max 스케일링은 이상치에 매우 민감하다.
# 예를 들어 이상치가 매우 크면 분모가 매우 커져서 변환된 값이 0 근처에 몰리게 된다.
# 사이킷런의 `MinMaxScaler` 변환기가 min-max 스케일링을 지원한다.

# **표준화(standardization)**
# 
# 아래식을 이용하여 특성값을 변환한다.
# 단, $\mu$ 와 $\sigma$ 는 각각 특성값들의 평균값과 표준편차를 가리킨다.
# 
# $$
# \frac{x-\mu}{\sigma}
# $$
# 
# 변환된 데이터셋은 **표준정규분포**를 따르며,
# 이상치에 상대적으로 영향을 덜 받는다.
# 여기서는 사이킷런의 `StandardScaler` 변환기를 이용하여 표준화를 적용한다.

# :::{admonition} 변환기 사용법
# :class: warning
# 
# `fit()` 과 `fit_transform()` 두 메서드 훈련셋에 대해서만 사용한다.
# 반면에 테스트셋, 검증셋, 새로운 데이터 등에 대해서는 `transform()` 메서드만 적용한다. 
# 즉, 훈련셋을 대상으로 계산된 파라미터를 이용하여 
# 훈련 이외의 경우에 `transform()` 메서드를 확인하여 데이터를 변환한다.
# :::

# ### 사용자 정의 변환기

# * 아래 특성 추가 용도 변환기 클래스 직접 선언하기
#   * 가구당 방 개수(rooms for household)
#   * 방 하나당 침실 개수(bedrooms for room)
#   * 가구당 인원(population per household)

# * 변환기 클래스: `fit()`, `transform()` 메서드를 구현하면 됨.
#     * 주의: fit() 메서드의 리턴값은 self

# #### 예제: CombinedAttributesAdder 변환기 클래스 선언
# 
# * `__init__()` 메서드: 생성되는 모델의 __하이퍼파라미터__ 지정 용도 
#     - 모델에 대한 적절한 하이퍼파라미터를 튜닝할 때 유용하게 활용됨.
#     - 예제: 방 하나당 침실 개수 속성 추가 여부
# * `fit()` 메서드: 계산해야 하는 파라미터가 없음. 바로 `self` 리턴
# * `transform()` 메서드: 넘파이 어레이를 입력받아 속성을 추가한 어레이를 반환

# ```python
# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, add_bedrooms_per_room = True):
#         ...
# 
#     def fit(self, X, y=None):
#         return self
# 
#     def transform(self, X):
#         ...
# ```

# #### 상속하면 좋은 클래스

# * `BaseEstimator` 상속: 하이퍼파라미터 튜닝 자동화에 필요한 `get_params()`, `set_params()` 메서드 제공 

# * `TransformerMixin` 상속: `fit_transform()` 자동 생성

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/custom-transformer.png" width="350"></div>
# 
# <그림 아이디어 출처: [Get the Most out of scikit-learn with Object-Oriented Programming](https://towardsdatascience.com/get-the-most-out-of-scikit-learn-with-object-oriented-programming-d01fef48b448)>

# ### 변환 파이프라인

# * 모든 전처리 단계가 정확한 순서대로 진행되어야 함

# * 사이킷런의 `Pipeline` 클래스를 이용하여 파이프라인 변환기 객체 생성 가능

# #### 수치형 특성 변환 파이프라인

# ```python
# num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy="median")),
#         ('attribs_adder', CombinedAttributesAdder()),
#         ('std_scaler', StandardScaler()),
#     ])
# ```

# * 인자: 이름과 추정기로 이루어진 쌍들의 리스트

# * 마지막 추정기 제외 나머지 추정기는 모두 변환기이어야 함.
#     * `fit_transform()` 메서드 지원

# * 파이프라인으로 정의된 추정기의 유형은 마지막 추정기의 유형과 동일
#     - `num_pipeline`는 변환기. 이유는 `std_scaler`가 변환기이기 때문임.

# * `num_pipeline.fit()` 호출: 
#     * 마지막 단계 이전 추정기: `fit_transform()` 메소드 연속 호출.
#         즉, 변환기가 실행될 때마다 변환도 동시에 진행.
#     * 마지막 추정기: `fit()` 메서드 호출

# #### 수치형 / 범주형 특성 전처리 과정 통합 파이프라인

# * 사이킷런의 `ColumnTransformer` 클래스를 이용하여 특성별로 지정된 전처리를 처리할 수 있도록 지정 가능

# - 인자: (이름, 추정기, 적용 대상 열(column) 리스트) 튜플로 이루어진 리스트

# - `fit()` 메서드에 pandas의 데이터프레임을 직접 인자로 사용 가능

# * 수치형 특성: `num_pipeline` 변환기
#     - 적용 대상 열(columns): `list(housing_num)`

# * 범주형 특성: `OneHotEncoder` 변환기
#     - 적용 대상 열(columns): `["ocean_proximity"]`

# ```python
# num_attribs = list(housing_num)
# cat_attribs = ["ocean_proximity"]
# 
# full_pipeline = ColumnTransformer([
#         ("num", num_pipeline, num_attribs),
#         ("cat", OneHotEncoder(), cat_attribs),
#     ])
# 
# housing_prepared = full_pipeline.fit_transform(housing)
# ```

# ## 모델 선택과 훈련

# * 목표 달성에 필요한 두 요소를 결정해야함
#   * 학습 모델
#   * 회귀 모델 성능 측정 지표

# * 목표: 구역별 중간 주택 가격 예측 모델

# * 학습 모델: 회귀 모델

# * 회귀 모델 성능 측정 지표: 평균 제곱근 오차(RMSE)를 기본으로 사용

# ### 훈련셋에서 훈련하고 평가하기

# * 지금까지 한 일
#     * 훈련셋 / 테스트셋 구분
#     * 변환 파이프라인을 활용한 데이터 전처리

# * 이제 할 일
#     * 예측기 모델 선택 후 훈련시키기
#     * 예제: 선형 회귀, 결정트리 회귀

# * 예측기 모델 선택 후 훈련과정은 매우 단순함.
#     * `fit()` 메서드를 전처리 처리가 된 훈련 데이터셋에 적용

# #### 선형 회귀 모델(4장)

# * 선형 회귀 모델 생성: 사이킷런의 **`LinearRegression`** 클래스 활용

# * 훈련 및 예측
#     - 훈련: `LinearRegression` 모델은 무어-펜로즈 역행렬을 이용하여 파라미터 직접 계산 (4장 참조)
#     - 예측: (여기서는 연습 용도로) 훈련셋에 포함된 몇 개 데이터를 대상으로 예측 실행
# 
# ```python
# from sklearn.linear_model import LinearRegression
# 
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# 
# lin_reg.predict(housing_prepared)
# ```

# #### 선형 회귀 모델의 훈련셋 대상 예측 성능

# - RMSE(평균 제곱근 오차)가 68628.198 정도로 별로 좋지 않음.

# * 훈련된 모델이 훈련셋에 __과소적합__ 됨.
#     - 보다 좋은 특성을 찾거나 더 강력한 모델을 적용해야 함.
#     - 보다 좋은 특성 예제: 로그 함수를 적용한 인구수 등
#     - 모델에 사용되는 규제(regulaization, 4장)를 완화할 수도 있지만 위 모델에선 어떤 규제도 적용하지 않았음.

# #### 결정트리 회귀 모델(6장)

# * 결정 트리 모델은 데이터에서 복잡한 비선형 관계를 학습할 때 사용

# * 결정트리 회귀 모델 생성: 사이킷런의 **`DecisionTreeRegressor`** 클래스 활용

# * 훈련 및 예측
#     - 예측은 훈련셋에 포함된 몇 개 데이터를 대상으로 예측 실행
# 
# ```python
# from sklearn.tree import DecisionTreeRegressor
# 
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)
# 
# housing_predictions = tree_reg.predict(housing_prepared)
# ```

# #### 결정트리 회귀 모델의 훈련셋 대상 예측 성능

# - RMSE(평균 제곱근 오차)가 0으로 완벽해 보임.

# * 훈련된 모델이 훈련셋에 심각하게 __과대적합__ 됨.
#     - 실전 상황에서 RMSE가 0이 되는 것은 불가능.
#     - 훈련셋이 아닌 테스트셋에 적용할 경우 RMSE가 크게 나올 것임.

# ### 교차 검증을 사용한 평가

# * 테스트셋을 사용하지 않으면서 훈련 과정을 평가할 수 있음.

# * __교차 검증__ 활용

# #### k-겹 교차 검증

# * 훈련셋을 __폴드__(fold)라 불리는 k-개의 부분 집합으로 무작위로 분할
# 
# * 총 k 번 지정된 모델을 훈련
#     * 훈련할 때마다 매번 다른 하나의 폴드 선택하여 검증 데이터셋으로 활용
#     * 다른 (k-1) 개의 폴드를 이용해 훈련
# 
# * 최종적으로 k 번의 평가 결과가 담긴 배열 생성

# * k = 5인 경우
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/cross-val10.png" width="400"></div>

# #### 예제: 결정 트리 모델 교차 검증 (k = 10인 경우)

# ```python
# from sklearn.model_selection import cross_val_score
# 
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# 
# tree_rmse_scores = np.sqrt(-scores)
# ```

# * k-겹 교차 검증의 모델 학습 과정에서 성능을 측정할 때 높을 수록 좋은 __효용함수__ 활용
#     * `scoring="neg_mean_squared_error"`
#     * RMSE의 음숫값
# 

# * 교차 검증의 RMSE: 다시 음숫값(`-scores`) 사용
#     - 평균 RMSE: 약 71407
#     - 별로 좋지 않음.

# #### 예제: 선형 회귀 모델 교차 검증 (k = 10 인 경우)

# ```python
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# 
# lin_rmse_scores = np.sqrt(-lin_scores)
# ```

# * 교차 검증의 RMSE 평균: 약 69052
#     - 결정트리 회귀 모델보다 좋음.

# #### 앙상블 학습 (7장)

# * 여러 개의 다른 모델을 모아서 하나의 모델을 만드는 기법

# * 머신러닝 알고리즘의 성능을 극대화는 방법 중 하나

# #### 랜덤 포레스트 회귀 모델 (7장)

# * 앙상블 학습에 사용되는 하나의 기법

# * 무작위로 선택한 특성을 이용하는 결정 트리 여러 개를 훈련 시킨 후 
#     훈련된 모델들의 평균 예측값을 예측값으로 사용하는 모델

# * 사이킷런의 `RandomForestRegressor` 클래스 활용

# * 훈련 및 예측
# 
# ```python
# from sklearn.ensemble import RandomForestRegressor
# 
# forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
# forest_reg.fit(housing_prepared, housing_labels)
# 
# housing_predictions = forest_reg.predict(housing_prepared)
# ```

# * 랜덤 포레스트 모델의 RMSE: 약 50182
#     - 지금까지 사용해본 모델 중 최고
#     - 하지만 여전히 과대적합되어 있음. 

# ## 모델 세부 튜닝

# * 살펴 본 모델 중에서 **랜덤 포레스트** 모델의 성능이 가장 좋았음

# * 가능성이 높은 모델을 선정한 후에 **모델 세부 설정을 튜닝**해야함

# * 튜닝을 위한 세 가지 방식
#   * **그리드 탐색**
#   * **랜덤 탐색**
#   * **앙상블 방법**

# ### 그리드 탐색

# * 지정한 하이퍼파라미터의 모든 조합을 교차검증하여 최선의 하이퍼파라미터 조합 찾기

# * 사이킷런의 `GridSearchCV` 활용

# #### 예제: 그리드 탐색으로 랜덤 포레스트 모델에 대한 최적 조합 찾기

# ```python
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]
# 
# forest_reg = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)
# grid_search.fit(housing_prepared, housing_labels)
# ```

# * 총 (3x4 + 2x3 = 18) 가지의 경우 확인

# * 5-겹 교차검증(`cv=5`)이므로, 총 (18x5 = 90)번 훈련함.

# #### 그리드 탐색 결과 

# * 최고 성능의 랜덤 포레스트 하이퍼파라미터가 다음과 같음. 
#     - `max_features`: 8
#     - `n_estimators`: 30
#     - 지정된 구간의 최고값들이기에 구간을 좀 더 넓히는 게 좋아 보임

# * 최고 성능의 랜덤 포레스트에 대한 교차검증 RMSE: 49682
#     - 하나의 랜덤 포레스트보다 좀 더 좋아졌음.

# ### 랜덤 탐색

# * 그리드 탐색은 적은 수의 조합을 실험해볼 때 유용

# * 조합의 수가 커지거나, 설정된 탐색 공간이 커지면 랜덤 탐색이 효율적
#   * 설정값이 연속적인 값을 다루는 경우 랜덤 탐색이 유용

# * 사이킷런의 `RandomizedSearchCV` 추정기가 랜덤 탐색을 지원

# #### 예제: 랜덤 탐색으로 랜덤 포레스트 모델에 대한 최적 조합 찾기

# ```python
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
# 
# param_distribs = {
#         'n_estimators': randint(low=1, high=200),
#         'max_features': randint(low=1, high=8),
#     }
# 
# forest_reg = RandomForestRegressor(random_state=42)
# rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                 n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
# ```

# * `n_iter=10`: 랜덤 탐색이 총 10회 진행
#     * `n_estimators`와 `max_features` 값을 지정된 구간에서 무작위 선택

# * `cv=5`: 5-겹 교차검증. 따라서 랜덤 포레스트 학습이 (10x5=50)번 이루어짐.

# #### 랜덤 탐색 결과 

# * 최고 성능의 랜덤 포레스트 하이퍼파라미터가 다음과 같음. 
#     - `max_features`: 7
#     - `n_estimators`: 180

# * 최고 성능의 랜덤 포레스트에 대한 교차검증 RMSE: 49150

# ### 앙상블 방법

# * 결정 트리 모델 하나보다 랜덤 포레스트처럼 여러 모델로 이루어진 모델이 보다 좋은 성능을 낼 수 있음.

# * 또한 최고 성능을 보이는 서로 다른 개별 모델을 조합하면 보다 좋은 성능을 얻을 수 있음

# * 7장에서 자세히 다룸

# ### 최상의 모델과 오차 분석

# * 그리드 탐색과 랜덤 탐색 등을 통해 얻어진 최상의 모델을 분석해서 문제에 대한 좋은 통창을 얻을 수 있음

# * 예를 들어, 최상의 랜덤 포레스트 모델에서 사용된 특성들의 중요도를 확인하여 일부 특성을 제외할 수 있음.
#     * 중간 소득(median income)과 INLAND(내륙, 해안 근접도)가 가장 중요한 특성으로 확인됨
#     * 해안 근접도의 다른 네 가지 특성은 별로 중요하지 않음
#     * 중요도가 낮은 특성은 삭제할 수 있음.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/feature-importance.png" width="400"></div>

# ### 테스트 셋으로 시스템 평가하기

# 1. 최고 성능 모델 확인: 예를 들어, 그리드 탐색으로 찾은 최적 모델 사용
# 
# ```python
# final_model = grid_search.best_estimator_
# ```

# 2. 테스트셋 전처리
#     * 전처리 파이프라인의 `transform()` 메서드를 직접 활용
#     * __주의__: `fit()` 메서드는 전혀 사용하지 않음

# 3. 최고 성능 모델을 이용하여 예측하기

# 4. 최고 성능 모델 평가 및 론칭

# #### 최상의 모델 성능 평가

# * 테스트셋에 대한 최고 성능 모델의 RMSE: 47730

# #### 최상의 모델 성능 배포

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/model-launching01.png" width="600"></div>

# #### 데이터셋 및 모델 백업

# * 완성된 모델은 항상 백업해 두어야 함. 업데이트된 모델이 적절하지 않은 경우 이전 모델로 되돌려야 할 수도 있음.
#     * 백업된 모델과 새 모델을 쉽게 비교할 수 있음.

# * 동일한 이유로 모든 버전의 데이터셋을 백업해 두어야 함.
#     * 업데이트 과정에서 데이터셋이 오염될 수 있기 때문임.
