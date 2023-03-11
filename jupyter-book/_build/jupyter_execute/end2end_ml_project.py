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
# 훈련 과정을 이용하여 머신러닝 시스템의 전체 훈련 과정을 살펴본다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-01d.png" width="600"></div>

# 특히 데이터 정제 및 전처리 과정으로 구성된 데이터 준비와
# 최선의 모델을 찾는 과정을 상세히 소개한다. 

# ## 실전 데이터 활용

# 다양한 실전 데이터를 모아놓은 데이터 저장소를
# 머신러닝 공부에 잘 활용할 수 있어야 한다. 
# 가장 유명한 데이터 저장소는 다음과 같다.
# 
# * [OpenML](https://www.openml.org/)
# * [캐글(Kaggle) 데이터셋](http://www.kaggle.com/datasets)
# * [페이퍼스 위드 코드](https://paperswithcode.com/)
# * [UC 얼바인(UC Irvine) 대학교 머신러닝 저장소](http://archive.ics.uci.edu/ml)
# * [아마존 AWS 데이터셋](https://registry.opendata.aws)
# * [텐서플로우 데이터셋](https://www.tensorflow.org/datasets)
# 
# 여기서는 1990년 미국 캘리포니아 주에서 수집한 인구조사 데이터를 사용하며,
# 데이터의 원본은 다양한 공개 저장소에서 다운로드할 수 있다.

# ## 큰 그림 그리기

# 머신러닝으로 해결하고자 하는 문제를 파악하기 위해
# 주어진 데이터에 대한 기초적인 정보를 확인하고,
# 문제 파악 및 해결법 등을 구상해야 한다. 

# ### 데이터 정보 확인

# 1990년도에 시행된 미국 캘리포니아 주의 20,640개 구역별 인구조사 데이터는
# 경도, 위도, 중간 주택 연도, 방의 총 개수, 침실 총 개수, 인구, 가구 수, 중간 소득, 중간 주택 가격, 해안 근접도
# 등 총 10개의 특성을 포함한다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/LA-USA01.png" width="600"></div>

# ### 훈련 모델 확인

# **훈련 모델 종류**

# 구역별 중간 주택 가격을 예측하는 시스템에 활용될
# **회귀 모델**을 훈련시키고자 한다.
# 훈련시킬 모델의 특성은 다음과 같다.
# 
# * 지도 학습: 구역별 '중간 주택 가격'을 레이블(타깃)로 지정한다.
# 
# * 회귀: 가격을 예측한다. 보다 세분화하면 다중 회귀이자 단변량 회귀 모델이다.
#   * 다중 회귀<font size="2">multiple regression</font>: 구역별로 여러 특성을 주택 가격 예측에 사용한다.
#   * 단변량 회귀<font size="2">univariate regression</font>: 구역별로 한 종류의 값만 예측한다.
# 
# * 배치 학습: 빠르게 변하는 데이터에 적응할 필요가 없으며, 데이터셋의 크기도 충분히 작다.

# :::{admonition} 다변량 회귀
# :class: info
# 
# 다변량 회귀<font size="2">multivariate regression</font>는 여러 종류의 값을 동시에 예측한다.
# :::

# **훈련 모델 성능 측정 지표**

# 회귀 모델의 성능은 일반적으로 예측값과 타깃 사이의 오차를 활용하는 아래 
# 두 평가하는 지표 중 하나를 사용한다.
# 
# * 평균 제곱근 오차(RMSE)
# * 평균 절대 오차(MAE)

# **평균 제곱근 오차**<font size="2">Root Mean Square Error</font>(RMSE)는
# 예측값과 타깃 사이의 오차의 제곱의 평균값이다. 
# **유클리디안 노름** 또는 **$\ell_2$ 노름**으로 불린다.

# $$\text{RMSE}(\mathbf X, h) = \sqrt{\frac 1 m \sum_{i=1}^{m} (h(\mathbf x^{(i)}) - y^{(i)})^2}$$

# 위 수식에 사용된 기호의 의미는 다음과 같다.
# 
# * $\mathbf X$: 훈련셋 전체 샘플들의 특성값들로 구성된 행렬, 레이블(타깃) 제외.
# * $m$: $\mathbf X$의 행의 수. 즉, 훈련셋 크기.
# * $\mathbf x^{(i)}$: $i$ 번째 샘플의 특성값 벡터. 레이블(타깃) 제외.
# * $y^{(i)}$: $i$ 번째 샘플의 레이블(타깃)
# * $h$: 예측 함수
# * $h(\mathbf x^{(i)})$: $i$번째 샘플에 대한 예측 값. $\hat y^{(i)}$ 로 표기되기도 함.

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

# **평균 절대 오차**<font size="2">Mean Absolute Error</font>(MAE)는
# **맨해튼 노름** 또는 **$\ell_1$ 노름**으로도 불리며
# 예측값과 타깃 사이의 오차의 평균값이다.
# 
# $$\text{MAE}(\mathbf X, h) = \frac 1 m \sum_{i=1}^{m} \mid h(\mathbf x^{(i)}) - y^{(i)} \mid$$
# 
# 훈련셋에 이상치가 많이 포함된 경우 주로 사용되지만,
# 그렇지 않다면 일반적으로 RMSE가 선호된다.

# ## 데이터 다운로드와 적재

# 캘리포니아 주택가격 데이터셋은 매우 유명하여 많은 공개 저장소에서 다운로드할 수 있다.
# 여기서는 깃허브 리포지토리에 압축파일로 저장한 파일을 다운로드해서 사용하며
# `housing` 변수가 가리키도록 적재되었다고 가정한다.
# 
# - `housing` 변수: 캘리포티아 주택 가격 데이터를 담은 데이터 프레임 할당
# 

# ### 데이터셋 훑어보기

# pandas의 데이터프레임으로 데이터셋을 적재하여 기본적인 데이터 구조를 훑어볼 수 있다.

# **`head()` 메서드 활용**
# 
# 데이터프레임 객체의 처음 5개 샘플을 보여준다.
# 
# ```python
# housing.head()
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-05.png" width="600"></div>

# **`info()` 메서드 활용**
# 
# 데이터셋의 정보를 요약해서 보여준다.
# 
# * 구역 수: 20,640개. 한 구역의 인구는 600에서 3,000명 사이.
# * 구역별로 경도, 위도, 중간 주택 연도, 해안 근접도 등 총 10개의 조사 항목
#     * '해안 근접도'는 범주형 특성이고 나머지는 수치형 특성.
# * '방의 총 개수'의 경우 207개의 null 값, 즉 결측치 존재.
# 
# ```python
# housing.info()
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-05a.png" width="350"></div>

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
# 
# ```python
# housing.describe()
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/housing-describe.png"></div>

# **수치형 특성별 히스토그램**
# 
# `hist()` 메서드는 수치형 특성별 히스토그램을 그린다.
# 히스토그램을 통해 각 특성별 데이터셋의 다양한 정보를 확인할 수 있다.
# 
# - 각 특성마다 사용되는 단위와 스케일(척도)가 다르다.
# - 일부 특성은 한쪽으로 치우쳐저 있다.
# - 일부 특성은 값을 제한한 것으로 보인다.
# 
# ```python
# housing.hist(bins=50, figsize=(12, 8))
# ```

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/feature-histogram.png" width="600px"></div>

# ### 훈련셋과 테스트셋

# 모델 학습 시작 이전에 준비된 데이터셋을 **훈련셋**과 **테스트셋**으로 구분해야 한다.
# 테스트셋은 훈련 과정중에 전혀 사용되지 않으며 보통 전체 데이터셋의 20% 정도 이하로
# 선택하며, 전체 데이터셋의 크기에 따라 테스트셋의 크기가 너무 크지 않게 
# 비율을 적절히 조절한다.
# 
# 테스트셋에 대한 정보는 절대로 모델 훈련에 이용하지 않는다.
# 만약 이용하게 되면 미래에 실전에서 사용되는 데이터를 미리 안다고 가정하고 모델을 훈련시키는
# 것과 동일하게 되어 매우 잘못된 모델을 훈련시킬 위험을 키우게 된다.
# 
# 데이터셋을 훈련셋과 데이터셋으로 구분할 때 보통 계층 샘플링을 사용한다.

# **계층 샘플링**
# 
# **계층 샘플링**<font size="2">stratified sampling</font>은 각 계층별로 적절한 샘플을 추측하는 기법이다. 
# 이유는 계층별로 충분한 크기의 샘플이 포함되도록 지정해야 학습 과정에서 편향이 발생하지 않는다.
# 예를 들어, 특정 소득 구간에 포함된 샘플이 과하게 적거나 많으면 해당 계층의 중요도가 
# 과소 혹은 과대 평가될 수 있다.
# 
# 캘리포니아 데이터셋의 중간 소득을 대상으로하는 히스토그램을 보면
# 대부분 구역의 중간 소득이 1.5~6.0, 즉 15,000 달러에서 60,000 달러 사이인 것을 알 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-08.png" width="400"></div>

# 소득 구간을 아래 숫자들을 기준으로 5개로 구분한 다음에 계층 샘플링을 이용하여
# 훈련셋과 테스트셋을 구분할 수 있다.
# 
# | 구간 | 범위 |
# | :---: | :--- |
# | 1 | 0 ~ 1.5 |
# | 2 | 1.5 ~ 3.0 |
# | 3 | 3.0 ~ 4.5 |
# | 4 | 4.5 ~ 6.0 |
# | 5 | 6.0 ~  |

# 5 개의 구간으로 구분한 결과는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-08a.png" width="400"></div>

# 무작위 추출 방식과는 달리 계층별 샘플의 비율을 거의 동일하게 유지함을 확인할 수 있다.

# | 소득 구간 | 전체(%) | 계층 샘플링(%) | 무작위 샘플링(%) | 계층 샘플링 오류율 | 무작위 샘플링 오류율(%) |
# | :---: | ---: | ---: | ---: | ---: | ---: |
# | 1 | 3.98 | 4.00 | 4.24 | 0.36 | 6.45 |
# | 2 | 31.88 | 31.88 | 30.74 | -0.02 | -3.59 |
# | 3 | 35.06 | 35.05 | 34.52 | -0.01 | -1.53 |
# | 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
# | 5	| 11.44 | 11.43 | 12.09 | -0.08 | 5.63 |

# ## 데이터 탐색과 시각화

# 테스트셋을 제외한 훈련셋에 대해서 시각화를 이용하여 데이터셋을 탐색한다.

# ### 지리적 데이터 시각화

# 경도와 위도 정보를 이용하여 구역을 산포도로 나타내면 인구의 밀집 정도를 확인할 수 있다. 
# 예를 들어, 샌프란시스코의 Bay Area, LA, 샌디에고 등 유명 대도시의 특정 구역이 높은 인구 밀도를 갖는다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-09.png" width="500"></div>

# 인구 밀도가 높은 유명 대도시의 특정 구역에 위치한
# 주택 가격이 높다는 일반적인 사실 또한 산포도록 확인할 수 있다.
# 산포도를 그릴 때 해당 구역의 중간 주택 가격을 색상으로, 
# 인구밀도는 원의 크기로 활용한 결과는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-11.png" width="500"></div>

# ### 상관관계 조사

# 중간 주택 가격 특성과 다른 특성 사이의 선형 상관관계를 나타내는 상관계수는 다음과 같다.

# ```python
# median_house_value    1.000000
# median_income         0.688380
# rooms_per_house       0.143663
# total_rooms           0.137455
# housing_median_age    0.102175
# households            0.071426
# total_bedrooms        0.054635
# population           -0.020153
# people_per_house     -0.038224
# longitude            -0.050859
# latitude             -0.139584
# bedrooms_ratio       -0.256397
# ```

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
# 상관계수가 0이라는 것은 선형 상관관계가 없다는 의미이지 서로 아무런 상관관계가 없다는 말이 아니다.
# 또한 선형계수가 1이라 하더라도 두 특성이 1대 1로 의존한다는 의미도 아님을
# 아래 그림이 잘 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-14.png" width="400"></div>
# 
# <그림 출처: [위키백과](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)>
# :::

# 중간 주택 가격과 중간 소득의 상관계수가 0.68로 가장 높다.
# 이는 중간 소득이 올라가면 중간 주택 가격도 상승하는 경향이 있음을 의미한다.
# 하지만 아래 산점도의 점들이 너무 넓게 퍼져 있어서 완벽한 선형관계와는 거리가 멀다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-13.png" width="400"></div>

# 위 산점도를 볼 때 다음 사항들에 주의해야 한다.
# 
# * 50만 달러에서 보이는 수평선은 가격을 제한한 결과로 보여진다.
# * 45만, 35만, 28만, 그 아래 정도에서도 수평선이 존재하는데 이유는 알려지지 않았다.
# * 이처럼 이상한 성질을 모델이 형태를 학습하지 못하도록 해당 구역을 제거하는 것이 
#     일반적으로 좋다. 하지만 여기서는 그대로 두고 사용한다.

# 경우에 따라 기존의 특성을 조합해서 새로운 특성을 활용할 수도 있다.
# 예를 들어 구역별 방의 총 개수와 침실의 총 개수 대신 아래 특성이 보다 유용해 보인다.
# 
# * 가구당 방 개수(`rooms_for_house`)
# * 방 하나당 침실 개수(`bedrooms_ratio`)
# * 가구당 인원(`people_perhouse`)
# 
# 실제로 세 특성을 새로 추가한 다음에 상관계수를 확인하면 
# 방 하나당 침실 개수와 중간 주택 가격 사이의 선형 상관관계가
# 중간 소득을 제외한 기존의 다른 특성들에 비해 높게 나타난다.

# ## 데이터 준비

# 머신러닝 모델 훈련에 사용되는 알고리즘을 이용하려면
# 적재된 데이터셋을 데이터 정제와 전처리 과정을 수행해서
# 바로 모델 훈련에 사용될 수 있도록 해야 한다. 
# 또한 모든 과정을 자동화할 수 있어야 한다.
# 
# 정제와 전처리 모든 과정을 
# __파이프라인__<font size="2">pipeline</font>으로
# 자동화해서 언제든지 재활용하는 방식을 상세히 설명한다.

# **입력 데이터셋과 타깃 데이터셋**
# 
# 계층 샘플링으로 얻어진  훈련셋 `strat_train_set` 을 
# 다시 입력 데이터셋 과 타깃 데이터셋으로 구분한다. 
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

# ### 데이터 정제와 전처리

# 데이터 정제는 결측치 처리, 이상치 및 노이즈 데이터 제거 등을 의미한다.
# 캘리포니아 주택 가격 데이터셋은 구역별 방 총 개수(`total_rooms`) 특성에서 
# 결측치가 일부 포함되어 있지만 이상치 또는 노이즈 데이터는 없다.

# 전처리는 수치형 특성과 범주형 특성을 구분하여 수행한다. 
# 
# * 수치형 특성에 대한 전처리
#     * 특성 스케일링
#     * 특성 조합
# 
# * 범주형 특성 전처리 과정
#     * 원-핫-인코딩

# 데이터 정제와 전처리의 모든 과정은 데이터셋에 포함된 샘플을 한꺼번에 변환한다.
# 따라서 모든 변환 과정을 자동화는
# __파이프라인__<font size="2">pipeline</font> 기법을 활용할 수 있어야 한다.

# **사이킷런 API 활용**

# 사이킷런<font size="2">Scikit-Learn</font>이 제공하는
# 모든 클래스는 간단하게 합성할 수 있다.
# 이점을 이해하려면 먼저 사이킷런이 제공하는 API의 유형을 구분해야 한다.
# 사이킷런의 API는 크게 세 종류의 클래스로 나뉜다.

# * 추정기<font size="2">estimator</font>: `fit()` 메서드를 제공하는 클래스
#     * 주어진 데이터로부터 필요한 정보인 파라미터<font size='2'>parameter</font> 계산. 
#     * 계산된 파라미터를 클래스 내부의 속성<font size='2'>attribute</font>으로 저장
#     * 반환값: 계산된 파라미터를 속성으로 갖는 동일한 클래스

# * 변환기<font size="2">transformer</font>
#     * `fit()` 가 계산한 값을 이용하여 데이터셋을 변환하는 `transform()` 메서드 지원.
#     * `fit()` 메서드와 `transform()` 메서드를 연속해서 호출하는 `fit_transform()` 메서드 지원.

# * 예측기<font size="2">predictor</font>
#     * `fit()` 가 계산한 값을 이용하여 예측에 활용하는 `predict()` 메서드 지원.
#     * `predict()` 메서드가 예측한 값의 성능을 측정하는 `score()` 메서드 지원.
#     * 일부 예측기는 예측값의 신뢰도를 평가하는 기능도 제공

# :::{admonition} 변환기 사용법
# :class: warning
# 
# `fit()` 과 `fit_transform()` 두 메서드는 훈련셋에 대해서만 적용한다.
# 반면에 테스트셋, 검증셋, 새로운 데이터 등에 대해서는 `transform()` 메서드만 적용한다. 
# :::

# 사이킷런의 모든 클래스는 적절한 
# **하이퍼파라미터**<font size="2">hyperparameter</font>로 초기화되어 있으며
# 데이터 변환 및 값 예측에 필요한 모든 파라미터를 효율적으로 관리한다.

# :::{admonition} 하이퍼파라미터 vs. 파라미터
# :class: info
# 
# 사이킷런 클래스의 하이퍼파라미터는 해당 클래스의 객체를 생성할 때 사용되는 값을 가리킨다.
# 즉, API 객체를 생성하기 위해 해당 API 클래스의 생성자인 
# `__init__()` 메서드를 호출할 때 사용되는 인자를 가리킨다.
# 
# 반면에 파라미터는 `fit()` 메서드가 데이터를 이용하여 계산하는 값을 가리킨다.
# 추정기, 변환기, 예측기는 각각의 역할에 맞는 파라미터를 계산한다.
# :::

# ### 데이터 정제

# 입력 데이터셋의 `total_bedrooms` 특성에 207개 구역이 null 값으로 채워져 있다.
# 즉, 일부 구역에 대한 방의 총 개수 정보가 누락되었다.

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

# 해안 근접도(`ocean_proximity`)는 수가 아닌 5 개의 범주를 나타내는 텍스트를 값으로 사용한다.
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
# 원-핫 인코딩은 수치화된 범주들 사이의 크기 비교를 피하기 위해
# 더미<font size="2">dummy</font> 특성을 활용한다.
# 
# 원-핫 인코딩을 적용하면 해안 근접도 특성을 삭제하고 대신 다섯 개의 범주 전부를 
# 새로운 특성으로 추가한다.
# 또한 다섯 개의 특성에 사용되는 값은 다음 방식으로 지정된다.
# 
# * 해당 카테고리의 특성값: 1
# * 나머지 카테고리의 특성값: 0
# 
# 예를 들어, `INLAND`를 해안 근접도 특성값으로 갖던 샘플은 다음 모양의 특성값을 갖게 된다.
# 
# ```python
# [0, 1, 0, 0, 0]
# ```

# 사이킷런의 `OneHotEncoder` 변환기가 원-핫-인코딩을 지원하며
# 해안 근접도를 변환한 결과는 아래 모양을 갖는다.

# ```python
# array([[0., 0., 0., 1., 0.],
#        [1., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0.],
#        ...,
#        [0., 0., 0., 0., 1.],
#        [1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 1.]])
# ```

# ### 특성 스케일링

# 머신러닝 알고리즘은 입력 데이터셋의 특성값들의 
# **스케일**<font size="2">scale</font>(척도)이 다르면 제대로 작동하지 않는다.
# 따라서 모든 특성의 척도를 통일하는 **스케일링**<font size="2">scaling</font>이 요구된다.

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
# 예를 들어 이상치가 매우 크면 분모가 분자에 비해 훨씬 크게 되어 변환된 값이 0 근처에 몰리게 된다.
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
# 변환된 데이터셋은 평균값은 0, 표준편차는 1인 분포를 따르며, 이상치에 상대적으로 덜 영향을 받는다.
# 여기서는 사이킷런의 `StandardScaler` 변환기를 이용하여 표준화를 적용한다.

# :::{admonition} 타깃 데이터셋 전처리
# :class: info
# 
# 데이터 준비는 기본적으로 입력 데이터셋만을 대상으로 **정제**<font size="2">cleaning</font>와 
# **전처리**<font size="2">preprocessing</font> 단계로 실행된다. 
# 타깃 데이터셋은 결측치가 없는 경우라면 일반적으로 정제와 전처리 대상이 아니지만
# 경우에 따라 변환이 요구될 수 있다.
# 예를 들어, 타깃 데이터셋의 두터운 꼬리 분포를 따르는 경우
# 로그 함수를 적용하여 데이터의 분포가 보다 균형잡히도록 하는 것이 권장된다.
# 하지만 이런 경우 예측값을 계산할 때 원래의 척도로 되돌려야 하며
# 이를 위해 대부분의 사이킷런 변환기가 지원하는 `inverse_transorm()` 메서드를 활용할 수 있다.
# :::

# ### 사용자 정의 변환기

# 데이터 준비 과정에서 경우에 따라 사용자가 직접 변환기를 구현해야할 필요가 있다.

# #### `FunctionTransformer` 변환기

# `fit()` 메서드를 먼저 사용하지 않고 `transform()` 메서드를 바로 적용해도 되는
# 변환기는 `FunctionTransformer` 객체를 활용하여 생성할 수 있다.

# **로그 함수 적용 변환기**
# 
# 데이터셋이 두터운 꼬리 분포를 따르는 경우, 
# 즉 히스토그램이 지나치게 한쪽으로 편향된 경우
# 스케일링을 적용하기 전에 먼저
# 로그 함수를 적용하여 어느 정도 좌우 균형이 잡힌 분포로 변환하는 게 좋다. 
# 아래 그림은 인구에 로그함수를 적용할 때 분포가 보다 균형잡히는 것을 잘 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-log_app.jpg" width="600"></div>

# 두터운 꼬리 분포를 갖는 데이터셋에 로그 함수를 적용하고자 하면 아래 변환기를 사용하면 된다.
# 
# ```python
# FunctionTransformer(np.log, inverse_func=np.exp)
# ```

# **비율 계산 변환기**
# 
# 두 개의 특성 사이의 비율을 계산하여 새로운 특성을 생성하는 변환기 또한 
# `FunctionTransformer`를 활용할 수 있다.
# 
# ```python
# FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
# ```

# 비율 계산 변환기를 이용하여 아래 특성을 새롭게 생성할 수 있다.
# 
# - 가구당 방 개수(rooms for household)
# - 방 하나당 침실 개수(bedrooms for room)
# - 가구당 인원(population per household)

# #### 사용자 정의 변환 클래스

# `SimpleImputer` 변환기의 경우처럼 
# 먼저 `fit()` 메서드를 이용하여 평균값, 중앙값 등을 확인한 다음에
# `transform()` 메서드를 적용할 수 있는 변환기는 클래스를 직접 선언해야 한다. 
# 이때 사이킷런의 다른 변환기와 호환이 되도록 하기 위해
# `fit()` 과 `transform()` 등 다양한 메서드를 모두 구현해야 한다. 
# 
# 예를 들어, 캘리포니아 주 2만 여개의 구역을 서로 가깝게 위치한 구역들의 군집으로 구분하는 변환기는
# 다음과 같다. 단, 아래 코드를 지금 이해할 필요는 없다.

# ```python
# class ClusterSimilarity(BaseEstimator, TransformerMixin):
#     def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
#         self.n_clusters = n_clusters
#         self.gamma = gamma
#         self.random_state = random_state
# 
#     def fit(self, X, y=None, sample_weight=None):
#         self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
#         self.kmeans_.fit(X, sample_weight=sample_weight)
#         return self  # 항상 self 반환
# 
#     def transform(self, X):
#         return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
#     
#     def get_feature_names_out(self, names=None):
#         return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
# ```

# :::{admonition} `KMeans` 모델과 `rbf_kernel()` 함수
# :class: info
# 
# 위 클래스는 `KMeans` 모델과 `rbf_kernel()` 함수를 활용한다.
# 
# **`KMeans` 모델**
# 
# {numref}`%s장 <ch:unsupervisedLearning>` 비지도 학습에서 다룰 군집 알고리즘 모델이다.
# 
# **`rbf_kernel()` 함수**
# 
# 다음 가우시안 RBF 함수를 활용한다.
# $\mathbf{p}$ 는 특정 지점을 가리키며,
# $\mathbf{p}$ 에서 조금만 멀어져도 함숫값이 급격히 작아진다. 
# 
# 
# $$
# \phi(\mathbf{x},\mathbf{p}) = \exp \left( -\gamma \|\mathbf{x} - \mathbf{p} \|^2 \right)
# $$
# 
# 예를 들어 아래 이미지는 중간 주택 년수가 35년에서 멀어질 수록 
# 함숫값이 급격히 0에 가까워지는 것을 보여준다.
# 하이퍼파라미터인 **감마**($\gamma$, gamma)는 얼마나 빠르게 감소하도록 하는가를 결정한다.
# 즉, 감마 값이 클 수록 보다 좁은 종 모양의 그래프가 그려진다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-rbf_kernel.jpg" width="400"></div>
# :::

# `ClusterSimilarity` 변환기를 이용하여 얻어진 군집 특성을 이용하면
# 아래 그림과 같은 결과를 얻을 수 있다.
# 
# - 모든 구역을 10개의 군집으로 나눈다.
# -  &#9587; 는 각 군집의 중심 구역을 나타낸다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/homl02-cluster.jpg" width="550"></div>

# ### 변환 파이프라인

# 모든 전처리 단계가 정확한 순서대로 진행되어야 한다.
# 이를 위해 사이킷런의 `Pipeline` 클래스를 이용하여 여러 변환기를 순서대로 
# 실행하는 파이프라인 변환기를 활용한다.

# **`Pipeline` 클래스** 
# 
# 예를 들어, 수치형 특성을 대상으로 결측치를 중앙값으로 채우는 정제와
# 표준화 스케일링을 연속적으로 실행하는 파이프라인은 다음과 같이 정의한다.
# 
# ```python
# num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
#                          ("standardize", StandardScaler())])
# ```
# 
# * `Pipeline` 객체를 생성할 때 사용되는 인자는 이름과 추정기로 이루어진 쌍들의 리스트이다.
# 
# * 마지막 추정기를 제외한 나머지 추정기는 모두 변환기다.
#     즉, 마지막 추정기는 `fit()` 메서드만 지원해도 되지만
#     나머지는 `fit_transform()` 메서드가 지원되는 변환기어야 한다.
#     
# * `num_pipeline.fit()` 를 호출하면 
#     마지막 변환기 까지는 `fit_transform()` 메소드가 연속적으로 호출되고
#     마지막 변환기의 `fit()` 메서드 최종 호출된다.
#     
# * 파이프라인으로 정의된 추정기의 유형은 마지막 추정기의 유형과 동일하다.
#     따라서 `num_pipeline` 은 변환기다.

# **`make_pipeline()` 함수**
# 
# 파이프라인에 포함되는 변환기의 이름이 중요하지 않다면 `make_pipeline()` 함수를 이용하여
# `Pipeline` 객체를 생성할 수 있다. 이름은 자동으로 지정된다.
# 
# 위 파이프라인과 동일한 파이프라인 객체를 다음과 같이 생성할 수 있다.
# 
# ```python
# make_pipeline(SimpleImputer(strategy="median"), 
#               StandardScaler())
# ```

# **`ColumnTransformer` 클래스**
# 
# `ColumnTransformer` 클래스는 특성별로 전처리를 지정할 수 있다.
# 이 기능을 이용하여 수치형 특성과 범주형 특성을 구분해서 
# 전처리하는 통합 파이프라인을 다음과 같이 구성할 수 있다.
# 
# * 수치형 특성: `num_pipeline` 변환기
# * 범주형 특성: `OneHotEncoder` 변환기
# 
# ```python
# num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
#                "total_bedrooms", "population", "households", "median_income"]
# cat_attribs = ["ocean_proximity"]
# 
# cat_pipeline = make_pipeline(
#     SimpleImputer(strategy="most_frequent"),
#     OneHotEncoder(handle_unknown="ignore"))
# 
# preprocessing = ColumnTransformer([
#     ("num", num_pipeline, num_attribs),
#     ("cat", cat_pipeline, cat_attribs),
# ])
# ```

# **`make_column_selector()` 함수**
# 
# 파이프라인에 포함되는 각 변환기를 적용할 특성을 일일이 나열하는 일이 어려울 수 있다.
# 이때 지정된 자료형을 사용하는 특성들만을 뽑아주는 `make_column_selector()` 함수를 
# 유용하게 활용할 수 있다.
# 
# 위 `preprocessing` 변환기를 아래와 같이 정의할 수 있다.
# 
# ```python
# preprocessing = ColumnTransformer([
#     ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
#     ("cat", cat_pipeline, make_column_selector(dtype_include=object)
# ])
# ```

# **`make_column_transformer()` 함수**
# 
# `ColumnTransformer` 파이프라인에 포함되는 변환기의 이름이 중요하지 않다면 
# `make_column_transformer()` 함수를 이용할 수 있으며,
# `make_pipeline()` 함수와 유사하게 작동한다.
# 
# 위 `preprocessing` 변환기를 아래와 같이 정의할 수 있다.
# 
# ```python
# preprocessing = make_column_transformer(
#     (num_pipeline, make_column_selector(dtype_include=np.number)),
#     (cat_pipeline, make_column_selector(dtype_include=object)),
# )
# ```

# ### 캘리포니아 데이터셋 변환 파이프라인

# 다음 변환기를 모아 캘리포니아 데이터셋 전용 변환 파이프라인을 생성할 수 있다.

# **(1) 비율 변환기**
# 
# 가구당 방 개수, 방 하나당 침실 개수, 가구당 인원 등 
# 비율을 사용하는 특성을 새로 추가할 때 사용되는 변화기를 생성하는 함수를 정의한다.

# ```python
# def column_ratio(X):
#     return X[:, [0]] / X[:, [1]]
# 
# def ratio_pipeline(name=None):
#     return make_pipeline(
#         SimpleImputer(strategy="median"),
#         FunctionTransformer(column_ratio,
#                             feature_names_out=[name]),
#         StandardScaler())
# ```

# **(2) 로그 변환기**
# 
# 데이터 분포가 두터운 꼬리를 갖는 특성을 대상으로 로그 함수를 적용하는 변환기를 지정한다.

# ```python
# log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
#                              FunctionTransformer(np.log),
#                              StandardScaler())
# ```

# **(3) 군집 변환기**
# 
# 구역의 위도와 경도를 이용하여 구역들의 군집 정보를 새로운 특성으로 추가하는 변환기를 지정한다.

# ```python
# cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
# ```

# **(4) 기타**
# 
# 특별한 변환이 필요 없는 경우에도 기본적으로 결측치 문제 해결과 스케일을 조정하는 변환기를 사용한다.

# ```python
# default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
#                                      StandardScaler())
# ```

# 앞서 언급된 모든 변환기를 특성별로 알아서 처리하는 변환기는 다음과 같다.
# `remainder=default_num_pipeline`: 언급되지 않은 특성을 처리하는 변환기를 지정한다.
# 삭제를 의미하는 `drop` 이 기본값이며 이외에 `passthrough` 는 변환하지 않는 것을 의미한다.

# ```python
# preprocessing = ColumnTransformer([
#         ("bedrooms_ratio", ratio_pipeline("bedrooms_ratio"),                   # 방당 침실 수
#                            ["total_bedrooms", "total_rooms"]),
#         ("rooms_per_house", ratio_pipeline("rooms_per_house"),                 # 가구당 방 수
#                             ["total_rooms", "households"]),
#         ("people_per_house", ratio_pipeline("people_per_house"),               # 가구당 인원
#                              ["population", "households"]),
#         ("log", log_pipeline, ["total_bedrooms", "total_rooms",                # 로그 변환
#                                "population", "households", "median_income"]),
#         ("geo", cluster_simil, ["latitude", "longitude"]),                     # 구역별 군집 정보
#         ("cat", cat_pipeline, make_column_selector(dtype_include=object)),     # 범주형 특성 전처리
#     ],
#     remainder=default_num_pipeline)                                            # 중간 주택 년수(housing_median_age) 대상
# ```

# ## 모델 선택과 훈련

# 훈련셋 준비가 완료된 상황에서 모델을 선택하고 훈련시키는 일이 남아 있다.
# 
# 사이킷런이 제공하는 예측기 모델을 사용하면 훈련은 기본적으로 간단하게 진행된다.
# 여기서는 사이킷런이 제공하는 다양한 모델의 사용법과 차이점을 간단하게 살펴본다.
# 각 모델의 자세한 특징과 상세 설명은 앞으로 차차 이루어질 것이다.

# :::{admonition} 전처리 포함 파이프라인 모델
# :class: info
# 
# 소개되는 모든 모델은 앞서 설명한 전처리 과정과 함께 하나의 파이프라인으로 묶여서 정의된다.
# 이는 테스트셋과 미래의 모든 입력 데이터셋에 대해서도 전처리를 별도로 신경쓸 필요가 없게 해준다.
# :::

# ### 훈련셋 대상 훈련 및 평가

# **선형 회귀 모델 ({numref}`%s장 <ch:trainingModels>`)**

# * 훈련 및 예측
# 
#     ```python
#     lin_reg = make_pipeline(preprocessing, LinearRegression())
#     lin_reg.fit(housing, housing_labels)
#     lin_reg.predict(housing)
#     ```
# 
# - RMSE(평균 제곱근 오차)
# 
#     ```python
#     lin_rmse = mean_squared_error(housing_labels, housing_predictions,
#                                   squared=False)
#     ```

# - 훈련 결과
#     - RMSE(`lin_rmse`)가 68687.89 정도로 별로 좋지 않다.
#     - 훈련된 모델이 훈련셋에 __과소적합__ 되었다. 
#     - 보다 좋은 특성을 찾거나 더 강력한 모델을 적용해야 한다. 

# **결정트리 회귀 모델 ({numref}`%s장 <ch:decisionTrees>`)**

# 결정트리 회귀 모델은 데이터에서 복잡한 비선형 관계를 학습할 때 사용한다. 
# 
# * 훈련 및 예측
# 
# ```python
# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# tree_reg.fit(housing, housing_labels)
# housing_predictions = tree_reg.predict(housing)
# 
# tree_rmse = mean_squared_error(housing_labels, housing_predictions,
#                               squared=False)
# ```

# - 훈련 결과
#     - RMSE(`tree_rmse`)가 0으로 완벽해 보인다.
#     - 모델이 훈련셋에 심각하게 __과대적합__ 되었음을 의미한다.
#     - 실전 상황에서 RMSE가 0이 되는 것은 불가능하다.
#     - 테스트셋에 대한 RMSE는 매우 높게 나온다.

# ### 교차 검증

# __교차 검증__<font size="2">cross validation</font>을 이용하여 
# 훈련중인 모델의 성능을 평가할 수 있다.

# **k-겹 교차 검증**
# 
# * 훈련셋을 __폴드__(fold)라 불리는 k-개의 부분 집합으로 무작위로 분할한다.
# * 모델을 총 k 번 훈련한다.
#     * 매 훈련마나다 하나의 폴드를 선택하여 검증 데이터셋 지정.
#     * 나머지 (k-1) 개의 폴드를 대상으로 훈련
#     * 매 훈련이 끝날 때마다 선택된 검증 데이터셋을 이용하여 모델 평가
#     * 매번 다른 폴드 활용
# * 최종평가는 k-번 평가 결과의 평균값을 활용한다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch02/cross-val10a.png" width="550"></div>

# **사이킷런의 `cross_val_score()` 함수**
# 
# `cross_val_score()` 함수는 k-겹 교차 검증 과정에서
# 훈련중인 모델의 성능을 측정한다.
# 
# 측정값은 높을 수록 좋은 성능으로 평가되기에 회귀 모델의 경우
# 일반적으로 RMSE의 음숫값을 사용한다.
# 이를 위해 `scoring="neg_mean_squared_error"` 키워드 인자를 사용한다.
# 
# 아래 코드는 10 개의 폴드를 사용(`cv=10`)하여 결정트리 회귀 모델에 대한 교차 검증을 진행하고 평가한다. 
# 
# ```python
# tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
#                               scoring="neg_root_mean_squared_error", cv=10)
# ```

# :::{admonition} `scoring` 키워드 인자
# :class: info
# 
# 교차 검증에 사용되는 모델의 종류에 따라 다양한 방식으로 모델의 성능을 측정할 수 있으며
# `scoring` 키워드 인자를 이용하여 지정한다. 
# 현재 사용 가능한 옵션값은 [사이킷런의 Metrics and Scoring 문서](https://scikit-learn.org/stable/modules/model_evaluation.html)에서
# 확인할 수 있다.
# :::

# **랜덤 포레스트 회귀 모델 ({numref}`%s장 <ch:ensemble>`)**
# 
# **랜덤 포레스트**<font size="2">random forest</font> 회귀 모델은 
# 여러 개의 결정트리를 동시에 훈련시킨 후 
# 각 모델의 예측값의 평균값 등을 이용하는 모델이다.
# 각 모델은 교차 검증처럼 서로 다른 훈련셋을 대상으로 학습한다.
# 
# 사이킷런의 `RandomForestRegressor` 모델은 기본값으로 100개의 결정트리를 동시에 훈련시킨다.
# 
# ```python
# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(n_estimators=100, random_state=42))
# ```
# 
# 래덤 포레스트 모델에 대한 교차 검증을 적용하면 폴드 수에 비례하여 훈련 시간이 더 오래 걸린다.
# 
# ```python
# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error", cv=10)
# ```

# ## 모델 튜닝

# 지금까지 살펴 본 모델 중에서 랜덤 포레스트 회귀 모델의 성능이 가장 좋았다.
# 이렇게 가능성이 높은 모델을 찾은 다음엔 모델의 세부 설정(하이퍼파라미터)을 조정하거나
# 성능이 좋은 모델 여러 개를 이용하여 모델의 성능을 최대한 끌어올릴 수 있다.
# 
# 모델 튜닝은 보통 다음 두 가지 방식을 사용한다.
# 
# * 그리드 탐색
# * 랜덤 탐색

# ### 그리드 탐색

# 지정된 하이퍼파라미터의 모든 조합에 대해 교차 검증을 진행하여 최선의 하이퍼파라미터 조합을 찾는다. 

# **`GridSearchCV` 클래스**
# 
# 랜덤 포레스트 모델을 대상으로 그리드 탐색을 다음과 같이 실행하면
# 총 (3x3 + 2x3 = 15) 가지의 모델의 성능을 확인한다. 
# 또한 3-겹 교차 검증(`cv=3`)을 진행하기에 모델 훈련을 총 45(=15x3)번 진행한다.
# 
# ```python
# full_pipeline = Pipeline([
#     ("preprocessing", preprocessing),
#     ("random_forest", RandomForestRegressor(random_state=42)),
# ])
# 
# param_grid = [
#     {'preprocessing__geo__n_clusters': [5, 8, 10],
#      'random_forest__max_features': [4, 6, 8]},
#     {'preprocessing__geo__n_clusters': [10, 15],
#      'random_forest__max_features': [6, 8, 10]},
# ]
# 
# grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
#                            scoring='neg_root_mean_squared_error')
# 
# grid_search.fit(housing, housing_labels)
# ```

# ### 랜덤 탐색

# 그리드 탐색은 적은 수의 조합을 실험해볼 때만 유용하다.
# 반면에 하이퍼파라미터의 탐색 공간이 크면 랜덤 탐색이 보다 유용하다.
# 랜덤 탐색은 하이퍼라라미터 조합을 임의로 지정된 횟수만큼 진행한다.

# **`RandomizedSearchCV` 클래스**
# 
# 아래 코드는 다음 두 하이퍼파라미터를 대상으로 
# 10번(`n_iter=10`) 지정된 구간 내에서 무작위 선택을 진행한다. 
# 
# - `preprocessing__geo__n_clusters`
# - `random_forest__max_features`
# 
# 또한 3-겹 교차검증(`cv=3`)을 진행하기에 모델 훈련을 총 30(=10x30)번 진행한다.
# 
# ```python
# param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
#                   'random_forest__max_features': randint(low=2, high=20)}
# 
# rnd_search = RandomizedSearchCV(
#     full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
#     scoring='neg_root_mean_squared_error', random_state=42)
# 
# rnd_search.fit(housing, housing_labels)
# ```

# ### 앙상블 기법

# 결정트리 모델 하나보다 랜덤 포레스트처럼 여러 모델을 활용하는 모델이
# 일반적으로 보다 좋은 성능을 낸다.
# 이처럼 좋은 성능을 내는 여러 모델을 **함께**<font size="2">ensemble</font> 
# 학습시킨 후 평균값을 사용하면 보다 좋은 성능을 내는 모델을 얻게 된다. 
# 앙상블 기법에 대해서는 {numref}`%s장 <ch:ensemble>`에서 자세히 다룬다.

# ### 훈련된 최선의 모델 활용

# 그리드 탐색 또는 랜덤 탐색을 통해 얻어진 최선의 모델을 분석해서 문제에 대한 통찰을 얻을 수 있다.
# 
# 예를 들어, 최선의 랜덤 포레스트 모델로부터 타깃 예측에 사용된 특성들의 상대적 중요도를 확인하여
# 중요하지 않은 특성을 제외할 수 있다.
# 
# 캘리포니아 주택 가격 예측 모델의 경우 랜덤 탐색을 통해 찾아낸 최선의 모델에서 
# `feature_importances_`를 확인하면 다음 정보를 얻는다.
# 
# - `log__median_income` 특성이 가장 중요하다.
# - 해안 근접도 특성 중에서 `INLAND` 특성만 중요하다.

# ```python
# final_model = rnd_search.best_estimator_                                 # 최선 모델
# feature_importances = final_model["random_forest"].feature_importances_  # 특성뱔 상대적 중요도
# 
# # 중요도 내림차순 정렬
# sorted(zip(feature_importances,
#            final_model["preprocessing"].get_feature_names_out()),
#            reverse=True)
# 
# [(0.18694559869103852, 'log__median_income'),
#  (0.0748194905715524, 'cat__ocean_proximity_INLAND'),
#  (0.06926417748515576, 'bedrooms_ratio__bedrooms_ratio'),
#  (0.05446998753775219, 'rooms_per_house__rooms_per_house'),
#  (0.05262301809680712, 'people_per_house__people_per_house'),
#  (0.03819415873915732, 'geo__Cluster 0 similarity'),
#  [...]
#  (0.00015061247730531558, 'cat__ocean_proximity_NEAR BAY'),
#  (7.301686597099842e-05, 'cat__ocean_proximity_ISLAND')]
# ```

# ## 최선 모델 저장 및 활용

# 완성된 모델은 항상 저장해두어야 한다.
# 업데이트된 모델이 적절하지 않은 경우 이전 모델로 되돌려야 할 수도 있기 때문이다.
# 모델의 저장과 불러오기는 `joblib` 모듈을 활용한다. 
# 
# - 저장하기
# 
#     ```python
#     joblib.dump(final_model, "my_california_housing_model.pkl")
#     ```
# - 불러오기
# 
#     ```python
#     final_model_reloaded = joblib.load("my_california_housing_model.pkl")
#     ```    

# ## 연습문제

# 참고: [(실습) 머신러닝 프로젝트 처음부터 끝까지 1부](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_end2end_ml_project_1.ipynb) 와
# [(실습) 머신러닝 프로젝트 처음부터 끝까지 2부](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_end2end_ml_project_2.ipynb)
