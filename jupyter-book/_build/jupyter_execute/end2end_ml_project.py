#!/usr/bin/env python
# coding: utf-8

# (ch:end2end)=
# # 머신러닝 프로젝트 처음부터 끝까지

# #### 감사의 글
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# ## 주요 내용
# 
# * 주택 가격을 예측하는 회귀 작업을 살펴보면서 선형 회귀, 결정 트리, 랜덤 포레스트 등 여러 알고리즘의 기본 사용법 소개
# 
# * 머신러닝 시스템 전체 훈련 과정 살펴보기
# 
# <div align="center"><img src="imgs/ch02/homl02-01d.png" width="600"></div>

# ## 실제 데이터로 작업하기

# * 유명한 공개 데이터 저장소
#     * [OpenML](https://www.openml.org/)
#     * [캐글(Kaggle) 데이터셋](http://www.kaggle.com/datasets)
#     * [페이퍼스 위드 코드](https://paperswithcode.com/)
#     * [UC 얼바인(UC Irvine) 대학교 머신러닝 저장소](http://archive.ics.uci.edu/ml)
#     * [아마존 AWS 데이터셋](https://registry.opendata.aws)
#     * [텐서플로우 데이터셋](https://www.tensorflow.org/datasets)

# * 메타 포털(공개 데이터 저장소가 나열)
#     * [데이터 포털(Data Portals)](http://dataportals.org)
#     * [오픈 데이터 모니터(Open Data Monitor)](http://opendatamonitor.eu)

# * 인기 있는 공개 데이터 저장소가 나열되어 있는 다른 페이지
#     * [위키백과 머신러닝 데이터셋 목록](https://goo.gl/SJHN2k)
#     * [Quora.com](https://homl.info/10)
#     * [데이터셋 서브레딧(subreddit)](http://www.reddit.com/r/datasets)

# ## 큰 그림 보기

# ### 주어진 데이터
# 
# * 1990년도 미국 캘리포니아 주의 20,640개 구역별 인구조사 데이터
# 
# * 특성 10개: 경도, 위도, 중간 주택 연도, 방의 총 개수, 침실 총 개수, 인구, 가구 수, 중간 소득, 중간 주택 가격, 해안 근접도
# 
# * 목표: 구역별 중간 주택 가격 예측 시스템(모델) 구현하기
# 
# * 미국 캘리포니아 지도

# <div align="center"><img src="imgs/ch02/LA-USA01.png" width="600"></div>

# ### 문제 정의

# * 지도 학습(supervised learning)
#     - 레이블: 구역별 중간 주택 가격
# 
# * 회귀(regression): 중간 주택 가격 예측
#   * 다중 회귀(multiple regression): 여러 특성을 활용한 예측
#   * 단변량 회귀(univariate regression): 구역마다 한 종류의 값만 예측
#       - 참고: 다변량 회귀(multivariate regression). ...
# 
# * 배치 학습(batch learning): 빠르게 변하는 데이터에 적응할 필요가 없으며, 데이터셋의 크기도 충분히 작음.

# ### 성능 측정 지표 선택

# 사용하는 모델에 따라 모델 성능 측정 기준(norm)을 다르게 선택한다. 
# 선형 회귀 모델의 경우 일반적으로 아래 두 기준 중 하나를 사용한다.
# 
# * 평균 제곱근 오차(RMSE)
# 
# * 평균 절대 오차(MAE)

# #### 평균 제곱근 오차(root mean square error, RMSE)
# 
# - 유클리디안 노름(Euclidean norm) 또는 $\ell_2$ 노름(norm)으로도 불림
# - 참고: 노름(norm)은 거리 측정 기준을 나타냄.

# $$\text{RMSE}(\mathbf X, h) = \sqrt{\frac 1 m \sum_{i=1}^{m} (h(\mathbf x^{(i)}) - y^{(i)})^2}$$

# * 기호 설명
#     * $\mathbf X$: 모델 성능 평가에 사용되는 데이터셋 전체 샘플들의 특성값들로 구성된 행렬, 레이블(타겟) 제외.
#     * $m$: $\mathbf X$의 행의 수. 즉, 훈련 데이터셋 크기.
#     * $\mathbf x^{(i)}$: $i$ 번째 샘플의 전체 특성값 벡터. 레이블(타겟) 제외.
#     * $y^{(i)}$: $i$ 번째 샘플의 레이블
#     * $h$: 예측 함수
#     * $\hat y^{(i)} = h(\mathbf x^{(i)})$: $i$번째 샘플에 대한 예측 값

# #### 평균 절대 오차(mean absolute error, MAE)
# 
# - MAE는 맨해튼 노름 또는 $\ell_1$ 노름으로도 불림
# 
# $$\text{MAE}(\mathbf X, h) = \frac 1 m \sum_{i=1}^{m} \mid h(\mathbf x^{(i)}) - y^{(i)} \mid$$
# 
# - 이상치가 많은 경우 활용
# 
# * $\ell_1$ 노름과 $\ell_2$ 노름을 일반해서 $\ell_n$ 노름을 정의할 수도 있음
# 
# * RMSE가 MAE보다 이상치에 더 민감하지만, 이상치가 많지 않을 경우 일반적으로 RMSE 사용

# ## 데이터 가져오기

# ### 데이터 다운로드
# 
# * 저자의 깃허브 저장소에 있는 압축파일 다운로드
# 
# * 압축파일을 풀어 csv 파일로 저장

# ### 데이터 구조 훑어보기

# #### 데이터셋 기본 정보 확인
# 
# * pandas의 데이터프레임 활용
#     * `head()`, `info()`, `describe()`, `hist()` 등을 사용하여 데이터 구조 훑어보기

# #### `head()` 메서드 활용 결과
# 
# <div align="center"><img src="imgs/ch02/homl02-05.png" width="600"></div>

# #### `info()` 메서드 활용 결과

# <div align="center"><img src="imgs/ch02/homl02-05a.png" width="450"></div>

# * 구역 수: 20,640개
# 
# * 구역별로 경도, 위도, 중간 주택 연도, 해안 근접도 등 총 10개의 조사 항목
#     * '해안 근접도'는 범주형 특성이고 나머지는 수치형 특성.
# 
# * '방의 총 개수'의 경우 누락된 데이터인 207개의 null 값 존재

# #### 범주형 특성 탐색
# 
# * '해안 근접도'는 5개의 범주로 구분

# | 특성값 | 설명 |
# | --- | --- |
# | <1H OCEAN | 해안에서 1시간 이내 |
# | INLAND | 내륙 |
# | NEAR OCEAN | 해안 근처 |
# | NEAR BAY | 샌프란시스코의 Bay Area 구역 |
# | ISLAND | 섬  |

# #### 수치형 특성 탐색

# <div align="center"><img src="imgs/ch02/housing-describe.png"></div>

# #### 수치형 특성별 히스토그램

# <div align="center"><img src="imgs/ch02/feature-histogram.png" width="600px"></div>

# ### 2.3.4 테스트셋 만들기

# * 모델 학습 시작 이전에 준비된 데이터셋을 훈련셋과 테스트셋으로 구분
#     * 테스트셋 크기: 전체 데이터셋의 20%

# * 테스트셋에 포함된 데이터는 미리 분석하지 말 것.
#   * 미리 분석 시 **데이터 스누핑 편향**을 범할 가능성이 높아짐
#   * 미리 보면서 알아낸 직관이 학습 모델 설정에 영향을 미칠 수 있음 

# * 훈련셋과 데이터셋을 구분하는 방식에 따라 결과가 조금씩 달라짐 
#     * 무작위 샘플링 vs. 계층적 샘플링

# * 여기서는 계층적 샘플링 활용

# #### 계층적 샘플링
# 
# * 계층: 동질 그룹
#     * 예제: 소득별 계층

# * 테스트셋: 전체 계층을 대표하도록 각 계층별로 적절한 샘플 추출

# * 예제: 소득 범주
#     * 계층별로 충분한 크기의 샘플이 포함되도록 지정해야 학습 과정에서 편향이 발생하지 않음
#     * 특정 소득 구간에 포함된 샘플이 과하게 적거나 많으면 해당 계층의 중요도가 과대 혹은 과소 평가됨

# * 전체 데이터셋의 중간 소득 히스토그램 활용
# 
# <div align="center"><img src="imgs/ch02/homl02-08.png" width="500"></div>

# * 대부분 구역의 중간 소득이 **1.5~6.0**(15,000~60,000&#x24;) 사이

# * 소득 구간을 아래 숫자를 기준으로 5개로 구분
# 
#     ```python
#     [0, 1.5, 3.0, 4.6, 6.0, np,inf]
#     ```

# #### 계층 샘플링과 무작위 샘플링 비교

# <div align="center"><img src="imgs/ch02/homl02-07.png" width="700"></div>

# ## 2.4 데이터 이해를 위한 탐색과 시각화

# ### 주의 사항
# 
# * 테스트셋을 제외한 훈련셋에 대해서만 시각화를 이용하여 탐색

# * 데이터 스누핑 편향 방지 용도

# ### 2.4.1 지리적 데이터 시각화
# 
# * 구역이 집결된 구역과 그렇지 않은 구역 구분 가능

# * 샌프란시스코의 베이 에어리어, LA, 샌디에고 등 밀집된 구역 확인 가능

# <div align="center"><img src="imgs/ch02/homl02-09.png" width="500"></div>

# * 주택 가격이 해안 근접도 또는 인구 밀도와 관련이 큼

# * 해안 근접도: 위치에 따라 다르게 작용
#   * 대도시 근처: 해안 근처 주택 가격이 상대적 높음
#   * 북부 캘리포니아 구역: 높지 않음

# <div align="center"><img src="imgs/ch02/homl02-11.png" width="500"></div>

# ### 2.4.2 상관관계 조사
# 
# * 중간 주택 가격 특성과 다른 특성 사이의 상관관계: 상관계수 활용

# <div align="center"><img src="imgs/ch02/homl02-12.png" width="600"></div>

# #### 상관계수의 특징

# <div align="center"><img src="imgs/ch02/homl02-14.png" width="400"></div>
# 
# <그림 출처: [위키백과](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)>

# * 상관계수: $[-1, 1]$ 구간의 값
# 
# * 1에 가까울 수록: 강한 양의 선형 상관관계
# 
# * -1에 가까울 수록: 강한 음의 선형 상관관계
# 
# * 0에 가까울 수록: 매우 약한 선형 상관관계

# #### 주의사항
# 
# * 상관계수가 0에 가까울 때: 선형 관계가 거의 없다는 의미이지, 아무런 관계가 없다는 의미는 아님

# * 상관계수는 기울기와 아무 연관 없음

# #### 상관계수를 통해 확인할 수 있는 정보
# 
# * 중간 주택 가격과 중간 소득의 상관계수가 0.68로 가장 높음
#     * 중간 소득이 올라가면 중간 주택 가격도 상승하는 경향이 있음
#     * 점들이 너무 넓게 퍼져 있음. 완벽한 선형관계와 거리 멂.
# 
# <div align="center"><img src="imgs/ch02/homl02-13.png" width="400"></div>

# * 50만 달러 수평선: 가격 제한 결과로 보임
#     * 45만, 35만, 28만, 그 아래 정도에서도 수평선 존재. 이유는 알려지지 않음.
#     * 이상한 형태를 학습하지 않도록 해당 구역을 제거하는 것이 좋음. (여기서는 그대로 두고 사용)

# ### 2.4.3 특성 조합으로 실험
# 
# * 구역별 방의 총 개수와 침실의 총 개수 대신 아래 특성이 보다 유용함
#     * 가구당 방 개수(rooms for household)
#     * 방 하나당 침실 개수(bedrooms for room)
#     * 가구당 인원(population per household)

# <div align="center"><img src="imgs/ch02/homl02-12a.png" width="600"></div>

# * 중간 주택 가격과 방 하나당 침실 개수의 연관성 다소 있음

# * 가구당 방 개수의 역할은 여전히 미미함

# # 2장 머신러닝 프로젝트 처음부터 끝까지 (2부)

# #### 감사의 글
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# ## 2.5 머신러닝 알고리즘을 위한 데이터 준비

# ### 데이터 준비 자동화

# * 모든 전처리 과정의 자동화. 언제든지 재활용 가능.

# * 자동화는 __파이프라인__(pipeline)으로 구현

# * 훈련셋 준비: 훈련에 사용되는 특성과 타깃 특성(레이블) 구분하여 복사본 생성

# ```python
# housing = strat_train_set.drop("median_house_value", axis=1)
# housing_labels = strat_train_set["median_house_value"].copy()
# ```

# * 테스트셋은 훈련이 완성된 후에 성능 측정 용도로만 사용.

# ### 데이터 전처리

# * 데이터 전처리(data preprocessing): 효율적인 모델 훈련을 위한 데이터 변환

# * 수치형 특성과 범주형 특성에 대해 다른 변환과정을 사용

# * 수치형 특성 전처리 과정
#   * 데이터 정제
#   * 조합 특성 추가
#   * 특성 스케일링

# * 범주형 특성 전처리 과정
#   * 원-핫-인코딩(one-hot-encoding)

# ### 변환 파이프라인

# * __파이프라인__(pipeline)
#     - 여러 과정을 한 번에 수행하는 기능을 지원하는 도구
#     - 여러 사이킷런 API를 묶어 순차적으로 처리하는 사이킷런 API

# * 파이프라인 적용 과정
#     * 수치형 특성 전처리 과정에 사용된 세 가지 변환 과정의 자동화 파이프라인 구현
#     * 수치형 특성 파이프라인과 범주형 특성 전처리 과정을 결합한 파이프라인 구현

# ### 사이킷런 API 활용

# * '조합 특성 추가' 과정을 제외한 나머지 변환은 사이킷런에서 제공하는 관련 API 직접 활용 가능

# * '조합 특성 추가' 과정도 다른 사이킷런 API와 호환이 되는 방식으로 사용자가 직접 구현 가능

# * 사이킷런에서 제공하는 API는 일관되고 단순한 인터페이스를 제공

# ### 사이킷런 API의 세 가지 유형

# #### 추정기(estimator)
#     

# * 주어진 데이터셋과 관련된 특정 파라미터 값들을 추정하는 객체

# * `fit()` 메서드 활용: 특정 파라미터 값을 저장한 속성이 업데이트된 객체 자신 반환

# #### 변환기(transformer):

# * fit() 메서드에 의해 학습된 파라미터를 이용하여 주어진 데이터셋 변환

# * `transform()` 메서드 활용

# * `fit()` 메서드와 `transform()` 메서드를 연속해서 호출하는 `fit_transform()` 메서드 활용 가능

# #### 예측기(predictor)

# * 주어진 데이터셋과 관련된 값을 예측하는 기능을 제공하는 추정기

# * `predict()` 메서드 활용

# * `fit()`과 `predict()` 메서드가 포함되어 있어야 함

# * `predict()` 메서드가 추정한 값의 성능을 측정하는 `score()` 메서드도 포함

# * 일부 예측기는 추정치의 신뢰도를 평가하는 기능도 제공

# <div align="center"><img src="imgs/ch02/scikit-learn-flow01.png" width="800"></div>
# 
# <그림 출처: [Scikit-Learn: A silver bullet for basic machine learning](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)>

# ### 2.5.1 데이터 정제: 수치형 특성 전치러 과정 1

# * 누락된 특성값이 존재 경우, 해당 값 또는 특성을 먼저 처리해야 함.

# * `total_bedrooms` 특성에 207개 구역에 대한 값이 null로 채워져 있음, 즉, 일부 구역에 대한 정보가 누락됨.

# <div align="center"><img src="imgs/ch02/null-value01.png" width="800"></div>

# #### null 값 처리 옵션

# * 옵션 1: 해당 구역 제거

# * 옵션 2: 전체 특성 삭제

# * 옵션 3: 평균값, 중앙값, 0, 주변에 위치한 값 등 특정 값으로 채우기. 책에서는 중앙값으로 채움.

# | 옵션 | 코드 |
# |--- | :--- |
# | 옵션 1 | `housing.dropna(subset=["total_bedrooms"])` |
# | 옵션 2 | `housing.drop("total_bedrooms", axis=1)` |
# | 옵션 3 | `median = housing["total_bedrooms"].median()` |
# |       | `housing["total_bedrooms"].fillna(median, inplace=True)` |
# 

# <옵션 3 활용>
# 
# <div align="center"><img src="imgs/ch02/null-value02.png" width="800"></div>

# #### SimpleImputer 변환기

# * 옵션 3를 지원하는 사이킷런 변환기

# * 중앙값 등 통계 요소를 활용하여 누락 데이터를 특정 값으로 채움

# ```python
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# housing_num = housing.drop("ocean_proximity", axis=1)
# imputer.fit(housing_num)
# ```

# ### 2.5.2 텍스트와 범주형 특성 다루기: 원-핫 인코딩

# * 범주형 특성인 해안 근접도(ocean_proximity)에 사용된 5개의 범주를 수치형 특성으로 변환해야 함.

# #### 단순 수치화의 문제점

# * 단순 수치화 적용 가능
# 
# | 범주 | 숫자 |
# |---|---|
# | <1H OCEAN | 0 |
# | INLAND | 1 |
# | ISLAND | 2 |
# | NEAR BAY | 3 |
# | NEAR OCEAN | 4 |

# * 하지만 여기서는 다음 문제 발생
#     * 해안 근접도는 단순히 구분을 위해 사용. 해안에 근접하고 있다 해서 주택 가격이 기본적으로 더 비싸지 않음.
#     * 반면에 수치화된 값들은 크기를 비교할 수 있는 숫자
#     * 따라서 모델 학습 과정에서 숫자들의 크기 때문에 잘못된 학습이 이루어질 수 있음.

# #### 원-핫 인코딩(one-hot encoding)

# * 수치화된 범주들 사이의 크기 비교를 피하기 위해 더미(dummy) 특성을 추가하여 활용
#     * 범주 수 만큼의 더미 특성 추가

# * 예를 들어, 해안 근접도 특성 대신에 다섯 개의 범주 전부를 새로운 특성으로 추가한 후 각각의 특성값을 아래처럼 지정
#   * 해당 카테고리의 특성값: 1
#   * 나머지 카테고리의 특성값: 0

# * 더미 특성 별로 1 또는 0의 값을 취하도록 모델 훈련 유도.

# #### OneHotEncoder  변환기

# * 원-핫 인코딩 지원

# * `sparse` 키워드 인자
#     - 기본값은 `True`. 
#     - 1의 위치만 기억하는 희소 행렬로 처리. 대용량 행렬 처리에 효과적임.
#     - `False`로 지정할 경우 일반 행렬로 처리.

# <div align="center"><img src="imgs/ch02/homl02-16.png" width="600"></div>

# ### 2.5.3 나만의 변환기: 수치형 특성 전처리 과정 2 (조합 특성 추가)

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

# <div align="center"><img src="imgs/ch02/custom-transformer.png" width="350"></div>
# 
# <그림 아이디어 출처: [Get the Most out of scikit-learn with Object-Oriented Programming](https://towardsdatascience.com/get-the-most-out-of-scikit-learn-with-object-oriented-programming-d01fef48b448)>

# ### 2.5.4 특성 스케일링: 수치형 특성 전처리 과정 3

# * 머신러닝 알고리즘은 입력 데이터셋의 특성값들의 스케일(범위)이 다르면 제대로 작동하지 않음

# * 특성에 따라 다루는 숫자의 크기가 다를 때 통일된 스케일링이 필요

# * 아래 두 가지 방식이 일반적으로 사용됨.
#     - min-max 스케일링
#     - 표준화 (책에서 사용)

# * __주의__: 타깃(레이블)에 대한 스케일링은 하지 않음

# #### min-max 스케일링

# * **정규화**(normalization)라고도 불림

# * 특성값 $x$를 **$\frac{x-min}{max-min}$**로 변환

# * 변환 결과: **0에서 1** 사이

# * 이상치에 매우 민감
#   * 이상치가 매우 **크면 분모가 매우 커져서** 변환된 값이 **0 근처**에 몰림

# #### 표준화(standardization)

# * 특성값 $x$ 를 **$\frac{x-\mu}{\sigma}$**로 변환
#   * $\mu$: 특성값들의 **평균**값
#   * $\sigma$: 특성값들의 **표준편차**

# * 결과: 변환된 데이터들이 **표준정규분포**를 이룸
#   * 이상치에 상대적으로 영향을 덜 받음.
#   

# * 사이킷런의 `StandardScaler` 변환기 활용 가능 (책에서 사용)

# #### 변환기 관련 주의사항

# * `fit()` 메서드: 훈련셋에 대해서만 적용. 테스트셋은 활용하지 않음.

# * `transform()` 메서드: 테스트셋 포함 모든 데이터에 적용 
#   * 훈련셋을 이용하여 필요한 파라미터를 확인한 후 그 값들을 이용하여 전체 데이터셋을 변환

# ### 2.5.5 변환 파이프라인

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

# # 2장 머신러닝 프로젝트 처음부터 끝까지 (3부)

# #### 감사의 글
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# ## 2.6 모델 선택과 훈련

# * 목표 달성에 필요한 두 요소를 결정해야함
#   * 학습 모델
#   * 회귀 모델 성능 측정 지표

# * 목표: 구역별 중간 주택 가격 예측 모델

# * 학습 모델: 회귀 모델

# * 회귀 모델 성능 측정 지표: 평균 제곱근 오차(RMSE)를 기본으로 사용

# ### 2.6.1 훈련셋에서 훈련하고 평가하기

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

# ### 2.6.2 교차 검증을 사용한 평가

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
# <div align="center"><img src="imgs/ch02/cross-val10.png" width="400"></div>

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

# ## 2.7 모델 세부 튜닝

# * 살펴 본 모델 중에서 **랜덤 포레스트** 모델의 성능이 가장 좋았음

# * 가능성이 높은 모델을 선정한 후에 **모델 세부 설정을 튜닝**해야함

# * 튜닝을 위한 세 가지 방식
#   * **그리드 탐색**
#   * **랜덤 탐색**
#   * **앙상블 방법**

# ### 2.7.1 그리드 탐색

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

# ### 2.7.2 랜덤 탐색

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

# ### 2.7.3 앙상블 방법

# * 결정 트리 모델 하나보다 랜덤 포레스트처럼 여러 모델로 이루어진 모델이 보다 좋은 성능을 낼 수 있음.

# * 또한 최고 성능을 보이는 서로 다른 개별 모델을 조합하면 보다 좋은 성능을 얻을 수 있음

# * 7장에서 자세히 다룸

# ### 2.7.4 최상의 모델과 오차 분석

# * 그리드 탐색과 랜덤 탐색 등을 통해 얻어진 최상의 모델을 분석해서 문제에 대한 좋은 통창을 얻을 수 있음

# * 예를 들어, 최상의 랜덤 포레스트 모델에서 사용된 특성들의 중요도를 확인하여 일부 특성을 제외할 수 있음.
#     * 중간 소득(median income)과 INLAND(내륙, 해안 근접도)가 가장 중요한 특성으로 확인됨
#     * 해안 근접도의 다른 네 가지 특성은 별로 중요하지 않음
#     * 중요도가 낮은 특성은 삭제할 수 있음.

# <div align="center"><img src="imgs/ch02/feature-importance.png" width="400"></div>

# ### 2.7.5 테스트 셋으로 시스템 평가하기

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

# <div align="center"><img src="imgs/ch02/model-launching01.png" width="600"></div>

# #### 데이터셋 및 모델 백업

# * 완성된 모델은 항상 백업해 두어야 함. 업데이트된 모델이 적절하지 않은 경우 이전 모델로 되돌려야 할 수도 있음.
#     * 백업된 모델과 새 모델을 쉽게 비교할 수 있음.

# * 동일한 이유로 모든 버전의 데이터셋을 백업해 두어야 함.
#     * 업데이트 과정에서 데이터셋이 오염될 수 있기 때문임.
