#!/usr/bin/env python
# coding: utf-8

# # 한눈에 보는 머신러닝

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# ## 머신러닝이란?

# **아서 새뮤얼<font size="2">Artuhr Samuel</font> (1959)**
# 
# > 명시적인 프로그래밍 없이 컴퓨터 스스로 학습하는 능력 에 대한 연구 분야
# 
# 
# **톰 미첼<font size="2">Tom Michell</font> (1977)**
# 
# > 경험 E를 통해 과제 T에 대한 프로그램의 수행 성능 P가 향상되면
# > 해당 프로그램은 경험 E를 통해 학습한다 라고 말한다.

# **머신러닝 주요 용어**
# 
# * __훈련 셋__<font size="2">training set</font>: 
#     머신러닝 프로그램이 훈련(학습)하는 데 사용하는 데이터 집합
# 
# * __훈련 사례__<font size="2">training instance</font> 
#     혹은 __샘플__<font size="2">sample</font>: 
#     각각의 훈련 데이터

# :::{prf:example} 스팸 필터
# :label: spam_filter
# 
# 스팸<font size="2">spam</font> 메일과 아닌 메일<font size="2">ham</font>의 구분법을 
# 머신러닝으로 학습시킬 수 있으며, 톰 미첼의 머신러닝 정의와 관련해서 다음이 성립한다.
# 
# - 작업 T = 새로운 메일의 스팸 여부 판단
# - 경험 E = 훈련 데이터
# - 성능 P = 스팸 여부 판단의 정확도
# :::

# ## 머신러닝 활용

# ### 전통적 프로그래밍

# 전통적 프로그래밍 다음 과정으로 진행된다.
# 
# 1. 문제 연구: 문제 해결 알고리즘 연구
# 1. 규칙 작성: 알고리즘 규현
# 1. 평가: 구현된 프로그램 테스트
#     * 테스트 통과: 프로그램 론칭
#     * 테스트 실패: 오차 분석 후 1단계로 이동
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-01.png?raw=true" style="width:400px;"></div>

# :::{prf:example} 전통적 프로그래밍 스팸 메일 분류
# :label: spam_classification
# 
# * 특정 단어가 들어가면 스팸 메일로 처리.
# * 프로그램이 론칭된 후 새로운 스팸단어가 사용될 때 스팸 메일 분류 실패.
# * 개발자가 새로운 규칙을 매번 업데이트 시켜줘야 함.
# * 유지 보수 어려움.
# :::

# ### 머신러닝 프로그래밍

# 스팸으로 지정된 메일에 "광고", "투^^자", "무&#10084;료" 등의 표현이 
# 자주 등장하는 경우 새로운 메일에 그런 표현이 사용되면 
# 자동으로 스팸으로 분류하도록 스스로 학습하는 프로그램을 작성한다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-02.png?raw=true" style="width:420px;"></div>

# **머신러닝 프로그램 학습 과정의 자동화**
# 
# 머신러닝 프로그램을 학습시키는 과정을 관장하는 __머신러닝 파이프라인__ 
# 또는 __MLOps(Machine Learning Operations, 머신러닝 운영)__ 의
# 자동화가 가능하다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-03.png?raw=true" style="width:440px;"></div>

# **머신러닝 프로그래밍의 장점**
# 
# * 스팸 메일 분류기 처럼 알고리즘에 대한 너무 많은 세부 튜닝과
#     매우 긴 규칙을 요구하는 문제를 해결할 수 있다.
# * 음성 인식 등 전통적인 방식으로 해결하기에 너무 복잡한 문제를 해결할 수 있다.
# * 새로운 데이터에 바로 적용이 가능한 시스템을 쉽게 재훈련할 수 있다.
# * 머신러닝 프로그램으로 생성된 솔루션 분석을 통해 
#     데이터에 대한 통찰을 얻을 수 있다. 
#     즉, **데이터 마이닝**<font size="2">data mining</font>이 가능하다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-04.png?raw=true" style="width:440px;"></div>

# ## 머신러닝 활용 사례

# **회귀 분석**
# 
# 회사의 차년도 수익 등을 예측할 수 있다.
# 
# - 선형 회귀 (4장)
# - SVM 회귀 (5장)
# - 랜덤 포레스트 회귀 (7장)
# - 신경망 회귀 모델
# - 주식, 날씨 등 순서의 흐름을 고려해야 하는 경우:
#     - 순환 신경망(RNN)
#     - 합성곱 신경망(CNN)
#     - 트랜스포머(Transformer)

# **데이터 시각화**
# 
# 고차원의 복잡한 데이터셋을 2, 3차원의 그래프로 시각화할 수 있다.
# 
# - 차원 축소 (8장)

# **이상치 탐지**
# 
# 신용 카드의 부정 사용을 감지할 수 있다.
# 
# - 가수시안 혼합 모델 (9장)
# - 오토인코더

# **군집화**
# 
# 구매 이력을 활용하는 고객 분류 및 마케팅 전략 계획 수립이 가능하다.
# 
# - K-평균 (9장)
# - DBSCAN (9장)

# **이미지 분류 작업**
# 
# 생산 라인 제품 이미지 자동 분류
# 
# - CNN
# - Transformer

# **시맨틱 분할 작업**
# 
# 뇌 스캔 활용 종양 진단
# 
# - CNN
# - Transformer

# **자연어 처리(NLP)**
# 
# 텍스트 분류(뉴스 기사 자동 분류. 공격적 언급/기사 자동 분류), 텍스트 요약
# 
# - RNN
# - CNN
# - Transformer

# **자연어 이해(NLU)**
# 
# 챗봇(chatbot) 또는 개인 비서 만들기
# 
# - RNN
# - CNN
# - Transformer
# - 질문-답변 모듈

# **음성 인식**
# 
# 음성 명령에 반응하는 앱을 구현할 수 있다.
# 
# - RNN
# - CNN
# - Transformer

# **추천 시스템**
# 
# 과거 구매 이력을 활용하여 관심 상품을 추천한다.
# 
# - 신경망<font size="2">Neural Network</font> 활용

# **강화 학습**
# 
# 알파고(AlphaGo) 등 지능형 봇을 활용한 게임을 구현할 수 있다.
# 
# - 강화 학습<font size="2">Reinforcement Learning</font>

# :::{admonition} 용어 정리
# :class: tip
# 
# - 회귀 = Regression
# - 선형 회귀 = Linear Regression
# - SVM = Support Vector Machine (서포트 벡터 머신)
# - 랜덤 포레스트 = Random Forest
# - 신경망 = Neural Network
# - 인공 신경망 = Artificial Neural Network
# - CNN = Convolutional Neural Network (합성곱 신경망)
# - NLP = Natural Language Processing (자연어 처리)
# - NLU = Natural Language Understanding (자연어 이해)
# - 트랜스포머 = Transformer
# :::

# ## 머신러닝 시스템 유형

# 머신러닝 시스템의 유형을 다양한 기준으로 분류할 수 있다.

# 1. 훈련 지도 여부
#     * 지도 학습
#     * 비지도 학습
#     * 준지도 학습
#     * 자기주도 학습
#     * 강화 학습
# 1. 실시간 훈련 여부
#     * 온라인 학습
#     * 배치 학습
# 1. 예측 모델 사용 여부
#     * 사례 기반 학습
#     * 모델 기반 학습

# 언급된 분류 기준이 상호 배타적이지 않다. 
# 예를 들어, 신경망을 활용하는 스팸 필터 프로그램의 다음 방식을 모두 사용할 수 있다.
# - 지도 학습: 스팸 메일과 스팸이 아닌 메일로 이루어진 훈련 데이터셋으로 스팸 필터 분류 프로그램 훈련
# - 온라인 학습: 실시간 학습 가능
# - 모델 기반 학습: 훈련 결과로 생성된 모델을 이용하여 스팸 여부 판단

# ### 훈련 지도 여부

# #### 지도 학습

# **지도 학습**<font size="2">supervised learning</font>은
# 훈련 데이터에 **레이블**<font size="2">label</font>이라는 답을 표기하여 
# 레이블을 맞추도록 유도하는 학습을 가리킨다. 
# 지도 학습은 기본적으로 **분류**<font size="2">classification</font> 또는 
# **회귀**<font size="2">regression</font> 문제에 적용된다.

# **분류**
# 
# 예를 들어, 스팸 필터는 메일 데이터의 **특성**<font size="2">feature</font>을 이용하여 스팸 여부를 판단한다.
# 
# * 메일 데이터 특성: 소속 정보, 특정 단어 포함 여부 등
# * 레이블: 스팸 또는 햄    
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-05.png?raw=true" style="width:450px;"></div>

# **회귀**
# 
# 예를 들어, 중고차 데이터의 특성을 이용하여 가격을 예측한다.
# 회귀 문제의 경우 레이블 대신에 **타깃**<font size="2">target</font> 표현을 보다 많이 사용한다.
# 
# * 중고차 데이터 특성: 주행거리, 연식, 브랜드 등
# * 타깃: 중고차 가격
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-06.png?raw=true" style="width:400px;"></div>

# #### 비지도 학습

# **비지도 학습**<font size="2">unsupervised learning</font>은
# 군집화, 데이터 시각화, 차원 축소, 이상 탐지, 연관 규칙 학습 등의 
# 프로그램 구현에 적용되며,
# 프로그램 학습에 레이블이 없는 훈련 데이터를 이용한다.

# **군집화**
# 
# 쇼핑몰 사이트 방문자를 비슷한 특징을 갖는 사람들의 그룹으로 구분하는 
# **군집화**<font size="2">clustering</font> 프로그램을 학습시킬 수 있다.

# - 훈련 전
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-07.png?raw=true" style="width:410px;"></div>

# - 훈련 후 
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-08.png?raw=true" style="width:400px;"></div>

# **데이터 시각화**
# 
# 매우 많은 특성을 갖는 
# 훈련 데이터를 두 개 또는 세 개의 특성만을 갖는
# 데이터로 변환한 후에 2D 또는 3D로 시각화 할 수 있다.
# 이를 위해 훈련 데이터의 특성을 대표하는 2, 3 개의 특성을 추출해야 하며
# 이를 위해 **차원 축소**<font size="2">dimensionality reduction</font> 
# 프로그램을 비지도 학습으로 훈련시킬 수 있다.

# 아래 그림은 다양한 종류의 데이터를 두 개의 특성만을 갖는 데이터로 차원을 축소한 후
# 시각화한 결과를 보여준다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-09.png?raw=true" style="width:650px;"></div>

# :::{admonition} 차원 축소 활용
# :class: tip
# 
# 차원 축소 기법은 시각화 뿐만 아니라 자체적으로 많이 활용된다.
# 훈련 데이터의 차원을 축소하면 머신러닝 프로그램의 학습이 보다 빠르게 진행되고
# 보다 성능이 좋은 프로그램이 구현될 수 있다.
# :::

# **이상 탐지**
# 
# 신용카드 거래 중에서 부정거래 사용을 감지하거나
# 제조 과정 중에 결함있는 제품 확인하기 등을 
# **이상 탐지**<font size="2">anomaly detection</font>라고 한다.
# 이상 탐지 프로그램은 수 많은 정상 샘플을 이용하여 훈련한 후에
# 새롭게 샘플의 정상 여부를 판단한다.
# 또한 훈련 데이터셋에 포함된 비정상 샘플을 제거하여 머신러닝 모델의 학습을 
# 보다 효율적으로 진행하도록 할 수 있다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-10.png?raw=true" style="width:400px;"></div>

# :::{prf:example} 이상치 탐지 vs. 특이치 탐지
# :label: outlier_novelty
# 
# **이상치**<font size="2">outlier</font>는 
# 대부분을 차지하는 정상 데이터와 다른 특성을 갖는 데이터를 가리킨다.
# **이상치 탐지**<font size="2">outlier detection</font>은 
# 데이터셋에 포함된 대다수의 정상 데이터와는 다른 속성을 갖는 이상치를 탐지한다.
# 반면에 **특이치 탐지**<font size="2">novelty detection</font>는 
# 기존의 데이터셋에 포함되지 않은 다른 종류의 데이터를 탐지한다. 
# 
# 예를 들어, 수 천장의 강아지 사진으로 구성된 데이터셋에 치와와 사진이 1%정도 포함되었다고 가정하자.
# 그러면 새로운 치와와 사진이 입력되었을 때 특이치 탐지 알고리즘은 그 사진에
# 포함된 치와와를 특이한 것으로 간주하지 않는다.
# 반면에 이상치 탐지 알고리즘은 새로운 사진의 치와와를
# 훈련 데이터셋에 포함된 대부분의 강아지들과 다른 품종으로 분류한다.
# :::

# **연관 규칙 학습**
# 
# 훈련 데이터 특성들 간의 흥미로운 관계를 찾는 데에 비지도 학습이 활용될 수 있다.
# 예를 들어, 마트 판매 기록 데이터에서 
# 바비규 소스와 감자를 구매하는 고객이 스테이크도 구매하는 경향이 있음을 파악하여 
# 서로 연관된 상품을 가깝게 진열할 수 있다.

# #### 준지도 학습

# * 레이블이 적용된 적은 수의 샘플이 주어졌을 때 유횽함.
# 
# * 비지도 학습을 통해 군집을 분류한 후 샘플들을 활용해 지도 학습 실행
# 
# * 대부분 지도 학습과 비지도 학습 혼합 사용

# **준지도 학습 예제**
# 
# * 아래 그림 참조: 새로운 사례 `X`를 세모에 더 가깝다고 판단함.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-11.png?raw=true" style="width:400px;"></div>
# 
# * 구글 포토 호스팅: 가족 사진 몇 장에만 레이블 적용. 이후 모든 사진에서 가족사진 확인 가능.

# #### 자기지도 학습

# ...

# #### 강화 학습

# * 에이전트(학습 시스템)가 취한 행동에 대해 보상 또는 벌점을 주어 가장 큰 보상을 받는 방향으로 유도하기

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-12.png?raw=true" style="width:400px;"></div>

# * 예제: 딥마인드(DeepMind)의 알파고(AlphaGo)

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/alphago01.png?raw=true" style="width:500px;"></div>

# ### 실시간 훈련 여부 구분

# #### 배치 학습(batch learning)

# * 주어진 훈련 세트 전체를 사용해 오프라인에서 훈련

# * 먼저 시스템을 훈련시킨 후 더 이상의 학습 없이 제품 시스템에 적용

# * 단점
#     * 컴퓨팅 자원(cpu, gpu, 메모리, 저장장치 등)이 충분한 경우에만 사용 가능
#     * 새로운 데이터가 들어오면 처음부터 새롭게 학습해야 함. 
#         * 하지만 MLOps 등을 이용한 자동화 가능

# #### 온라인 학습(online learing)

# * 하나씩 또는 적은 양의 데이터 묶음(미니배치, mini-batch)를 사용해 점진적으로 훈련

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-13.png?raw=true" style="width:500px;"></div>

# * 단점
#     * 나쁜 데이터가 주입되는 경우 시스템 성능이 점진적으로 떨어질 수 있음.
#     * 지속적인 시스템 모니터링 필요

# * 예제
#     * 주식가격 시스템 등 실시간 반영이 중요한 시스템
#     * 스마트폰 등 제한된 자원의 시스템
#     * 외부 메모리 학습: 매우 큰 데이터셋 활용하는 시스템

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-14.png?raw=true" style="width:500px;"></div>

# ### 예측 모델 사용 여부 구분

# * 훈련 모델의 __일반화(generalization)__ 방식에 따른 분류
# * 일반화 = '새로운 데이터에 대한 예측'

# #### 사례 기반 학습

# * **샘플을 기억**하는 것이 훈련의 전부

# * 예측을 위해 기존 샘플과의 **유사도** 측정

# * 예제: k-최근접 이웃(k-NN, k-nearest neighbors) 알고리즘

# * k-NN 활용 예제: 새로운 샘플 `X`가 기존에 세모인 샘플과의 유사도가 높기 때문에 세모로 분류.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-15.png?raw=true" style="width:450px;"></div>

# #### 모델 기반 학습

# * 모델을 미리 지정한 후 훈련 세트를 사용해서 모델을 훈련시킴

# * 훈련된 모델을 사용해 새로운 데이터에 대한 예측 실행

# * 예제: 이 책에서 다루는 대부분의 알고리즘

# * 예제: 학습된 모델을 이용하여 새로운 데이터 `X`를 세모 클래스로 분류

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-16.png?raw=true" style="width:450px;"></div>

# ### 선형 모델 학습 예제

# * 목표: OECD 국가의 1인당 GDP(1인당 국가총생산)와 삶의 만족도 사이의 관계 파악

# * 1인당 GDP가 증가할 수록 삶의 만족도가 선형으로 증가하는 것처럼 보임.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-17.png?raw=true" style="width:450px;"></div>

# * 데이터를 대표하는 하나의 직선(선형 모델)을 찾기
# 
#     $$
#     \text{'삶의만족도'} = \theta_0 + \theta_1 \times \text{'1인당GDP'}
#     $$

# * 데이터를 대표할 수 있는 선형 방정식을 찾아야 함

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-18.png?raw=true" style="width:450px;"></div>

# * 학습되는 모델의 성능 평가 기준을 측정하여 가장 적합한 모델 학습
#     * 효용 함수: 모델이 얼마나 좋은지 측정
#     * 비용 함수: 모델이 얼마나 나쁜지 측정

# * 아래 선형 모델이 최적!

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-19.png?raw=true" style="width:450px;"></div>

# ## 머신러닝의 주요 도전 과제

# **충분하지 않은 양의 훈련 데이터**
# 
# * 간단한 문제라도 수천 개 이상의 데이터가 필요
# 
# * 이미지나 음성 인식 같은 문제는 수백만 개가 필요할 수도 있음
# 
# * 데이터가 부족하면 알고리즘 성능 향성 어려움
# 
# * 일반적으로 데이터가 많을 수록 모델의 성능 높아짐.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-20.png?raw=true" style="width:450px;"></div>

# **대표성 없는 훈련 데이터**
# 
# * 샘플링 잡음: 우연에 의해 추가된 대표성이 없는 데이터
# 
# * 샘플링 편향: 표본 추출 방법이 잘못되어 한 쪽으로 쏠린 대표성이 없는 데이터
# 
# * 예제: 1인당 GDP와 삶의 만족도 관계
#     - 잡음: 빨강 네모 데이터가 추가 될 경우 선형 모델 달라짐.
#     - 편향: OECD 국가중에서 이름에 영어 알파벳 W가 포함된 국가들은 삶의 만족도가 매우 높음. 하지만 일반화는 불가능.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-21.png?raw=true" style="width:600px;"></div>

# **낮은 품질의 데이터 처리**
# 
# * 이상치 샘플이라면 고치거나 무시
# 
# * 특성이 누락되었다면
#     * 해당 특성을 제외
#     * 해당 샘플을 제외
#     * 누락된 값을 채움
#     * 해당 특성을 넣은 경우와 뺀 경우 각기 모델을 훈련

# **관련이 없는 특성**
# 
# * 풀려는 문제에 관련이 높은 특성을 찾아야 함
# 
# * 특성 선택: 준비되어 있는 특성 중 가장 유용한 특성을 찾음
# 
# * 특성 추출: 특성을 조합하여 새로운 특성을 만듦

# **과대 적합**
# 
# * 훈련 세트에 특화되어 일반화 성능이 떨어지는 현상
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-22.png?raw=true" style="width:550px;"></div>
# 
# * 규제를 적용해 과대적합을 감소시킬 수 있음 
# 
# * 파라미터를 조정되는 과정에 규제 적용
# 
# * 파랑 점선이 규제를 적용해 훈련된 선형 모델임.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-23.png?raw=true" style="width:550px;"></div>

# **과소 적합**
# 
# * 모델이 너무 단순해서 훈련 세트를 잘 학습하지 못함
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch04/homl04-06.png?raw=true" style="width:400px;"></div>
# 
# * 해결 방법
#     * 보다 많은 모델 파라미터를 사용하는 모델 적용
#     * 보다 좋은 특성 활용
#     * 보다 규제 강도 적용

# ## 테스트와 검증

# ### 검증

# * 훈련된 모델의 성능 평가: 테스트 세트 활용
# 
# * 전체 데이터셋을 훈련 세트(80%)와 테스트 세트(20%)로 구분
#     * 훈련 세트: 모델 훈련용.
#     * 테스트 세트: 모델 테스트용
#     * 데이터셋이 매우 크면 테스트 세트 비율을 낮출 수 있음.
# 
# * 검증 기준: __일반화 오차__
#     * 새로운 샘플에 대한 오류 비율
#     * 학습된 모델의 일반화 성능의 기준
# 
# * 과대 적합: 훈련 오차에 비해 일반화 오차가 높은 경우

# ### 하이퍼파라미터(hyper-parameter)

# * 알고리즘 학습 모델을 지정에 사용되는 파라미터
# 
# * 훈련 과정에 변하는 파라미터가 아님
# 
# * 하이퍼파라미터를 조절하면서 가장 좋은 성능의 모델 선정

# ### 교차 검증

# * 예비표본(홀드아웃, holdout) 검증
#     * 예비표본(검증세트): 훈련 세트의 일부로 만들어진 데이터셋
#     * 다양한 하이퍼파라미터 값을 사용하는 후보 모델을 평가하는 용도로 예비표본을 활용하는 기법
# 
# * 교차 검증
#     * 여러 개의 검증세트를 사용한 반복적인 예비표본 검증 적용 기법
#     * 장점: 교차 검증 후 모든 모델의 평가를 평균하면 훨씬 정확한 성능 측정 가능
#     * 단점: 훈련 시간이 검증 세트의 개수에 비례해 늘어남

# ### 검증 예제: 데이터 불일치

# * 모델 훈련에 사용된 데이터가 실전에 사용되는 데이터를 완벽하게 대변하지 못하는 경우
# 
# * 예제: 꽃이름 확인 알고리즘 
#     * 인터넷으로 구한 꽃사진으로 모델 훈련
#     * 이후 직접 촬영한 사진으로 진행한 성능측정이 낮게 나오면 __데이터 불일치__ 가능성 높음
# 
# * 데이터 불일치 여부 확인 방법
#     * 훈련-개발 세트: 예를 들어, 인터넷에서 다운로드한 꽃사진의 일부로 이루어진 데이터셋
#     * 훈련-개발 세트를 제외한 나머지 꽃사진으로 모델 훈련 후, 훈련-개발 세트를 이용한 성능 평가 진행
# 
# * 훈련-개발 세트에 대한 평가가 좋은 경우: 과대적합 아님
#     * 훈련-개발 세티트에 평가는 좋지만 (실제 찍은 사진으로 이루어진) 검증 세트에 대한 평가 나쁜 경우: 데이터 불일치
#     * 다운로드한 사진을 실제 찍은 사진처럼 보이도록 전처리 한 후에 다시 훈련시키면 성능 향상시킬 수 있음.
# 
# * 훈련-개발 세트에 대한 평가가 나쁜 경우: 과대적합
#     * 모델에 규제를 적용하거나 더 많은 훈련 데이터 활용해야 함.
