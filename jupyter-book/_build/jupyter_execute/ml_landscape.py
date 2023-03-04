#!/usr/bin/env python
# coding: utf-8

# (ch:ml_landscape)=
# # 한눈에 보는 머신러닝

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 한눈에 보는 머신러닝](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_ml_landscape.ipynb)에서 
# 확인할 수 있다.

# ## 머신러닝이란?

# **아서 새뮤얼<font size="2">Artuhr Samuel</font> (1959)**
# 
# > 컴퓨터 프로그램을 명시적으로 구현하는 대신 컴퓨터 스스로 학습하는 능력를
#     갖도록 하는 연구 분야
# 
# **톰 미첼<font size="2">Tom Mitchell</font> (1977)**
# 
# > 과제 T에 대한 프로그램의 성능 P가 경험 E를 통해 향상되면
# > 해당 "프로그램이 경험 E를 통해 학습한다" 라고 말한다.

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

# **데이터셋 용어**
# 
# * __훈련셋__<font size="2">training set</font>: 
#     머신러닝 프로그램이 훈련(학습)하는 데 사용하는 데이터 집합
# 
# * __훈련 사례__<font size="2">training instance</font> 
#     혹은 __샘플__<font size="2">sample</font>: 
#     각각의 훈련 데이터

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
# * 새로운 규칙이 생겼을 때 사용자가 매번 업데이트를 시켜줘야하기 때문에 유지 보수가 어려움.
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
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-04.png?raw=true" style="width:440px;"></div>

# ## 머신러닝 시스템 유형

# 머신러닝 시스템의 유형을 다양한 기준으로 분류할 수 있다.

# 1. 훈련 지도 여부
#     * 지도 학습
#     * 비지도 학습
#     * 준지도 학습
#     * 자기주도 학습
#     * 강화 학습
# 1. 실시간 훈련 여부
#     * 배치 학습
#     * 온라인 학습
# 1. 예측 모델 사용 여부
#     * 사례 기반 학습
#     * 모델 기반 학습

# :::{admonition} 분류 기준 적용
# :class: tip
# 
# 언급된 분류 기준이 상호 배타적이지 않다. 
# 예를 들어, 신경망을 활용하는 스팸 필터 프로그램의 다음 방식을 모두 사용할 수 있다.
# - 지도 학습: 스팸 메일과 스팸이 아닌 메일로 이루어진 훈련셋으로 모델 학습 진행
# - 온라인 학습: 실시간 학습 가능
# - 모델 기반 학습: 훈련 결과로 생성된 모델을 이용하여 스팸 여부 판단
# :::

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
# 군집화, 데이터 시각화, 차원 축소, 예외 탐지, 연관 규칙 학습 등의 
# 프로그램 구현에 적용되며,
# 프로그램 학습에 레이블이 없는 훈련 데이터를 이용한다.

# **군집화**
# 
# 쇼핑몰 사이트 방문자를 비슷한 특징을 갖는 사람들의 그룹(남성, 여성, 주말, 주중, 의류, 전자기기 등)으로 묶는 
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

# **차원 축소**
# 
# 차원 축소 기법은 시각화 뿐만 아니라 자체적으로 많이 활용된다.
# 훈련 데이터의 차원을 축소하면 머신러닝 프로그램의 학습이 보다 빠르게 진행되고
# 보다 성능이 좋은 프로그램이 구현될 수 있다.

# **예외 탐지**
# 
# 신용카드 거래 중에서 부정거래 사용을 감지하거나
# 제조 과정 중에 결함있는 제품을 확인하는 일을 
# **예외 탐지**<font size="2">anomaly detection</font>라고 한다.
# 예외 탐지 프로그램은 수 많은 샘플을 이용하여 훈련한 후에
# 새롭게 입력된 샘플의 정상/비정상 여부를 판단하여 이상치를 확인한다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-10.png?raw=true" style="width:400px;"></div>

# **연관 규칙 학습**
# 
# 훈련 데이터 특성들 간의 흥미로운 관계를 찾는 데에 비지도 학습이 활용될 수 있다.
# 예를 들어, 마트 판매 기록 데이터에서 
# 바비규 소스와 감자를 구매하는 고객이 스테이크도 구매하는 경향이 있음을 파악하여 
# 서로 연관된 상품을 가깝게 진열할 수 있다.

# #### 준지도 학습

# 레이블이 적용된 훈련 데이터의 수가 적고,
# 레이블이 없는 훈련 데이터가 훨씬 많이 있을 때
# **준지도 학습**<font size="2">semi-supervised learning</font>을 활용한다.
# 아래 그림은 새로운 샘플 &#x274C;를 기존에 레이블을 갖는 
# 훈련 데이터에 근거하여 세모와 같은 부류로 취급한다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-11.png?raw=true" style="width:400px;"></div>

# 준지도 학습은 또한 레이블이 적용된 훈련 데이터의 정보를
# 전체 훈련 데이터셋으로 전파하는 데에 활용될 수 있다.
# 예를 들어 구글 포토<font size="2">Google Photos</font>는 
# 가족 사진 몇 장에 사람 이름을 레이블로 지정하면
# 다른 모든 사진에서 지정된 이름의 사람이 포함된 사진을 찾아준다.
# 이처럼 실전에 사용되는 많은 머신러닝 알고리즘이 준지도 학습과 지도 학습을 함께 사용한다.

# #### 자기지도 학습

# **자기지도 학습**<font size="2">self-supervised learning</font>은
# 레이블이 전혀 없는 샘플로 구성된 데이터셋으로부터
# 레이블을 모두 갖는 샘플로 구성된 데이터셋을 생성하여 모델 훈련을 진행하는 기법이다.
# 
# 예를 들어 레이블이 전혀 없는 사진으로 구성된 데이터셋의 각 이미지를 대상으로
# 잘라내기<font size="2">crop</font> 또는 크기 변경<font size="2">resize</font>을 
# 수행하여 얻어진 훈련 데이터셋을 생성한다.
# 이제 얻어진 훈련 데이터셋으로부터 본래의 이미지를 재생하는 머신러닝 프로그램을 
# 학습시킨다. 이때 본래의 이미지 데이터를 생성된 이미지의 레이블로 사용한다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/self-supervised01.jpg?raw=true" style="width:300px;"></div>
# 
# <p style="text-align: center;">
#     그림 출처: <a href="https://paperswithcode.com/task/self-supervised-image-classification">Self-Supervised Image Classification</a>
# </p>

# **이미지 복원**
# 
# 자기지도 학습 프로그램을 이용하여 손상된 이미지를 복구할 수 있다.

# **반려동물 분류**
# 
# 손상된 이미지를 복구할 수 있는 능력을 갖는 프로그램을 조금 응용하면
# 고양이, 강아지 등 반려동물의 품종을 구분하는 프로그램으로 활용할 수 있다.
# 이유는 고양이 사진을 복구할 수 있다면 고양이와 강아지를 구분할 수 있기 때문이다.

# #### 강화 학습

# **에이전트**<font size="2">agent</font>라고 불리는 
# 학습 시스템이 주어진 상황에서 취한 행동에 따라 보상과 벌점을 받는다. 
# 이를 통해 주어진 상황에서 가장 많은 보상을 받는 **정책**<font size="2">policy</font>, 
# 즉 최선의 전략<font size="2">strategy</font>을
# 스스로 학습한다. 
# 정책은 특정 상황에서 에이전트가 취해야 하는 최선의 행동을 지정한다. 

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-12.png?raw=true" style="width:400px;"></div>

# :::{prf:example} 알파고와 이세돌의 바둑 시합
# :label: alphago
# 
# 2016년 3월 당시 세계 최고의 바둑기사 이세돌과 
# 구글의 딥마인드<font size="2">DeepMind</font>가 개발한
# 바둑 프로그램 알파고<font size="2">AlphaGo</font>의 다섯 번의 대국에서
# 알파고가 4대1로 승리했다.
# 알파고는 유명한 바둑 기보를 학습하고 스스로와의 대결을 통해 승리 정책을
# 학습했다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/alphago01.png?raw=true" style="width:450px;"></div>
# :::

# ### 실시간 훈련 여부

# 고정된 하나의 훈련 데이터셋을 대상으로 학습하느냐,
# 아니면 조금씩 추가되는 데이터셋을 대상으로 실시간 학습이 가능하느냐에 따라
# 머신러닝 시스템을 구분할 수 있다.

# #### 배치 학습

# 주어진 훈련셋 전체를 활용해 오프라인에서 훈련하는 것이 
# **배치 학습**<font size="2">batch learning</font>이다.
# 한 번 훈련된 시스템은 더 이상의 학습 없이 제품에 적용된다.
# 배치 학습은 컴퓨팅 자원(cpu, gpu, 메모리, 저장장치 등)이 충분한 경우에만 사용할 수 있다.
# 예를 들어 스마트폰, 화성 탐사선 등에서는 배치 학습이 어렵다.

# #### 온라인 학습

# **온라인 학습**<font size="2">online learning</font>은 
# 하나씩 또는 **미니 배치**<font size="2">mini-batch</font>라 불리는
# 적은 양의 데이터 묶음을 사용해 점진적으로 학습하는 훈련 기법을 가리킨다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-13.png?raw=true" style="width:500px;"></div>

# 온라인 학습은 주식가격 시스템 등 실시간 반영이 중요한 시스템 구현에 유용하게 활용된다. 
# 또한 스마트폰 등 제한된 컴퓨팅 자원을 갖는 시스템에서도 활용될 수 있다.

# 검색 순위 조작 등 시스템을 악용하려는 의도를 갖거나
# 시스템에 나쁜 영향을 주는 데이터가 주입되는 경우 시스템 성능이 떨어질 수 있다.
# 따라서 지속적인 시스템 모니터링이 요구되며
# 필요하면 시스템 전체를 새로 세팅해거나 이전에 잘 작동했던 세팅으로 되돌려야 한다.

# **외부 메모리 학습**
# 
# 매우 큰 훈련 데이터셋은 메모리에 한꺼번에 불러올 수 없기에
# 훈련 데이터셋을 미니 배치로 나누어 점진적 학습에 활용할 수 있다.
# 이를 **외부 메모리 학습**<font size="2">out-of-core learning</font>이라 한다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-14.png?raw=true" style="width:500px;"></div>

# ### 예측 모델 사용 여부

# 훈련된 모델의 일반화 방식에 따라
# **사례 기반 학습**<font size="2">instance-based learning</font>과 
# **모델 기반 학습**<font size="2">model-based learning</font>으로 나뉜다.
# 여기서 말하는 **일반화**<font size="2">generalization</font>는
# 훈련 과정에 사용되지 않는 새로운 데이터에 대한 예측 수행을 의미한다.

# #### 사례 기반 학습

# 새로운 데이터에 대한 예측을 수행하기 위해 기존 샘플과의 
# **유사도**<font size="2">similarity</font>를 측정한 후,
# 가장 높은 유사도를 갖는 샘플에 대해 사용했던 예측값을 그대로 사용한다.

# :::{prf:example} k-최근접 이웃
# :label: k-nearst
# 
# 새로운 샘플 &#x274C;가 세모 샘플과의 유사도가 가장 높기 때문에 세모로 분류된다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-15.png?raw=true" style="width:450px;"></div>
# :::

# (sec:model_based_learning)=
# #### 모델 기반 학습

# 훈련 데이터셋을 대상으로 모델을 미리 지정한 후 예측을 수행하는 모델을 훈련시킨다.

# :::{prf:example} 다항 회귀
# :label: model-based
# 
# 검정 파선으로 표시된 구분선을 기준으로 이용하여 새로운 데이터 &#x274C;를 세모 클래스로 분류한다.
# 구분선이 바로 학습된 모델이다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-16.png?raw=true" style="width:450px;"></div>
# :::

# **선형 모델 학습 예제**

# OECD(경제협력개발기구) 국가의 1인당 GDP(국내 총생산)와 삶의 만족도 사이의 관계를 
# 확인하려 한다.
# 여기서는 2015년 기준으로 OECD에 속한 36개 국가의 데이터를 이용한다. 
# 아래 표는 그중 5개 국가의 1인당 GDP와 삶의 만족도를 보여준다. 

# | 국가 | 1인당 GDP(미국 달러) | 삶의 만족도 |
# | :--- | :--- | :--- |
# | 헝가리 | 12,240 | 4.9 |
# | 한국 | 27,195 | 5.8 |
# | 프랑스 | 37,675 | 6.5 |
# | 호주 | 50,9672 | 7.3 |
# | 미국 | 55,805 | 7.2 |

# 아래 그래프는
# 36개 국가 전체를 대상으로 1인당 GDP와 삶의 만족도를 이용한
# 산점도<font size="2">scatter plot</font>이다. 
# 1인당 GDP가 증가할 수록 삶의 만족도가 
# **선형적**<font size="2">linear</font>으로 증가하는 것처럼 보인다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-17.png?raw=true" style="width:450px;"></div>

# 위 가정을 바탕으로 1인당 GDP가 알려진 국가의 삶의 만족도를 예측하는 머신러닝 모델을 구현해보자.
# 즉, 1인당 GDP와 삶의 만족도 사이의 관계를 설명하는 모델로 
# **선형 모델**<font size="2">linear model</font>을 선택한다.
# 이 결정에 따라 구현될 모델은 아래 모양의 일차 방정식을 이용하여 
# 1인당 GDP가 주어졌을 때 해당 국가의 삶의 만족도를 예측하게 된다.
# 
# $$
# \text{'삶의만족도'} = \theta_0 + \theta_1 \times \text{'1인당GDP'}
# $$

# 지정된 선형 모델은 훈련 과정에서 
# **최선**의 $\theta_0$와 $\theta_1$를 학습해야 한다. 
# 예를 들어, 아래 이미지는 적절하지 않은 후보들을 보여준다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-18.png?raw=true" style="width:450px;"></div>

# 반면에 아래 이미지는 매우 적절한 $\theta_0$와 $\theta_1$를 보여준다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-19.png?raw=true" style="width:450px;"></div>

# 선형 모델은 주어진 데이터셋 활용하여 $\theta_0$와 $\theta_1$을 학습해 나간다.
# 학습 도중에 찾아진 $\theta_0$와 $\theta_1$의 적합도를
# 기준으로 학습되는 선형 모델의 성능을 반복적으로 평가하여
# 성능을 향상시키는 방향으로 학습을 유도한다. 

# ## 머신러닝 모델 훈련의 어려움

# 머신러닝 모델을 훈련할 때 경험할 수 있는 어려운 점들을 살펴본다.
# 기본적으로 훈련에 사용되는 훈련 데이터 또는 훈련 알고리즘 둘 중에 하나에 기인한다. 

# ### 데이터 문제

# **충분치 않은 양의 훈련 데이터**

# 간단한 문제라도 머신러닝 알고리즘을 제대로 학습시키려면 수천 개 이상의 데이터가 필요하다. 
# 이미지 분류, 음성 인식 등의 문제는 수백만 개가 필요할 수도 있다.
# 데이터가 부족하면 알고리즘 성능을 제대로 끌어올릴 수 없다.
# 일반적으로 데이터가 많을 수록 모델의 성능은 올라간다.
# 아래 이미지는 훈련에 사용되는 알고리즘에 상관없이 데이터셋의 크기가 클 수록
# 훈련된 모델의 성능이 올라감을 보여준다.

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-20.png?raw=true" style="width:450px;"></div>

# **대표성 없는 훈련 데이터**

# **샘플링 노이즈**<font size="2">sampling noise</font>는
# 데이터셋에 우연히 추가된 대표성이 없는 데이터를 가리킨다.
# 반면에 **샘플링 편향**<font size="2">sampling bias</font>은
# 표본 추출 방법이 잘못되어 데이터셋이 한 쪽으로 편향되어 대표성이 없는 것을 의미한다.
# 머신러닝 모델은 데이터셋의 편향성과 노이즈에 매우 민감하게 작동할 수 있다.
# 
# 예를 들어, 아래 이미지는 OECD 데이터셋에서 의도적으로 제거되었던 7개 국가를 추가해서 선형 모델을
# 새롭게 훈련시킨 경우(검정 실선)와 이전의 모델(파랑 점선)이 많이 달라진다.
# 이는 선형 모델이 훈련 데이터에 민감하게 반응하고, 따라서
# 삶의 만족도를 선형 모델을 이용하여 예측하는 것은 적절하지 않음을 잘 보여준다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-21.png?raw=true" style="width:600px;"></div>

# **저품질 데이터**

# 많은 오류와, 이상치, 노이즈를 포함하는 데이터셋이 주어지면
# 성능 좋은 머신러닝 모델을 얻을 수 없다. 
# 따라서 모델 훈련 이전에 먼저 
# **데이터 정제**<font size="2">data cleaning</font> 작업을 진행할 필요가 있다.
# 
# - 이상치인 경우엔 해당 데이터를 수정하거나 무시힌다.
# - 나이, 성별 등 데이터의 특성 중 일부가 누락된 경우 아래 방식에서 선택한다.
#     - 해당 특성 제외
#     - 해당 데이터 제외
#     - 누락된 특성값을 평균값 등으로 채우기
#     - 해당 특성을 넣은 경우와 뺀 경우 각각에 대해 모델 훈련 진행

# **특성 공학**

# 해결하려는 문제에 관련이 높은 특성을 찾아야 하며,
# 이를 **특성 공학**<font size="2">feature engineering</font>이라 하며,
# 보통 아래 세 가지 방식을 사용한다.
# 
# * 특성 선택<font size="2">feature selection</font>: 예측해야 할 값과 가장 연관성이 높은 특성을 선택한다.
# * 특성 추출<font size="2">feature extraction</font>: 특성을 조합하여 보다 유용한 새로운 특성을 생성한다.
# * 모델 학습에 유용하다고 판단되는 새로운 특성을 추가한다.

# ### 알고리즘 문제

# **과대 적합**

# 모델이 훈련 과정에서 훈련셋에 특화되어 일반화 성능이 떨어지는 현상이 **과대 적합**<font size="2">overfitting</font>이다.
# 아래 이미지는 훈련 데이터에 아주 민감하게 반응하는 모델을 보여준다.
# 과대 적합이 발생하는 이유는 다양하다. 
# 앞으로 과대 적합을 감소시키는 방법을 하나씩 살펴볼 것이다. 

# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/homl01-22.png?raw=true" style="width:550px;"></div>

# **과소 적합**

# 모델이 너무 단순해서 훈련셋을 제대로 대변하지 못하는 경우를
# **과소 적합**<font size="2">underfitting</font>이라 한다. 
# 아래 이미지는 포물선 형태의 2차 다항식 모델을 따르는 데이터셋을 선형 모델로 
# 구현한 결과를 보여준다.
# 앞으로 과소 적합을 해결하는 다양한 방식을 살펴볼 것이다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch04/homl04-06.png?raw=true" style="width:400px;"></div>

# ## 테스트와 검증

# 훈련된 모델의 성능을 평가하기 위해 **테스트셋**<font size="2">test set</font>을 활용한다.
# 테스트셋은 전체 데이터셋의 일부로 구성된다. 
# 보통 20% 정도를 테스트셋으로 남겨 둔 후 나머지 80% 만을 이용하여 모델을 훈련시킨다.
# 즉, 모델 훈련 과정동안 테스트셋에 포함된 데이터를 모델은 전혀 보지 못한다.
# 
# <div align="center"><img src="https://github.com/codingalzi/handson-ml3/blob/master/jupyter-book/imgs/ch01/train-test-validation.jpg?raw=true" style="width:400px;"></div>
# 
# 물론 데이터셋이 매우 크면 테스트셋 비율을 낮출 수 있다. 
# 예를 들어 천만 개의 데이터가 있으면 그중 1% 정도인 10만 개를 테스트용으로 사용해도 된다.
# 즉, 테스트셋은 훈련셋보다 상대적으로 많이 작아도 된다.
# 핵심은 테스트셋이 훈련에 전혀 사용되지 않아야 한다는 점이다.

# **일반화 오차 vs 훈련 오차**
# 
# 테스트를 통해 훈련된 모델의 일반화 성능을 확인한다.
# 일반화 성능은 훈련 과정 중에 접하지 않는 새로운 데이터에 대해 
# 모델의 예측이 얼마나 틀리는가를 계산한 
# **일반화 오차**<font size="2">generalization error</font>로 측정된다.
# 반면에 **훈련 오차**<font size="2">training error</font>는 
# 훈련 과정에 사용된 데이터에 대한 모델의 예측이 틀린 정도를 가리킨다. 
# 
# 일반화 오차는 물론 정확한 계산이 불가능하다. 
# 하지만 테스트셋을 이용하여 일반화 오차를 어느 정도 가늠할 수 있다.
# 그리고 훈련 오차에 비해 테스트셋을 활용해서 가늠한 일반화 오차가 높다면 
# 이는 모델이 과대 적합되었음을 의미한다.

# **교차 검증**

# 모델의 훈련 결과가 별로 좋지 않은 나오도록 하는 과소 적합은 기본적으로 잘못된 모델 선택에 기인한다.
# 예를 들어 앞서 본 것처럼 이차 다항식 모델 대신에 일차 방정식 모델을 
# 선택하는 경우가 그렇다.
# 반면에 과대 적합은 모델 선택 뿐만 아니라 데이터셋 자체와 훈련 방식에 의해서도 영향을 받는다.
# 
# 일반화 성능이 높은 모델을 훈련시키기 위해 많이 사용되는 방식 중 하나가
# **교차 검증**<font size="2">cross validation</font>이다. 
# 교차 검증은 훈련 데이터셋의 일부인 
# **검증 셋**<font size="2">validation set</font>을 
# 이용하여 훈련 과정중에 훈련 중인 모델의 일반화 성능을 검증하는 기법이며,
# 이를 통해 일반화 성능이 높은 모델을 훈련시키도록 유도한다.
# 교차 검증의 아이디어와 사용법은 앞으로 자세히 다룰 것이다. 
# 
# 또한 과대 적합이 발생했을 때 검증 셋을 활용하여 모델, 데이터셋, 훈련 방식 중 
# 어디에 문제가 있는지를 알아내는 다양한 기법이 알려져 있다.

# ## 연습문제

# 참고: [(실습) 한눈에 보는 머신러닝](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_ml_landscape.ipynb)
