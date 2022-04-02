#!/usr/bin/env python
# coding: utf-8

# (ch:classification)=
# # 분류

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 분류](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_classification.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**
# 
# * MNIST 데이터셋
# * 이진 분류기 훈련
# * 분류기 성능 측정
# * 다중 클래스 분류
# * 에러 분석
# * 다중 레이블 분류
# * 다중 출력 분류

# ## MNIST

# **MNIST 데이터셋**

# * 미국 고등학생과 인구조사국 직원들이 손으로 쓴 70,000개의 숫자 이미지로 구성된 데이터셋
# 
# * 사용된 0부터 9까지의 숫자는 각각 28x28= 784크기의 픽셀로 구성된 이미지 데이터
#   * 2차원 어레이가 아닌 길이가 784인 1차원 어레이로 제공
# 
# * 레이블: 총 70,000개의 사진 샘플이 표현하는 값

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-01.png" width="500"/></div>

# **문제 정의**
# 
# * 지도학습: 각 이미지가 담고 있는 숫자가 레이블로 지정됨.
# 
# * 분류: 이미지 데이터를 분석하여 0부터 9까지의 숫자로 분류
#     * 이미지 그림을 총 10개의 클래스로 분류하는 __다중 클래스 분류__(multiclass classification)
#         __다항 분류__(multinomial classification)라고도 불림
# 
# * 배치 또는 온라인 학습: 둘 다 가능
#   * 모델에 따라 처리 방법이 다름
#       * 확률적 경사하강법(stochastic gradient descent, SGD):  배치와 온라인 학습 모두 지원
#       * 랜덤 포레스트 분류기: 배치 학습

# **훈련 셋과 데이터 셋 나누기**
# 
# * MNIST 데이터셋 이미 6:1 분류되어 있음\
# 
# * 무작위로 섞여 있음. 교차 검증 등에 문제없이 사용될 수 있음.
# 
# * 훈련 세트: 앞쪽 60,000개 이미지
# 
# * 테스트 세트: 나머지 10,000개의 이미지

# ## 이진 분류기 훈련

# **예제: 숫자 5-감지기**
# 
# * 이미지 샘플이 숫자 5를 표현하는지 여부를 판단하는 이진 분류기
# 
# * 모든 레이블을 0 또는 1로 수정해야 함
#     * 0: 숫자 5 이외의 수를 가리키는 이미지 레이블
#     * 1: 숫자 5를 가리키는 이미지 레이블
#     * 결과: `y_train_5`

# **SGD 분류기**

# * SGDClassifier(SGD 분류기)
#   * __확률적 경사 하강법__(stochastic gradient descent) 분류기라고 불림.
#   * 한 번에 하나씩 훈련 샘플 처리 후 파라미터 조정
#   * 매우 큰 데이터셋 처리에 효율적이며 온라인 학습에도 적합함.
# 
# * 훈련: `fit()` 메서드 호출
# 
#     ```python
#     from sklearn.linear_model import SGDClassifier
# 
#     sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
#     sgd_clf.fit(X_train, y_train_5)
#     ```

# ## 분류기 성능 측정

# **성능 측정 세가지 방법**
# 
# * 교차 검증을 활용한 정확도 측정
# * 정밀도/재현율 측정
# * AUC 측정

# ### 교차 검증을 사용한 정확도 측정

# * 2장에서 배운 교차검증 기술을 이용하여 SGD 분류기의 성능을 측정
# 
# * 성능 측정 기준: 정확도
# 
# * 예제: 숫자 5를 표현하는 이미지를 정확하게 예측한 비율.
#     `cross_val_score` 모델의 `scoring="accuracy"` 키워드 인자 지정 <br><br>
# 
#     ```python
#     from sklearn.model_selection import cross_val_score
# 
#     cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#     ```

# ### 오차 행렬, 정밀도, 재현율

# **오차 행렬**

# * 교차 검증 결과가 95% 이상으로 매우 우수한 것으로 나옴.
#     * 하지만 무조건 '5 아님'이라고 찍는 분류기도 90%의 정확도를 보임.
#     * 훈련 세트의 샘플이 불균형적으로 구성되었다면, 
#         정확도를 분류기의 성능 측정 기준으로 사용하는 것은 피해야 함
# 
# * 오차 행렬을 조사하여 분류기의 성능을 다르게 평가할 수 있음
# 
# * __오차 행렬__: 클래스별 예측 결과를 정리한 행렬
# 
# * 오차 행렬의 행은 실제 클래스를, 열은 예측된 클래스를 가리킴
#     * 클래스 A의 샘플이 클래스 B의 샘플로 분류된 횟수를 알고자 하면 A행 B열의 값을 확인 
# 
# * 예제: 숫자 5의 이미지 샘플을 3으로 잘못 예측한 횟수를 알고 싶다면?
#     * 6행 4열, 즉, (6,4) 인덱스에 위치한 값을 확인 (0부터 9까지의 숫자임에 주의)
# 
# * 예제: '숫자 5-감지기'에 대한 오차 행렬은 `(2, 2)` 모양의 2차원 (넘파이) 어레이로 생성됨.
#     * 레이블의 값이 0과 1 두 개의 값으로 구성되기 때문
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-02.png" width="500"/></div>

# **정밀도(precision)**

# * 책 134쪽의 오차 행렬
# 
#     ```
#     array([[53057, 1522],
#            [ 1325, 4096]])
#     ```
# 
# * 양성 예측의 정확도
# 
# * 여기서는 숫자 5라고 예측된 값들 중에서 진짜로 5인 숫자들의 비율
# 
#     $$\text{precision} = \frac{TP}{TP+FP} = \frac{4096}{4096 + 1522} = 0.729$$

# **재현율(recall)**

# * 정밀도 하나만으로 분류기의 성능을 평가할 수는 없음
#     * 숫자 5를 가리키는 이미지 중에 숫자 5라고 판명한 비율인 __재현율__을 고려하지 않기 때문
# 
# * 양성 샘플에 대한 정확도, 즉, 분류기가 정확하게 감지한 양성 샘플의 비율
# 
# * 재현율을 __민감도__(sensitivity) 또는 __참 양성 비율__(true positive rate)로도 부름
# 
#     $$\text{recall} = \frac{TP}{TP+FN} = \frac{4096}{4096 + 1325} = 0.756$$

# **F<sub>1</sub> 점수**

# * 정밀도와 재현율의 조화 평균인 F<sub>1</sub> 점수를 이용하여 분류기의 성능을 평가하기도 함.
# 
# $$\text{F}_1 = \frac{2}{\frac{1}{\text{정밀도}} + \frac{1}{\text{재현율}}}$$
# 
# * F<sub>1</sub> 점수가 높을 수록 분류기의 성능을 좋게 평가하지만
#   경우에 따라 재현율과 정밀도 둘 중의 하나에 높은 가중치를 두어야 할 때가 있음.
#     * 앞서 정의된 F<sub>1</sub> 점수는 재현율과 정밀도의 중요도가 동일하다고 가정하였음.

# **정밀도 vs. 재현율**

# * 모델 사용의 목적에 따라 정밀도와 재현율의 중요도가 다를 수 있음.
# 
# * 재현율이 보다 중요한 경우: 암 진단 기준
#   * 정밀도: 양성으로 진단된 경우 중에 실제로도 양성인 경우의 비율
#   * 재현율: 실제로 양성인 경우 중에서 양성으로 진단하는 경우의 비율
# 
# * 정밀도가 보다 중요한 경우: 아이에게 보여줄 안전한 동영상 선택 기준
#   * 정밀도: 안전하다고 판단된 동영상 중에서 실제로도 안전한 동영상의 비율
#   * 재현율: 실제로 좋은 동영상 중에서 좋은 동영상이라고 판단되는 동영상 비율

# **정밀도/재현율 트레이드오프**

# * 정밀도와 재현율은 상호 반비례 관계임.
# 
# * 정밀도와 재현율 사이의 적절한 비율을 유지하는 분류기를 찾아야 함.
#     
# 
# * 적절한 __결정 임곗값__을 지정해야 함.

# **결정 함수와 결정 임곗값**

# * __결정 함수__(decision function): 분류기가 각 샘플의 점수를 계산할 때 사용
# 
# * __결정 임계값__(decision threshold): 결정 함수가 양성 클래스 또는 음성 클래스로 분류하는 데에 사용하는 기준값
# 
# * 임곗값이 커질 수록 정밀도는 올라가지만 재현율은 떨어짐.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-03.png" width="700"/></div>

# **임곗값, 재현율, 정밀도**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-04.png" width="700"/></div>

# 재현율 대 정밀도
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-05.png" width="700"/></div>

# ### ROC 곡선과 AUC 점수

# * __수신기 조작 특성__(receiver operating characteristic, ROC) 곡선을 활용하여 이진 분류기의 성능 측정 가능
# 
# * ROC 곡선: __거짓 양성 비율__(false positive rate, FPR)에 대한 __참 양성 비율__(true positive rate, TPR)의 관계를 나타내는 곡선
#     * 결정 임곗값에 따른 두 비율의 변화를 곡선으로 보여줌.
#     * 참 양성 비율: 재현율
# 
# * 거짓 양성 비율: 원래 음성인 샘플 중에서 양성이라고 잘못 분류된 샘플들의 비율.
#     예를 들어, 5가 아닌 숫자중에서 5로 잘못 예측된 숫자의 비율
# 
#     $$\text{FPR} = \frac{FP}{FP+TN}$$

# **TPR vs. FPR**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-06.png" width="700"/></div>

# **AUC와 분류기 성능**

# * 재현율(TPR)과 거짓 양성 비율(FPR) 사이에도 서로 상쇄하는 기능이 있다는 것을 확인 가능
#     * 재현율(TPR)을 높이고자 하면 거짓 양성 비율(FPR)도 함께 증가
# 
# * 따라서 좋은 분류기는 재현율은 높으면서 거짓 양성 비율은 최대한 낮게 유지해야함
# 
# * ROC 곡선이 y축에 최대한 근접하는 결과가 나오도록 해야함.
# 
# * __AUC__(ROC 곡선 아래의 면적)가 1에 가까울 수록 성능이 좋은 분류기로 평가됨.

# **SGD와 랜덤 포레스트의 AUC 비교**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-07.png" width="700"/></div>

# ## 3.4 다중 클래스 분류

# ### 다중 클래스 분류기(multiclass classifier)

# * 세 개 이상의 클래스로 샘플을 분류하는 예측기
# 
# * 다항 분류기(multinomial classifier)라고도 부름
# 
# * 예를 들어, 손글씨 숫자 분류의 경우 0부터 9까지 10개의 클래스로 분류해야 함

# #### 다중 클래스 분류 지원 분류기
# 
# * SGD 분류기
# * 랜덤 포레스트 분류기
# * 나이브 베이즈(naive Bayes) 분류기

# #### 이진 분류만 지원하는 분류기
# 
# * 로지스틱 회귀
# * 서포트 벡터 머신
#     * 사이킷런의 `SVC()` 모델은 다중 클래스 분류도 특별한 기법을 적용하여 지원함.

# ### 이진 분류기 활용

# * 이진 분류기를 활용하여 다중 클래스 분류 가능
#     * 일대다(OvR 또는 OvA)
#     * 일대일(OvO)
# 
# * 일대다 방식 활용 예제
#     * 숫자 5 예측하기에서 사용했던 이진 분류 방식을 동일하게 모든 숫자에 대해서 실행
#     * 각 샘플에 대해 총 10번 각기 다른 이진 분류기를 실행
#     * 이후 각 분류기의 결정 점수 중에서 가장 높은 점수를 받은 클래스를 해당 샘플의 클래스로 선택
# 
# * 일대일 방식 활용 예제
#     * 조합 가능한 모든 클래스 일대일 분류 방식을 진행하여 가장 많은 결투(duell)를 이긴 숫자를 선택
#     * MNIST의 경우, 0과 1 구별, 0과 2 구별, ..., 1과 2 구별, 1과 3 구별, ..., 8과 9 구별 등 
#         (9+8+...+1 = 45)개의 결투를 판별하는 45개의 분류기 활용.
#         단, 각 결투에 해당되는 데이터 샘플만 훈련에 사용됨.
#     * 각각의 훈련 샘플에 대해 가장 많은 결투를 이긴 숫자의 클래스를 예측값으로 사용함.
#         예를 들어, 숫자 1이 9번의 결투를 모두 이기면 숫자 1을 예측값으로 지정함.

# 일대다 방식 vs. 일대일 방식
# 
# #### <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/ovo_ova.png" width="600"/></div>
# 
# <그림 출처: [SVM with the mlr package](https://www.r-bloggers.com/2019/10/support-vector-machines-with-the-mlr-package/)>

# #### 예제: 서포트 벡터 머신

# * 훈련 세트의 크기에 민감하여 작은 훈련 세트에서 많은 분류기를 훈련시키는 쪽이 훨씬 빠름. 
#     따라서 다중 클래스 분류에 일대일 전략을 사용함.
# 
# * 대부분의 이진 분류기는 일대다 전략을 선호

# #### 일대일 또는 일대다 전략 선택

# * 이진 분류기를 일대일 전략 또는 일대다 전략으로 지정해서 학습하도록 만들 수 있음.
# 
# * 사이킷런의 경우: `OneVsOneClassifier` 또는 `OneVsRestClassifier` 사용
# 
# * 예를 들어, SVC 모델을 일대다 전략으로 훈련시키려면 `OneVsRestClassifier` 활용
# 
#     ```python
#     from sklearn.multiclass import OneVsRestClassifier
#     ovr_clf = OneVsRestClassifier(SVC())
#     ovr_clf.fi(X_train, y_train)
#     ```

# #### 다중 클래스 지원 분류기

# * `SGDClassifier` 또는 `RandomForestClassifier`는 다중 클래스 분류를 직접 지원함.
# 
# * 따라서 사이킷런의 OvR, OvO 등을 적용할 필요 없음

# #### 다중 클래스 분류기 성능 측정

# * 다중 클래스 분류기의 성능 평가는 교차검증을 이용하여 정확도를 측정
# 
# * MNIST의 경우 0부터 9까지 숫자가 균형 있게 분포되어 있어서 데이터 불균형의 문제가 발생하지 않음.

# ## 3.5 에러 분석

# 가능성이 높은 모델을 하나 찾았을 때 에러 분석을 통해 모델의 성능을 향상시킬 방법을 찾아볼 수 있음.

# ### 오차 행렬 활용

# * 손글씨 클래스 분류 모델의 오차 행렬을 이미지로 표현 가능
# * 대체로 잘 분류됨: 대각선이 밝음.
# * 5행은 좀 어두움. 숫자 5의 분류 정확도가 상대적으로 낮음

# #### <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-08.png" width="400"/></div>

# #### 오차율 이미지
# 
# * 8행이 전반적으로 어두움. 즉, 8은 잘 분류되었음.
# * (3, 5)와 (5,3)의 위치가 상대적으로 밝음. 즉, 3과 5가 서로 많이 혼동됨.

# #### <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-09.png" width="400"/></div>

# * 3과 5의 오차행렬 그려보기
#     * 음성: 3으로 판정
#     * 양성: 5로 판정

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-10.png" width="400"/></div>

# * 3과 5의 구분이 어려운 이유
#     * 선형 모델인 SGD 분류기를 사용했기 때문
#     * SGD 모델은 단순히 픽셀 강도에만 의존함.
# 
# * 이미지 분류기의 한계
#     * 이미지의 위치나 회전 방향에 민감함
#     * 이미지를 중앙에 위치시키고 회전되지 않도록 전처리하거나, 8은 동그라미가 두 개 있다는 등
#         각 숫자의 특성을 추가하면 더 좋은 성능의 모델 구현 가능함.

# ### 다중 클래스 분류 일반화

# * 다중 레이블 분류(multilabel classification)
# 
# * 다중 출력 분류(multioutput classification)

# 사이킷런의 다중 클래스와 다중 출력 알고리즘

# <div align="center"><img src="https://github.com/codingalzi/handson-ml/blob/master/slides/images/ch03/multi_org_chart.png?raw=true" width="900"/></div>
# 
# <이미지 출처: [사이킷런: 다중 클래스와 다중 출력 알고리즘](https://scikit-learn.org/stable/modules/multiclass.html)>

# ## 3.6 다중 레이블 분류

# * 샘플마다 여러 개의 클래스 출력
# 
# * 예제: 얼굴 인식 분류기
#     * 한 사진에 여러 사람이 포함된 경우, 인식된 사람마다 하나씩 꼬리표(tag)를 붙여야 함. 
#     * 앨리스, 밥, 찰리의 포함여부를 확인 할 때: 밥이 없는 경우 `[True, False, True]` 출력
# 
# * 다중 레이블 분류기를 평가하는 방법은 다양함
#     * 예를들어, 각 레이블의 F<sub>1</sub> 점수를 
#         구하고 레이블에 대한 가중치를 적용한 
#         평균 점수 계산
#     * 가중치 예제: 타깃 레이블에 속한 샘플 수를 가중치로 사용 가능. 즉, 샘플 수가 많은 클래스의 가중치를 보다 크게 줄 수 있음.

# ## 3.7 다중 출력 분류

# * 다중 출력 다중 클래스 분류라고도 불림
# 
# * 다중 레이블 분류에서 한 레이블이 이진 클래스가 아닌 다중 클래스를 대상으로 예측하는 분류
# 
# * 예제: 이미지에서 잡음을 제거하는 시스템
#     * 다중 레이블: 각각의 픽셀에 대해 레이블 예측해야 함.
#     * 다중 클래스: 각각의 픽셀에서 예측하는 레이블이 0부터 255 중에 하나임.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-11.png" width="400"/></div>
# 
# * 아래 사진: 분류기가 예측한 이미지
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-12.png" width="130"/></div>

# In[ ]:




