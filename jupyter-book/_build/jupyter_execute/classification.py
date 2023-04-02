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
# * 오류 분석
# * 다중 레이블 분류
# * 다중 출력 분류

# **슬라이드**
# 
# 본문 내용을 요약한
# [슬라이드 1부](https://github.com/codingalzi/handson-ml3/raw/master/slides/slides-classification-1.pdf),
# [슬라이드 2부](https://github.com/codingalzi/handson-ml3/raw/master/slides/slides-classification-2.pdf),
# 다운로드할 수 있다.

# ## MNIST 데이터셋

# 미국 고등학생과 인구조사국 직원들이 손으로 쓴 70,000개의 숫자 이미지로 구성된 데이터셋이다.
# 사용된 0부터 9까지의 숫자는 모두 28x28 크기의 픽셀로 구성된 이미지 데이터이며,
# 2차원 어레이가 아닌 길이가 784(28x28)인 1차원 어레이로 제공된다.
# 
# 아래 이미지는 첫 손글씨 데이터를 28x28 모양으로 변환한 다음에 `pyplot.imshow()` 함수를 이용하여
# 그려진 것이며 숫자 5를 가리키는 것으로 보인다. 실제로도 타깃은 숫자 5이다. 

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/mnist_digit_5.jpg" width="250"/></div>

# 손글씨 이미지 첫 100개는 다음과 같다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-01.png" width="400"/></div>

# **문제 정의**
# 
# * 지도학습: 각 이미지가 담고 있는 숫자가 레이블로 지정됨.
# * 분류 모델: 이미지 데이터를 분석하여 0부터 9까지의 숫자로 분류
#     * 이미지 그림을 총 10개의 클래스로 분류하는 **다중 클래스 분류**<font size='2'>multiclass classification</font>.
#     * **다항 분류**<font size='2'>multinomial classification</font>라고도 불림
# 
# * 배치 또는 온라인 학습: 둘 다 가능
#   * **확률적 경사하강법**<font size='2'>stochastic gradient descent</font>(SGD):  배치와 온라인 학습 모두 지원
#   * 랜덤 포레스트 분류기: 배치 학습

# **훈련셋과 데이터셋**
# 
# 이미 6:1 의 비율로 훈련셋과 데이터셋으로 분류되어 있다.
# 모든 샘플은 무작위로 잘 섞여 있어서 교차 검증에 문제없이 사용될 수 있다.
# 
# * 훈련 세트(`X_train`): 앞쪽 60,000개 이미지
# * 테스트 세트(`X_test`): 나머지 10,000개의 이미지

# ## 이진 분류기 훈련

# 10개의 클래스로 분류하는 다중 클래스 모델을 훈련하기 전에 먼저
# 이미지 샘플이 숫자 5를 표현하는지 여부를 판단하는 이진 분류기를 훈련시킨다.
# 이를 통해 분류기의 기본 훈련 과정과 성능 평가 방법을 알아본다.
# 
# 이진 분류기의 훈련을 위해 타깃 데이터셋(`y_train_5`)을 새로 설정한다.
# 
# ```python
# y_train_5 = (y_train == '5')
# ```
# 
# * 1: 숫자 5를 가리키는 이미지 레이블
# * 0: 숫자 5 이외의 수를 가리키는 이미지 레이블

# 여기서 사용하는 모델은 `SGDClassifier` 클래스를 이용한다.
# `SGDClassifier` 분류기는 **확률적 경사하강법**<font size='2'>stochastic gradient descent</font> 분류기라고 불린다.
# 한 번에 하나씩 훈련 샘플을 이용하여 훈련한 후 파라미터를 조정하기에
# 매우 큰 데이터셋 처리에 효율적이며 온라인 학습에도 적합하다.
# 
# ```python
# sgd_clf = SGDClassifier(random_state=42)
# ```

# ## 분류기 성능 측정

# 분류기의 성능 측정 기준으로 보통 다음 세 가지를 사용한다.
# 
# * 정확도
# * 정밀도/재현율
# * ROC 곡선의 AUC

# ### 교차 검증 활용 정확도 측정

# 교차 검증을 이용하여 SGD 분류기의 성능을 측정한다. 
# 성능 측정 기준은 **정확도**다.
# 
# ```python
# cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# ```
# 
# 정확도가 95% 정도로 매우 좋은 결과로 보인다.
# 하지만 "무조건 5가 아니다" 라고 예측하는 모델도 90%의 정확도를 보인다.
# 특정 범주에 속하는 데이터가 상대적으로 너무 많을 경우 정확도는 신뢰하기 어려운 평가 기준임을 
# 잘 보여주는 사례다.

# ### 오차 행렬, 정밀도, 재현율

# #### 오차 행렬

# **오차 행렬**<font size="2">confusion matrix</font>은 클래스별 예측 결과를 정리한 행렬이다.
# 오차 행렬의 행은 실제 클래스를, 열은 예측된 클래스를 가리킨다.
# 예를 들어, 숫자 5의 이미지 샘플을 3으로 잘못 예측한 횟수를 알고 싶다면
# 6행 4열, 즉, (5, 3) 인덱스에 위치한 값을 확인해야 한다.
# 
# "숫자-5 감지기" 에 대한 오차 행렬은 아래와 같은 (2, 2) 모양의 2차원 (넘파이) 어레이로 생성된다.
# 이는 타깃 값이 0과 1 두 개의 값으로만 구성되기 때문이다.
# 
# ```python
# array([[53892,   687],
#        [ 1891,  3530]])
# ```

# :::{prf:example} 오차 행렬
# :label: exp_confusion_matrix
# 
# 아래 그림은 (2, 2) 모양의 오차 행렬을 맞춰야 하는 실제 타깃과 예측을 이용하여 보여준다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-02.png" width="400"/></div>
# :::

# #### 정밀도와 재현율

# **정밀도**
# 
# **정밀도**<font size="2">precision</font>는 양성 예측의 정확도를 가리킨다.
# 여기서는 숫자 5라고 예측된 값들 중에서 진짜로 5인 숫자들의 비율이다. 
# 
# $$\text{정밀도} = \frac{TP}{TP+FP} = \frac{3530}{3530 + 687} = 0.837$$

# **재현율**
# 
# 5인지 여부를 맞추는 모델의 예에서 확인했듯이 정밀도 하나만으로 분류기의 성능을 평가할 수는 없다.
# 보다 구체적인 이유는 숫자 5를 가리키는 이미지 중에 숫자 5라고 판명된 비율인 
# **재현율**<font size="2">recall</font> 함께 고려해야 하기 때문이다.
# 
# 재현율은 양성 샘플에 대한 정확도, 즉, 분류기가 정확하게 감지한 양성 샘플의 비율이며,
# **민감도**<font size="2">sensitivity</font> 또는 
# **참 양성 비율**<font size="2">true positive rate</font>로도 불린다.
# 
# $$\text{재현율} = \frac{TP}{TP+FN} = \frac{3530}{3530 + 1891} = 0.651$$

# **F<sub>1</sub> 점수**
# 
# 정밀도와 재현율의 조화 평균 F<sub>1</sub> 점수를 이용하여 분류기의 성능을 평가하기도 한다.
# 
# $$\text{F}_1 = \frac{2}{\frac{1}{\text{정밀도}} + \frac{1}{\text{재현율}}}$$
# 
# F<sub>1</sub> 점수가 높을 수록 분류기의 성능을 좋게 평가하지만
# 경우에 따라 재현율과 정밀도 둘 중의 하나에 높은 가중치를 두어야 할 때가 있다.

# **정밀도 vs. 재현율**
# 
# 모델 사용의 목적에 따라 정밀도와 재현율의 중요도가 다를 수 있다.
# 
# * 재현율이 보다 중요한 경우: 암 진단 기준
#   * 정밀도: 암이 있다고 진단된 경우 중에 실제로도 암이 있는 경우의 비율
#   * 재현율: 암으로 판정해야 하는 경우 중에서 양성 암진단으로 결론내린 경우의 비율
# 
# * 정밀도가 보다 중요한 경우: 아이에게 보여줄 안전한 동영상 선택 기준
#   * 정밀도: 안전하다고 판단된 동영상 중에서 실제로도 안전한 동영상의 비율
#   * 재현율: 실제로 좋은 동영상 중에서 좋은 동영상이라고 판정되는 동영상 비율

# #### 정밀도/재현율 트레이드오프

# 정밀도와 재현율은 상호 반비례 관계다. 
# 따라서 정밀도와 재현율 사이의 적절한 비율을 유지하는 분류기를 찾아야 한다. 
# 정밀도와 재현율의 비율은 결정 임곗값에 의해 결정된다.

# **결정 함수와 결정 임곗값**
# 
# **결정 함수**<font size="2">decision function</font>는 각 샘플에 대해 점수를 계산하며
# 이 점수를 기준으로 분류기가 해당 샘플의 양성 또는 음성 여부를 판단한다.
# 분류기는 주어진 샘플의 결정 함숫값이 지정된
# **결정 임계값**<font size="2">decision threshold</font>보다
# 같거나 크면 양성, 아니면 음성으로 판단한다.
# 
# 아래 예제는 다음 두 가지 사실을 잘 설명한다.
# 
# - 결정 임곗값의 위치에 따라 정밀도와 재현율이 서로 다른 방향으로 움직인다.
# - 결정 임곗값이 클 수록 분류기의 정밀도는 올라가지만 재현율은 떨어진다.
# - 결정 임곗값이 작을 수록 분류기의 정밀도는 내려가지만 재현율은 올라간다.

# :::{prf:example} 결정 임곗값, 정밀도, 재현율
# :label: exp_decision_threshold
# 
# 아래 그림에서 세 개의 화살표 (a), (b), (c)는 서로 다른 결정 임곗값을 가리키며, 
# 화살표 윗쪽에 위치한 정밀도와 재현율은 해당 결정 임곗값을 기준으로
# 주어진 샘플의 양성, 음성 여부를 판단할 경우의 정밀도와 재현율이다. 
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-03.png" width="500"/></div>
# 
# - 경우 (a)
#     - 정밀도 80%: 양성으로 예측된 5개의 샘플 중 정말로 5를 가리키는 샘플 4개, 아닌 샘플 1개
#     - 재현율 67%: 실제로 5인 샘플 총 6개 중에 5라고 판정된 샘플 4개
# - 경우 (b)
#     - 정밀도 75%: 양성으로 예측된 8개의 샘플 중 정말로 5를 가리키는 샘플 6개, 아닌 샘플 2개
#     - 재현율 100%: 실제로 5인 샘플 총 6개 중에 5라고 판정된 샘플 6개
# - 경우 (c)
#     - 정밀도 100%: 양성으로 예측된 3개의 샘플 중 정말로 5를 가리키는 샘플 3개, 아닌 샘플 0개
#     - 재현율 50%: 실제로 5인 샘플 총 6개 중에 5라고 판정된 샘플 3개
# :::

# **결정 임곗값, 정밀도, 재현율**
# 
# 아래 그래프는 `SGDClassifier` 모델을 "숫자-5 감지기"로 훈련시킨 결과를 이용한다.
# 그래프는 결정 임곗값을 변수로 해서 정밀도와 재현율의 변화를 보여준다.
# 결정 임곗값이 클 때 정밀도가 순간적으로 떨어질 수 있지만 결국엔 계속해서 상승한다.
# 
# - 정밀도는 90%, 재현율은 50% 정도가 되게 하는 경정 임곗값이 좋아보임.
# - `SGDClassifier` 는 0을 결정 임곗값으로 사용.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-04.png" width="500"/></div>

# **재현율 vs. 정밀도**
# 
# 위 그래프를 재현율 대 정밀도 그래프로 변환하면 다음과 같다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-05.png" width="400"/></div>

# ### ROC 곡선의 AUC

# __수신기 조작 특성__<font size="2">receiver operating characteristic</font>(ROC) 곡선을 활용하여 
# 이진 분류기의 성능을 측정할 수 잇다.
# **ROC** 곡선은 __거짓 양성 비율__<font size="2">false positive rate</font>(FPR)에 대한 
# __참 양성 비율__<font size="2">true positive rate</font>(TPR)의 관계를 나타내는 곡선이다.
# 
# * 참 양성 비율 = 재현율
# * 거짓 양성 비율: 원래 음성인 샘플 중에서 양성이라고 잘못 분류된 샘플들의 비율.
#     예를 들어, 5가 아닌 숫자중에서 5로 잘못 예측된 숫자의 비율
# 
#     $$\text{FPR} = \frac{FP}{FP+TN}$$

# **TPR vs. FPR**
# 
# 아래 그래프는 결정 임곗값에 따른 두 비율의 변화를 곡선으로 보여준다.
# 재현율(TPR)과 거짓 양성 비율(FPR) 사이에도 서로 상쇄하는 기능이 있다는 것을 확인할 수 있다.
# 이유는 재현율(TPR)을 높이고자 하면 거짓 양성 비율(FPR)도 함께 증가하기 때문이다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-06.png" width="400"/></div>

# **AUC와 분류기 성능**
# 
# 좋은 분류기는 재현율은 높으면서 거짓 양성 비율은 최대한 낮게 유지해야 한다.
# 즉, ROC 곡선이 y축에 최대한 근접하도록 해야 하며,
# 이는 ROC 곡선 아래의 면적, 즉 __AUC__(area under the curve)가 1에 가까울 수록 좋은 성능임을 의미한다. 

# :::{prf:example} SGD vs. 랜덤 포레스트
# :label: exp_SGD_RandomForest
# 
# MNIST 훈련 데이터셋으로 훈련된 `SGDClassifier` 와 `RandomForestClassifier`의 
# ROC 곡선을 함께 그리면 다음과 같이 랜덤 포레스트 모델의 AUC가 보다 1에 가깝다.
# 즉, 보다 성능이 좋다. 
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-07a.png" width="400"/></div>
# :::

# ## 다중 클래스 분류

# **다중 클래스 분류**<font size="2">multiclass classification</font>는 
# 세 개 이상의 클래스로 샘플을 분류하며,
# **다항 분류**<font size="2">multinomial classification</font>라고도 불린다.
# 예를 들어, MNIST 손글씨 숫자 분류의 경우 0부터 9까지 10개의 클래스로 분류해야 한다.

# 아래 모델은 다중 클래스 분류를 기본으로 지원한다.
# 
# * `LogisticRegression` 모델
# * `RandomForestClassifier` 모델
# * `GaussianNB` 모델

# 반면에 아래 모델은 이진 분류만 지원하지만, 필요에 따라
# 다중 클래스 분류를 지원하도록 작동한다.
# 
# * `SGDClassifier` 모델
# * `SVC` 모델

# **이진 분류기 활용**
# 
# 이진 분류기를 활용하여 다중 클래스 분류를 할 수 있으며
# 다음 두 방식 중 하나를 사용한다.
# 
# * 일대다: OvR(one-versus-the rest) 또는 OvA(one-versus-all)
# * 일대일: OvO(one-versus-one)

# **일대다 방식 활용**
# 
# "숫자-5 감지기" 모델에 적용했던 이진 분류 방식을 0부터 9까지 각 숫자에 대해 동일하게 적용한다.
# 즉, "숫자-0 감지지", "숫자-1 감지기" 등 총 10개의 이진 분류기를 훈련시킨다.
# 이후 10개의 이진 분류기가 각 샘플에 대해 계산한 결정 점수 중에서 가장 높은 점수를 주는
# 이진 분류기에 해당하는 클래스를 해당 훈련 샘플의 클래스로 결정한다.

# **일대일 방식 활용**
# 
# 각 샘플에 대해 가능한 모든 조합의 이진 분류기를 훈련한다.
# 이후 각 샘플에 대해 가장 많은 가장 많은 결투를 이긴,
# 즉, 해당 샘플에 대해 가장 많은 이진 분류기가 예측한 값을 해당 샘플의 예측값으로 결정한다.
# 
# 예를 들어, 모든 훈련 샘플에 대해 "0과 1 구별", "0과 2 구별", ..., "1과 2 구별", "1과 3 구별", ..., "8과 9 구별" 등 
# 총 45(= 9+8+...+1) 개의 이진 분류기를 훈련시킨다.
# 이제 각각의 이진 분류기는 분류기와 관련된 샘플만을 이용해서 훈련한다.
# 최종적으로 각 샘플에 대해 가장 많은 결투를 이긴 숫자가 해당 샘플의 예측값으로 사용된다.
# 예를 들어, 어떤 샘플에 대해 숫자 1이 9번의 결투를 모두 이기면 숫자 1을 해당 샘플의 예측값으로 지정한다.

# :::{prf:example} SVC 모델과 다중 클래스 분류
# :label: exp_SVC
# 
# 서포트 벡터 머신 분류기는 훈련 세트의 크기에 민감하다.
# 따라서 작은 훈련 세트에서 많은 분류기를 훈련시키는 OvO를 이용하여
# 다중 클래스 분류 훈련을 진행한다.
# 반면에 대부분의 이진 분류기는 OvR 전략을 선호한다.
# :::

# **일대일 또는 일대다 전략 선택**
# 
# 이진 분류기를 일대일 전략 또는 일대다 전략으로 지정해서 학습하도록 만들 수 있다.
# 
# * `OneVsOneClassifier` 클래스: 일대일 전략 지원
# * `OneVsRestClassifier` 클래스: 일대다 전략 지원
# 
# 아래 코드는 SVC 모델을 일대다(OvR) 전략으로 훈련시키는 과정을 보여준다.
# 
# ```python
# from sklearn.multiclass import OneVsRestClassifier
# 
# ovr_clf = OneVsRestClassifier(SVC(random_state=42))
# ovr_clf.fit(X_train[:2000], y_train[:2000])
# ```

# **다중 클래스 분류 모델 교차 검증**
# 
# MNIST의 경우 0부터 9까지 숫자가 균형 있게 분포되어 있어서 
# 정확도를 기준으로 교차 검증을 진행할 수 있다.
# 
# 예를 들어, 기본적으로 일대다(OvR) 전략을 사용해서 
# 다중 클래스 분류를 진행하는 `SGDClassifier` 모델에
# 다음과 같이 교차 검증을 적용할 수 있다.
# 
# ```python
# cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# ```

# ## 오류 분석

# 그리드 탐색, 랜덤 탐색 등을 이용한 모델 튜닝 과정을 실행하여 최선의 모델을 찾았다고 가정한다.
# 이제 오류 분석을 통해 모델을 좀 더 개선하고자 한다.

# **오차 행렬 활용**
# 
# 아래 왼쪽 이미지는 훈련된 분류 모델의 오차 행렬을 색상을 이용하여 표현한다.
# 대각선 상에 위치한 색상이 밝은 것은 분류가 대체로 잘 이루어졋음을 의미한다. 
# 다만 5번 행이 상대적으로 어두운데 이는 숫자 5의 분류 정확도가 상대적으로 낮기 때문이다.
# 
# 반면에 아래 오른쪽 이미지는 숫자별 비율로 변환하였다. 
# 즉, 행별로 퍼센티지의 합이 100이 되도록 정규화<font size="2">normalization</font> 하였다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-08.png" width="100%"/></div>

# **오차율 활용**
# 
# 위 오른쪽 이미지에서 많은 숫자가 8로 오인되었음을 알 수 있다.
# 실제로 올바르게 예측된 샘플을 제외한 다음에 행별로 오인된 숫자의 비율을 확인하면
# 아래 왼쪽 이미지와 같다.
# 8번 칸이 상대적으로 많이 밝으며, 이는 많은 숫자가 8로 오해되었다는 의미다. 
# 
# 아래 오른쪽 이미지는 칸 별 정규화 결과를 보여준다. 
# 예를 들어, 7로 오인된 이미지 중에 숫자 9 이미지의 비율이 56%임을 알 수 있다.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-09.png" width="100%"/></div>

# **개별 오류 확인**
# 
# 위 오른쪽 이미지에 의하면 5로 오인된 이미지 중에서 숫자 3 이미지의 비율이
# 34%로 가장 높다.
# 실제로 오차 행렬과 유사한 행렬을 3과 5에 대해 나타내면 다음과 같다.
# 
# * 음성: 3으로 판정
# * 양성: 5로 판정

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-10.png" width="500"/></div>

# **데이터 증식**
# 
# 사람 눈으로 보더라도 3과 5의 구분이 매우 어려울 수 있다.
# 여기서 사용한 SGD 분류 모델은 선형 회귀를 사용하기에 특히나 성능이 좋지 않다.
# 따라서 보다 좋은 성능의 모델을 사용할 수도 있지만
# 기본적으로 보다 많은 훈련 이미지가 필요하다.
# 새로운 이미지를 구할 수 있으면 좋겠지만 일반적으로 매우 어렵다.
# 반면에 기존의 이미지를 조금씩 회전하거나, 뒤집거나, 이동하는 방식 등으로
# 보다 많은 이미지를 훈련셋에 포함시킬 수 있다.
# 이런 방식을 **데이터 증식**<font size="2">data augmentation</font>이라 부른다.

# ## 다중 레이블 분류와 다중 출력 분류

# ### 다중 레이블 분류

# 다중 레이블 분류<font size='2'>multilabel classification</font>는 
# 각 훈련 샘플에 대해 여러 종류의 레이블 클래스와 관련된 분류 예측을 진행한다.
# 예를 들어, MNIST 손글씨 사진이 가리키는 숫자가 7 이상인지 여부와 홀수인지 여부를 
# 함께 판단하도록 훈련시키기 위해 아래 `y_multilabel`을 레이블로 지정할 수 있다.

# ```python
# y_train_large = (y_train >= '7')
# y_train_odd = (y_train.astype('int8') % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]
# ```

# 그러면 훈련된 모델은 아래 모양의 두 개의 값으로 구성된 어레이를 예측값으로 계산한다.
# 예를 들어, 숫자 5를 가리키는 이미지에 대한 예측값은 다음과 같다.
# 
# ```python
# array([[False,  True]])
# ```

# ### 다중 출력 분류

# 앞서 언급한 다중 레이블 분류는 7 이상인지 여부와 홀수인지 여부를 판단한다.
# 즉, 각각의 질문에 대해 이진 분류를 진행한다.  
# 반면에 **다중 출력 다중 클래스 분류**라고도 불리는 
# **다중 출력 분류**<font size='2'>multioutput classification</font>는
# 각 질문의 답에 사용되는 레이블이 다중 클래스 분류를 진행한다.
# 
# 예를 들어, 이미지에서 잡음을 제거하는 모델이 다중 출력 다중 클래스 분류기이다.
# 
# * 다중 출력: 각각의 픽셀에 대해 픽셀값 예측.
# * 다중 클래스: 픽셀값은 0부터 255 사이의 정수. 즉 266 개의 정수 중에 하나 선택.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-11.png" width="400"/></div>
# 
# 아래 이미지는 분류기가 예측한 이미지다. 즉, 각 픽셀에 대해 0~255 사이에서 예측된 값을 사용하였다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch03/homl03-12.png" width="200"/></div>

# ## 연습문제

# 참고: [(실습) 분류 1](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_classification_1.ipynb) 과
# [(실습) 분류 2](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/practices/practice_classification_2.ipynb)
