#!/usr/bin/env python
# coding: utf-8

# (ch:svm)=
# # 서포트 벡터 머신

# **감사의 글**
# 
# 자료를 공개한 저자 오렐리앙 제롱과 강의자료를 지원한 한빛아카데미에게 진심어린 감사를 전합니다.

# **소스코드**
# 
# 본문 내용의 일부를 파이썬으로 구현한 내용은 
# [(구글코랩) 서포트 벡터 머신](https://colab.research.google.com/github/codingalzi/handson-ml3/blob/master/notebooks/code_svm.ipynb)에서 
# 확인할 수 있다.

# **주요 내용**
# 
# * 선형 SVM 분류
# * 비선형 SVM 분류
# * SVM 회귀
# * SVM 이론

# **목표**
# 
# 서포트 벡터 머신의 주요 개념, 사용법, 작동법을 알아본다. 

# ## 선형 SVM 분류

# 선형 **서포트 벡터 머신**<font size="2">support vector machine</font>(SVM)은
# 두 클래스 사이를 최대한으로 경계 도로를 최대한 넓게 잡으려고 시도한다. 
# 이때 두 클래스 사이에 놓을 수 있는 결정 경계 도로의 폭의 **마진**<font size='2'>margin</font>이라 하며,
# 마진을 최대로 하는 분류가 **큰 마진 분류**<font size='2'>large margin classication</font>이다.
# 
# 아래 그림은 붓꽃 데이터셋을 대상으로 해서 선형 분류와 큰 마진 분류의 차이점을 보여준다.
# 선형 분류(왼쪽 그래프)의 경우 두 클래스를 분류하기만 해도 되는 반면에 큰 마진 분류(오른쪽 그래프)의 
# 결정 경계(검은 실선)는 두 클래스와 거리를 최대한 크게 두려는 방향으로 정해진다.
# 즉, 마진은 가능한 최대로 유지하려 한다. 
# 큰 마진 분류의 결정 경계는 결정 경계 도로의 가장자리에 위치한
# **서포트 벡터**<font size='2'>support vector</font>에만 의존하며 다른 데이터와는 전혀 상관 없다.
# 아래 오른쪽 그래프에서 서포트 벡터는 동그라미로 감싸져 있다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-01.png" width="700"/></div>
# 

# :::{admonition} 스케일링과 마진
# :class: info
# 
# 특성의 스케일을 조정하면 결정 경계가 훨씬 좋아진다. 
# 두 특성의 스케일에 차이가 많이 나는 경우(아래 왼쪽 그래프) 보다
# 표준화된 특성을 사용할 때(아래 오른쪽 그래프) 훨씬 좋은 결정 경계가 찾아진다. 
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-02.png" width="700"/></div>
# :::

# ### 하드 마진 분류

# 모든 훈련 샘플이 도로 바깥쪽에 올바르게 분류되도록 하는 마진 분류가
# **하드 마진 분류**<font size='2'>hard margin classification</font>이다. 
# 하지만 두 클래스가 선형적으로 구분되는 경우에만 적용 가능하다. 
# 
# 또한 이상치에 매우 민감하다.
# 하나의 이상치가 추가되면 선형 분류가 불가능하거나(아래 왼편 그래프)
# 일반화가 매우 어려운 분류 모델(아래 오른편 그래프)이 얻어질 수 있다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-03.png" width="700"/></div>

# ### 소프트 마진 분류

# **소프트 마진 분류**<font size='2'>soft margin classification</font>는 어느 정도의 마진 오류를 허용하면서
# 결정 경계 도로의 폭을 최대로 하는 방향으로 유도한다.
# **마진 오류**<font size='2'>margin violations</font>는 결정 경계 도로 상에 또는 결정 경계를 넘어 해당 클래스 반대편에 위치하는 샘플을 의미한다. 
# 
# 예를 들어 꽃잎 길이와 너비 기준으로 붓꽃의 버지니카와 버시컬러 품종을 하드 마진 분류하기는 불가능하며,
# 아래 그래프에서처럼 어느 정도의 마진 오류를 허용해야 한다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-03b.png" width="400"/></div>

# **`LinearSVC` 클래스**
# 
# 사이킷런의 `LinearSVC` 클래스는 선형 SVM 분류기를 생성한다.
# 
# ```python
# LinearSVC(C=1, random_state=42)
# ```
# 
# `C` 는 규제 강조를 지정하는 하이퍼파라미터이며 클 수록 적은 규제를 의미한다. 
# `C` 가 너무 작으면(아래 왼편 그래프) 마진 오류를 너무 많이 허용하는 과소 적합이
# 발생하며, `C` 를 키우면(아래 오른편 그래프) 결정 경계 도로 폭이 좁아진다.
# 여기서는 `C=100` 이 일반화 성능이 좋은 모델을 유도하는 것으로 보인다.
# 또한 `C=float("inf")`로 지정하면 하드 마진 분류 모델이 된다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-04.png" width="800"/></div>

# :::{admonition} 선형 SVM 지원 모델
# :class: info
# 
# `LinearSVC` 모델은 대용량 훈련 데이터셋을 이용해서도 빠르게 학습한다. 
# 이외에 `SVC` 모델과 `SGDClassifier` 모델도 선형 SVM 분류 모델로 활용될 수 있다.
# 
# * `SVC` 클래스 활용
# 
#     ```python
#     SVC(kernel="linear", C=1)
#     ```
# 
# * `SGDClassifier` 클래스 활용
#     
#     ```python
#     SGDClassifier(loss="hinge", alpha=1/(m*C))
#     ```
# 
# hinge 손실 함수는 어긋난 예측 정도에 비례하여 손실값이 선형적으로 커진다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-06c.png" width="400"/></div>
# :::

# ## 비선형 SVM 분류

# 선형적으로 구분되지 못하는 데이터셋을 대상으로 분류 모델을 훈련시키는 두 가지 방식을 소개한다.
# 
# * 방식 1: 특성 추가 + 선형 SVC
#     * 다항 특성 활용: 다항 특성을 추가한 후 선형 SVC 적용
#     * 유사도 특성 활용: 유사도 특성을 추가한 후 선형 SVC 적용
# 
# * 방식 2: `SVC` + 커널 트릭
#     * 커널 트릭: 새로운 특성을 실제로 추가하지 않으면서 동일한 결과를 유도하는 방식
#     * 예제 1: 다항 커널
#     * 예제 2: 가우시안 RBF(방사 기저 함수) 커널

# **다항 특성 추가 + 선형 SVM**
# 
# {numref}`%s절 <sec:poly_reg>`에서 설명한 다항 회귀 기법에서 다항 특성을 추가한 후에 
# 선형 회귀를 적용한 방식과 동일하다. 
# 아래 그래프는 특성 $x_1$ 하나만 갖는 데이터셋에 특성 $x_1^2$을 추가한 후 선형 회귀 모델을
# 적용한 결과를 보여준다.
# 
# $$\hat y = \theta_0 + \theta_1\, x_1 + \theta_2\, x_1^{2}$$
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch04/homl04-07.png" width="500"/></div>

# 동일한 아이디어를 특성 $x_1$ 하나만 갖는 데이터셋(아래 왼편 그래프)에 적용하면 
# 비선형 SVM 모델(아래 오른편 그래프)을 얻게 된다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-05.png" width="700"/></div>

# :::{admonition} 2차 다항 특성 추가 후 선형 SVM 분류 모델 훈련
# :class: info
# 
# 아래 동영상은 두 개의 특성을 갖는 데이터셋에 2차 다항 특성을 추가한 후에 선형 SVM 분류 모델을
# 적용하는 과정을 보여준다. 
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/svm-poly-kernel.gif" width="600"/></div>
# 
# <동영상 출처: [SVM with polynomial kernel visualization](https://www.youtube.com/watch?v=OdlNM96sHio)>
# 
# 참고로 3차원 상에서의 선형 방정식의 그래프는 평면으로 그려진다. 
# 
# $$z = \frac{3}{5} x + \frac{1}{5}y + 5 \quad\Longleftrightarrow\quad 3x + y - 5z + 25 = 0$$
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-06d.png" width="500"/></div>
# 
# <그림 출처: [지오지브라(GeoGebra)](https://www.geogebra.org/3d)>
# :::

# :::{prf:example} moons 데이터셋
# :label: exp:moons_dataset
# 
# moons 데이터셋은 마주보는 두 개의 반원 모양의 클래스로 구분되는 데이터셋을 가리킨다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-06.png" width="500"/></div>
# 
# 위 데이터셋에 선형 SVM 분류 모데를 적용하기 위해 먼저 3차 항에 해당하는 특성을 추가하면
# 비선형 분류 모델을 얻게 된다.
# 
# ```python
# # 3차 항까지 추가
# polynomial_svm_clf = make_pipeline(
#     PolynomialFeatures(degree=3),
#     StandardScaler(),
#     LinearSVC(C=10, max_iter=10_000, random_state=42)
# )
# ```
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-07.png" width="500"/></div>
# :::

# ### 다항 커널

# 다항 특성을 추가하는 기법은 그만큼 비용을 지불해야 한다.
# 특히 축가해야 하는 특성이 많다면 시간과 메모리 사용 비용이 엄청날 수 있다.
# 반면에 **커널 트릭**<font size='2'>kernel trick</font>을 사용하면
# 다항 특성을 실제로는 추가하지 않지만 추가한 경우와 동일한 결과를 만들어 낼 수 있다.
# 다만 이것은 SVM을 적용하는 경우에만 해당한다.
# 이와 달리 다항 특성을 추가하는 기법은 어떤 모델과도 함께 사용될 수 있다.

# 아래 두 그래프는 커널 기법을 사용하는 SVC 모델을 moons 데이터셋에 대해 훈련시킨 결과를 보여준다.
# 
# ```python
# poly_kernel_svm_clf = make_pipeline(StandardScaler(),
#                                     SVC(kernel="poly", degree=3, coef0=1, C=5))
# ```
# 
# 위 코드는 3차 다항 커널을 적용한 모델이며 아래 왼편 그래프와 같은 분류 모델을 학습한다.
# 반면에 아래 오른편 그래프는 10차 다항 커널을 적용한 모델이다. 
# `coef0` 하이퍼파라미터는 고차항의 중요도를 지정하며, 아래 그래프에서는 $r$ 이 동일한 
# 하이퍼파라미터를 가리킨다.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-09.png" width="800"/></div>

# :::{admonition} 하이퍼파라미터 이해의 중요성
# :class: tip
# 
# 다항 커널 모델이 과대 적합이면 차수를 줄여야 하고, 과소 적합이면 차수를 늘려야 한다.
# 적절한 하이퍼파라미터는 그리드 탐색 등을 이용하여 찾으면 되지만,
# 그럼에도 불구하고 하이퍼파라미터의 의미를 잘 알고 있으면 탐색 구간을 줄일 수 있다.
# :::

# ### 유사도 특성

# **유사도 함수**
# 
# * 유사도 함수: __랜드마크__(landmark)라는 특정 샘플과 각 샘플 사이의 유사도(similarity)를 측정하는 함수
# 
# * 유사도 함수 예제: __가우시안 방사 기저 함수__(RBF, radial basis function)
# 
#     $$
#     \phi(\mathbf x, \ell) = \exp(-\gamma\, \lVert \mathbf x - \ell \lVert^2)
#     $$
# 
#     * $\ell$: 랜드마크
#     * $\gamma$: 랜드마크에서 멀어질 수록 0에 수렴하는 속도를 조절함
#     * $\gamma$ 값이 클수록 가까운 샘플 선호, 즉 샘플들 사이의 영향을 보다 적게 고려하여
#         모델의 자유도를 높이게 되어 과대적합 위험 커짐.

# * 예제
# 
# $$
# \exp(-5\, \lVert \mathbf x - 1 \lVert^2) \qquad\qquad\qquad \exp(-100\, \lVert \mathbf x - 1 \lVert^2)
# $$
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-08b.png" width="1200"/></div>
# 
# <그림 출처: [데스모스(desmos)](https://www.desmos.com/calculator?lang=ko)>

# **유사도 특성 추가 + 선형 SVC**
# 
# * 모든 샘플을 랜드마크로 지정 후 각 랜드마크에 대한 유사도를 새로운 특성으로 추가하는 방식이 가장 간단함.
# 
# * ($n$ 개의 특성을 가진 $m$ 개의 샘플) $\Rightarrow$ ($n + m$ 개의 특성을 가진 $m$ 개의 샘플)
# 
# * 장점: 차원이 커지면서 선형적으로 구분될 가능성이 높아짐.
# 
# * 단점: 훈련 세트가 매우 클 경우 동일한 크기의 아주 많은 특성이 생성됨.
# 
# * 예제
#     * 랜드마크: -2와 1
#     * $x_2$와 $x_3$: 각각 -2와 1에 대한 가우시안 RBF 함수로 계산한 유사도 특성
#     * 화살표가 가리키는 점: $\mathbf x = -1$
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-08.png" width="800"/></div>

# ### 가우시안 RBF 커널

# * SVM 모델을 훈련시킬 때 유사도 특성을 실제로는 추가 하지 않으면서 수학적으로는 추가한 효과를 내는 성질 이용
# 
# ```python
# rbf_kernel_svm_clf = Pipeline([
#         ("scaler", StandardScaler()),
#         ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=0.001)) ])
# ```

# **SVC + RBF 커널 예제: moons 데이터셋**
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-10.png" width="600"/></div>
# 
# 
# |      | 상단 그래프      | 하단 그래프    |
# | :--- | :------------- | :------------- |
# | gamma | 랜드마크에 조금 집중 | 랜드마크에 많이 집중 |
# 
# |      | 왼편 그래프&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 오른편 그래프    |
# | :--- | :------------- | :------------- |
# | C | 규제 많이 | 규제 적게 |

# **추천 커널**
# 
# * `SVC`의 `kernel` 기본값은 `"rbf"` => 대부분의 경우 이 커널이 잘 맞음
# 
# * 선형 모델이 예상되는 경우 `SVC`의 `"linear"` 커널을 사용할 수 있음
#     하지만 훈련 세트가 크거나 특성이 아주 많을 경우 `LinearSVC`가 빠름
# 
# * 시간과 컴퓨팅 성능이 허락한다면 교차 검증, 그리드 탐색을 이용하여 적절한 커널을 찾아볼 수 있음
# 
# * 훈련 세트에 특화된 커널이 알려져 있다면 해당 커널을 사용

# ### 계산 복잡도

# 분류기|시간 복잡도(m 샘플 수, n 특성 수)|외부 메모리 학습|스케일 조정|커널 트릭|다중 클래스 분류
# ----|-----|-----|-----|-----|-----
# LinearSVC | $O(m \times n)$ | 미지원 | 필요 | 미지원 | OvR 기본
# SGDClassifier | $O(m \times n)$ | 지원 | 필요 | 미지원 | 지원
# SVC | $O(m^2 \times n) \sim O(m^3 \times n)$ | 미지원 | 필요 | 지원 | OvR 기본

# ## SVM 회귀

# **SVM 분류 vs. SVM 회귀**
# 
# * SVM 분류 
#     - 목표: 마진 오류 발생 정도를 조절(`C` 이용)하면서 두 클래스 사이의 도로폭을 최대한 넓게 하기
#     - 마진 오류: 도로 위에 위치한 샘플
# 
# * SVM 회귀 
#     - 목표: 마진 오류 발생 정도를 조절(`C` 이용)하면서 지정된 폭의 도로 안에 가능한 많은 샘플 포함하기
#     - 마진 오류: 도로 밖에 위치한 샘플
#     - 참고: [MathWorks: SVM 회귀 이해하기](https://kr.mathworks.com/help/stats/understanding-support-vector-machine-regression.html)

# ### 선형 SVM 회귀

# * 선형 회귀 모델을 SVM을 이용하여 구현

# * 예제: LinearSVR 활용. `epsilon`은 도로폭 결정
# 
#     ```python
#     from sklearn.svm import LinearSVR
#     svm_reg = LinearSVR(epsilon=1.5)
#     ```    

# * 마진 안, 즉 결정 경계 도로 위에 포함되는 샘플를 추가해도 예측에 영향 주지 않음. 즉 `epsilon`에 둔감함.

# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-11.png" width="600"/></div>

# ### 비선형 SVM 회귀

# * SVC와 동일한 커널 트릭을 활용하여 비선형 회귀 모델 구현
# 
# * 예제: SVR + 다항 커널
# 
#     ```python
#     # SVR + 다항 커널
#     from sklearn.svm import SVR
# 
#     svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
#     ```
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-12.png" width="800"/></div>
# 
# 
# | 왼편 그래프(C=100)    | 오른편 그래프(C=0.01)    |
# | -------------: | -------------: |
# | 규제 보다 약함 | 규제 보다 강함 |
# | 샘플에 덜 민감 | 샘플에 더 민감 |
# | 마진 오류 보다 적게 | 마진 오류 보다 많이  |

# **회귀 모델 시간 복잡도**
# 
# * `LinearSVR`: `LinearSVC`의 회귀 버전
#     * 시간 복잡도가 훈련 세트의 크기에 비례해서 선형적으로 증가
# 
# * `SVR`: `SVC`의 회귀 버전
#     * 훈련 세트가 커지면 매우 느려짐

# ## SVM 이론

# ### SVM 분류기의 결정 함수, 예측, 결정 경계, 목적함수

# **결정 함수와 예측**
# 
# * 결정 함수: 아래 값을 이용하여 클래스 분류
# 
# $$
# h(\mathbf x) = \mathbf w^T \mathbf x + b = w_1 x_1 + \cdots + w_n x_n + b
# $$
# 
# * 예측값: 결정 함수의 값이 양수이면 양성, 음수이면 음성으로 분류
# 
# $$
# \hat y = \begin{cases}
#             0 & \text{if } h(\mathbf x) < 0\\
#             1 & \text{if } h(\mathbf x) \ge 0
#          \end{cases}
# $$

# **결정 경계**
# 
# * 결정 경계: 결정 함수의 값이 0인 점들의 집합
# 
# $$\{\mathbf x \mid h(\mathbf x)=0  \}$$
# 
# * 결정 경계 도로의 경계: 결정 함수의 값이 1 또는 -1인 샘플들의 집합
# 
# $$\{\mathbf{x} \mid h(\mathbf x)= \pm 1 \}$$

# **예제**
# 
# 붓꽃 분류. 꽃잎 길이와 너비를 기준으로 버지니카(Iris-Virginica, 초록 삼각형) 품종 여부 판단
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-13.png" width="600"/></div>

# **결정 함수의 기울기**
# 
# * 결정 경계면(결정 함수의 그래프, 하이퍼플레인)의 기울기가 작아질 수록 도로 경계 폭이 커짐.
# 
# * 결정 경계면 기울기가 $\| \mathbf w \|$에 비례함. 
#     따라서 결정 경계 도로의 폭을 크게 하기 위해 $\| \mathbf w \|$를 최소화해야 함.
# 
# <div align="center"><img src="https://raw.githubusercontent.com/codingalzi/handson-ml3/master/jupyter-book/imgs/ch05/homl05-14.png" width="600"/></div>
# 
# * 하드 마진 모델 훈련: 모든 양성(음성) 샘플이 결정 경계 도로 밖에 위치하도록 하는 기울기 찾기.
# 
# * 소프트 마진 모델 훈련: 결정 경계 도로 위에 위치하는 샘플의 수를 제한하면서 결정 경계 도로의 폭이 최대가 되도록 하는 기울기 찾기.

# **목적함수**
# 
# * 결정 경계면의 기울기 $\| \mathbf w \|$를 최소화하는 것과 아래 식을 최소화하는 것이 동일한 의미임.
#     따라서 아래 식을 목적함수로 지정함.
# 
# $$\frac 1 2 \| \mathbf w \|^2 = \frac 1 2 \mathbf w^T \mathbf w$$
#     
# 
# * 이유: 함수의 미분가능성 때문에 수학적으로 다루기가 보다 쉬움. $1/2$ 또한 계산의 편의를 위해 추가됨.

# **하드 마진 선형 SVM 분류기의 목적 함수**
# 
# * 목적함수를 최소화하는 파라미터 벡터 $\mathbf{w}$를 구하기 위해 다음 __최적화 문제__를 해결해야 함.
# 
# $$\frac 1 2 \mathbf w^T \mathbf w$$
#     
# 
# $$
# \text{(조건)}\quad t^{(i)} (\mathbf w^T \mathbf x^{(i)} + b) \ge 1
# $$
# 
# * 즉, 모든 샘플 $\mathbf{x}^{(i)}$에 대해 만족시켜야 하는 조건이 추가되었음. 
#     $t^{(i)}$는 $i$ 번째 샘플의 클래스(양성/음성)를 가리킴.
# 
# 
# $$
# t^{(i)} = 
# \begin{cases}
# -1 & \text{$x^{(i)}$가 음성인 경우} \\
# 1 & \text{$x^{(i)}$가 양성인 경우} 
# \end{cases}
# $$

# **조건식의 의미**
# 
# $$
# \text{(조건)}\quad t^{(i)} (\mathbf w^T \mathbf x^{(i)} + b) \ge 1
# $$
# 
# 위 조건식의 의미는 다음과 같다.
# 
# * $\mathbf x^{(i)}$가 양성인 경우
#     - $t^{(i)} = 1$
#     - 따라서 $\mathbf w^T \mathbf x^{(i)} + b \ge 1$, 즉 양성으로 예측해야 함.
# 
# * $\mathbf x^{(i)}$가 음성인 경우
#     - $t^{(i)} = -1$
#     - 따라서 $\mathbf w^T \mathbf x^{(i)} + b \le -1$, 즉 음성으로 예측해야 함.

# **소프트 마진 선형 SVM 분류기의 목적 함수**
# 
# * 목적함수와 조건이 다음과 같음.
# 
# $$\frac 1 2 \mathbf w^T \mathbf w + C \sum_{i=0}^{m-1} \zeta^{(i)}$$    
# 
# $$\text{(조건)}\quad t^{(i)} (\mathbf w^T \mathbf x^{(i)} + b) \ge 1 - \zeta^{(i)}$$
#     
# 
# * $\zeta^{(i)}\ge 0$: __슬랙 변수__. $i$ 번째 샘플에 대한 마진 오류 허용 정도 지정.
#     ($\zeta$는 그리스어 알파벳이며 '체타(zeta)'라고 발음함.)
# 
# * $C$: 아래 두 목표 사이의 트레이드오프를 조절하는 하이퍼파라미터
#     * 목표 1: 결정 경계 도로의 폭을 가능하면 크게 하기 위해 $\|\mathbf w\|$ 값을 가능하면 작게 만들기.
#     * 목표 2: 마진 오류 수를 제한하기, 즉 슬랙 변수의 값을 작게 유지하기.
# 
# - __참고:__ 결정 경계 도로의 폭, 즉 마진 폭은 결정 경계면($\hat y = \mathbf{w}^T \mathbf{x} + b$)의 기울기 $\|\mathbf w\|$ 에 의해 결정됨

# **$\zeta$의 역할**
# 
# - $\zeta^{(i)} > 0$이면 해당 샘플 $\mathbf{x}^{(i)}$에 대해 다음이 성립하여 마진 오류가 될 수 있음.
#     
#     $$1 - \zeta^{(i)} \le t^{(i)} (\mathbf w^T \mathbf x^{(i)} + b) < 1$$
# 
# - 이유: 결정 경계면(하이퍼플레인) 상에서 보면 결정 함숫값이 $1$보다 작은 샘플이기에
#     실제 데이터셋의 공간에서는 결정 경계 도로 안에 위치하게 됨.
#     (결정 경계 도로의 양 경계는 결정 함숫값이 $1$인 샘플들로 이루어졌음.)

# **`C`와 마진 폭의 관계**
# 
# $$\frac 1 2 \mathbf w^T \mathbf w + C \sum_{i=0}^{m-1} \zeta^{(i)}$$    
# 
# $$\text{(조건)}\quad t^{(i)} (\mathbf w^T \mathbf x^{(i)} + b) \ge 1 - \zeta^{(i)}$$
#     
# 
# - 가정: 보다 간단한 설명을 위해 편향 $b$는 $0$이거나 무시될 정도로 작다고 가정. (표준화 전처리를 사용하면 됨.)
# 
# - $C$가 매우 큰 경우
#     - $\zeta$는 $0$에 매우 가까울 정도로 아주 작아짐.
#     - 예를 들어 양성 샘플 $\mathbf{x}^{(i)}$에 대해, 즉 $t^{(i)} = 1$, 
#         $\mathbf{w}^T \mathbf{x}^{(i)}$ 가 $1$보다 크거나 아니면 $1$보다 아주 조금만 작아야 함.
#         즉, 결정 경계면의 기울기 $\|w\|$가 어느 정도 커야 함.
#     - 결정 경계의 도로폭이 좁아짐.
# 
# - $C$가 매우 작은 경우
#     - $\zeta$가 어느 정도 커도 됨.
#     - $\mathbf{w}^T \mathbf{x}^{(i)}$ 가 1보다 많이 작아도 됨. 즉, $\|w\|$ 가 작아도 됨.
#     - 결정 경계의 도로폭이 넓어짐.
#  

# ### 커널 SVM 작동 원리

# **쌍대 문제**
# 
# * 쌍대 문제(dual problem): 주어진 문제의 답과 동일한 답을 갖는 문제
# 
# * 선형 SVM 목적 함수의 쌍대 문제: 아래 식을 최소화하는 $\alpha$ 찾기(단, $\alpha^{(i)} > 0$).
# 
# $$
# \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha^{(i)}\alpha^{(j)} t^{(i)} t^{(j)} {\mathbf{x}^{(i)}}^T\mathbf{x}^{(j)} - \sum_{j=1}^{m} \alpha^{(i)}
# $$

# **쌍대 문제 활용 예제: 다항 커널**
# 
# * 원래 $d$차 다항식 함수 $\phi()$를 적용한 후에 쌍대 목적 함수의 최적화 문제를 해결해야 함.
#     즉, 아래 문제를 최소화하는 $\alpha$를 찾는 게 쌍대문제임.
# 
# $$
# \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha^{(i)}\alpha^{(j)} t^{(i)} t^{(j)} \phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)}) - \sum_{j=1}^{m} \alpha^{(i)}
# $$
# 
# * 하지만 다음이 성립함.
# 
# $$
# \phi(\mathbf a)^T \phi(\mathbf b) = ({\mathbf a}^T \mathbf b)^d
# $$
# 
# * 따라서 다항식 함수 $\phi$를 적용할 필요 없이, 즉 다항 특성을 전혀 추가할 필요 없이
#     아래 함수에 대한 최적화 문제를 해결하면 다항 특성을 추가한 효과를 얻게 됨.
# 
# $$
# \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha^{(i)}\alpha^{(j)} t^{(i)} t^{(j)} \left({\mathbf{x}^{(i)}}^T\mathbf{x}^{(j)}\right)^d - \sum_{j=1}^{m} \alpha^{(i)}
# $$

# **예제: 지원되는 커널**
# 
# * 다항식: 
#     
# 
# $$K(\mathbf a, \mathbf b) = \big( \gamma \mathbf a^T  \mathbf b + r \big)^d$$
# 
# * 가우시안 RBF:
# 
# $$K(\mathbf a, \mathbf b) = \exp \big( \!-\! \gamma \| \mathbf a -  \mathbf b \|^2 \big )$$

# ### 온라인 SVM

# * 경사하강법을 이용하여 선형 SVM 분류기를 직접 구현할 수 있음.
# 
# * 비용함수는 아래와 같음.
# 
# $$
# J(\mathbf{w}, b) = \dfrac{1}{2} \mathbf{w}^T \mathbf{w} \,+\, C {\displaystyle \sum_{i=1}^{m}\max\left(0, 1 - t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \right)}
# $$
# 
# * 자세한 내용은 주피터 노트북의 부록 B 참조: [[html]](https://codingalzi.github.io/handson-ml2/notebooks/handson-ml2-05.html), [[구글 코랩]](https://colab.research.google.com/github/codingalzi/handson-ml2/blob/master/notebooks/handson-ml2-05.ipynb)
