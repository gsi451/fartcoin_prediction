# LSTM을 이용한 주가 예측 모델

## 1. 모델 구현 방법
LSTM(Long Short-Term Memory) 모델을 사용하여 주가 예측을 수행합니다.

### 📌 라이브러리 임포트
```python
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import lstm_helper_ori as lstm
```

- `Sequential()` 모델을 사용하여 계층을 쌓음
- LSTM을 두 개의 층으로 구성하여 장기 의존성을 학습하도록 설계
- Dropout을 사용하여 과적합 방지

---

## 2. 모델 계층 구조
```python
model = Sequential()

model.add(LSTM(
    units=50,  # LSTM 첫 번째 층, 뉴런 50개
    input_shape=(None, 1),  # 입력 데이터 형태 (시퀀스 길이, 특징 수)
    return_sequences=True))  # 다음 LSTM 층에 시퀀스 전달
model.add(Dropout(0.3))  # 과적합 방지를 위한 Dropout

model.add(LSTM(
    units=100,  # 두 번째 LSTM 층, 뉴런 100개
    return_sequences=False))  # 마지막 LSTM 층이므로 False 설정
model.add(Dropout(0.3))

model.add(Dense(units=1))  # 출력층 (주가 예측값)
model.add(Activation('linear'))  # 선형 활성화 함수 사용

# 모델 컴파일
start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print("Compilation time:", time.time() - start)
```

### ✅ LSTM 모델 특징
- **첫 번째 LSTM 층**: 뉴런 50개, `return_sequences=True`로 설정 (다음 층으로 시퀀스를 전달)
- **Dropout(0.3)**: 과적합을 방지하기 위해 30%의 뉴런을 무작위로 비활성화
- **두 번째 LSTM 층**: 뉴런 100개, `return_sequences=False`로 마지막 시퀀스 값만 반환
- **Dense 출력층**: 1개의 출력 뉴런을 가지며, 활성화 함수는 `linear` (연속 값 예측)

---

## 3. 모델 학습
```python
history = model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=100,  # 학습 반복 횟수
    validation_split=0.2,  # 검증 데이터 20% 사용
    callbacks=[TqdmCallback()])  # 진행 상태 시각화
```

### ✅ 학습 설정
- **배치 크기**: 512
- **에포크**: 100
- **검증 데이터**: 20%로 설정
- **진행 상황을 보기 위해 `TqdmCallback()` 사용**

---

## 4. 모델 평가 및 시각화
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

- 학습 과정에서 **손실값(`loss`)과 검증 손실값(`val_loss`)**을 그래프로 시각화

---

## 📌 결론
- **LSTM 모델을 사용하여 시계열 주가 데이터를 학습**
- **2개의 LSTM 층과 Dropout을 통해 장기 패턴을 학습하면서 과적합을 방지**
- **선형 활성화 함수를 사용하여 연속된 주가 예측**
- **배치 크기 512, 100 에포크 동안 학습**
- **훈련 과정에서 손실값을 시각화하여 학습 진행을 평가**

---

📌 **개선점이나 추가 기능에 대한 제안이 있다면 자유롭게 Issue를 남겨주세요!** 🚀



---
---
# 코인 데이터 예측 및 웹 시각화 프로젝트

## 1. 개요
이 프로젝트는 LSTM을 이용하여 코인 데이터를 학습하고, 예측한 데이터를 실제 주식 차트에 시각적으로 표현하는 것을 목표로 합니다.

## 2. 프로젝트 구성
### 2.1 학습 코드
- **`stock.ipynb`**: LSTM을 이용한 학습 소스 코드가 포함되어 있습니다.

### 2.2 웹 서버
- **`webserver/`** 폴더는 웹서버를 구동하는 역할을 하며, 웹 화면에서 실제 예측한 값을 도식화할 수 있습니다.

## 3. 웹 화면 예제
### 예측(그린)라인 표시
- 아래 이미지는 웹 화면에서 예측 데이터를 시각적으로 표현한 예제입니다.

  ![예측(그린)라인 표시](https://github.com/gsi451/fartcoin_prediction/blob/main/newplot.png)
  ![예측(그린)라인 표시](https://github.com/gsi451/fartcoin_prediction/blob/main/newplot_zoom.png)

- 웹 화면에서 예측치를 확인하며, 왼쪽과 오른쪽 위치를 변경하여 예측값이 어떻게 변하는지 관찰할 수 있습니다.

## 4. 추가적으로 해볼만한 작업
### 4.1 다양한 코인 데이터 활용
- 여러 종류의 코인 데이터를 수집하여 학습 데이터로 적용하고 모델을 일반화할 수 있도록 개선합니다.

### 4.2 예측 데이터와 강화학습 결합
- 예측한 데이터를 강화학습에 접목하여 자동화된 매매 전략을 개발할 수 있습니다.

### 4.3 실전 매매 데모 구현
- 예측한 데이터가 실제 매매 전략에 유용한지 확인하기 위해 데모 형태로 실전 매매 시뮬레이션을 구현해볼 수 있습니다.

## 5. 향후 계획
- 모델 성능 개선 및 하이퍼파라미터 튜닝
- 실시간 데이터 수집 및 예측 모델 업데이트
- 예측 결과를 기반으로 자동 매매 시스템 개발

---
본 프로젝트는 지속적으로 개선되며, 새로운 아이디어나 피드백이 있다면 언제든지 공유해주세요!

