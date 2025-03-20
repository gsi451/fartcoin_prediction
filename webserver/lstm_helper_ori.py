import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    # Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]  # ✅ xrange → range 변경
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def load_data(filename, seq_len, normalise_window):
    with open(filename, 'r') as f:
        data = f.read().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = min_max_normalize(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def load_data_minmax(filename, seq_len, normalise_window):
    with open(filename, 'r') as f:
        data = f.read().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
    #for index in range(min(2, len(data) - sequence_length)):  # 최대 2번만 실행
        result.append(data[index: index + sequence_length])
    
    #print("원본데이터:", result[0][:5])

    if normalise_window:
        result, mm = min_max_normalize(result)

    mm = np.array(mm)
    #print("mm 타입확인", type(mm), mm.shape)
    #print(mm[:10])

    result = np.array(result)

    #print("노멀라이징 데이터:", result[0][:5])

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    min_max_y_train = mm[:int(row)]
    min_max_y_test = mm[int(row):]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, min_max_y_train, min_max_y_test]

def min_max_normalize(window_data):
    """
    Min-Max Scaling 정규화 함수 (각 윈도우별 적용)
    :param window_data: 2D 리스트 형태의 데이터 (윈도우 단위)
    :return: 정규화된 데이터, 각 윈도우별 min/max 값 리스트, 윈도우마다 하나의 값이다.
    """

    normalised_data = []
    min_max_values = []  # 각 윈도우의 (min, max) 저장

    #print("윈도우 데이터 개수", len(window_data))
    
    for window in window_data:
        # 문자열 데이터를 float 타입으로 변환
        window = [float(p) for p in window]

        #print(window)
        
        min_val = min(window)
        max_val = max(window)

        #print("최소값, 최대값", min_val, max_val)

        if max_val - min_val == 0:  # 모든 값이 동일하면 정규화 불가능 -> 0으로 설정
            normalised_window = [0] * len(window)
        else:
            normalised_window = [(p - min_val) / (max_val - min_val) for p in window]
        
        normalised_data.append(normalised_window)
        min_max_values.append((min_val, max_val))

        #print("노멀테스트 배열", normalised_data)

        #print("최소값, 최대값 배열", min_max_values)

        #print("한번만 실행하도록 처리")
        #return

    return normalised_data, min_max_values


def min_max_denormalize(normalized_value, min_val, max_val):
    """
    정규화된 값을 원래 값으로 복원
    :param normalized_value: 정규화된 값 (예측된 값)
    :param min_val: 정규화할 때 사용한 최소값
    :param max_val: 정규화할 때 사용한 최대값
    :return: 원래 데이터 값
    """
    if min_val is None or max_val is None:
        return normalized_value  # 정규화를 하지 않은 경우 원본 값 그대로 반환
    return normalized_value * (max_val - min_val) + min_val

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        units=layers[1],  # ✅ output_dim → units
        input_shape=(layers[0], 1),  # ✅ input_dim → input_shape 변경
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=layers[2],  # ✅ output_dim → units
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=layers[3]))  # ✅ output_dim → units
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time :", time.time() - start)  # ✅ print 수정
    return model

def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):  # ✅ xrange → range 변경
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(len(data) // prediction_len):  # ✅ `/` → `//` 정수 나눗셈 변경
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):  # ✅ xrange → range 변경
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def predict_sequences_multiple_index(model, data, window_size, prediction_len):
    #print(">> 함수 실행: predict_sequences_multiple_index22")
    #print(f"data.shape: {data.shape}")  # (1, 50, 1) 확인
    #print(f"window_size: {window_size}, prediction_len: {prediction_len}")

    prediction_seqs = []
    index = 0  # 인덱스 변수 초기화

    # ✅ 최소 1번 실행되도록 수정
    for i in range(max(1, len(data) // prediction_len)):  
        curr_frame = data[i * prediction_len].reshape(window_size, 1)

        #print(f"\n--- 예측 시작 (i={i}) ---")
        #print(f"curr_frame.shape: {curr_frame.shape}")  # (50, 1) 확인
        #print(f"curr_frame[:5]: {curr_frame[:5]}")  # 일부 데이터 출력

        predicted = []

        for j in range(prediction_len):
            # 모델 예측
            prediction_input = curr_frame[newaxis, :, :]
            prediction_output = model.predict(prediction_input, verbose=0)[0, 0]

            #print(f"  [{index}] 예측값: {prediction_output}")

            predicted.append([index, prediction_output])

            # 프레임 업데이트
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], prediction_output, axis=0)

            #print(f"  curr_frame.shape after insert: {curr_frame.shape}")
            #print(f"  curr_frame[-5:]: {curr_frame[-5:]}")

            index += 1

        prediction_seqs.append(predicted)

    #print("\n>> 최종 예측 결과")
    #print(prediction_seqs[:2])  # 일부 결과 출력

    return prediction_seqs


def denormalize_predictions(predictions_index, min_max_y_test):
    # 디노멀라이징 결과를 저장할 배열 생성
    # 디노멀라이징 결과를 저장할 배열 생성
    pre_index = np.array(predictions_index)
    denormalized_predictions = np.zeros((pre_index.shape[0], pre_index.shape[1], 3), dtype=float)

    print(pre_index.shape)
    print(pre_index)
    print(denormalized_predictions.shape)

    print(pre_index.shape[0])
    print(pre_index.shape[1])

    #print(min_max_y_test[0])

    #print(range(pre_index.shape[0]))

    # pre_index 순회하며 디노멀라이징 계산
    for i in range(pre_index.shape[0]):  # 예측 횟수 순회
        for j in range(pre_index.shape[1]):  # 예측 길이 순회
            index = int(pre_index[i, j, 0])
            predicted_value = pre_index[i, j, 1]

            print("index", index)
            print("predicted_value", predicted_value)
            
            # min_max_y_test에서 최소값, 최대값 가져오기
            min_val = min_max_y_test[index][0]
            max_val = min_max_y_test[index][1]
            print("min_val", min_val)
            print("max_val", max_val)
            
            # 디노멀라이징 계산
            denormalized_value = predicted_value * (max_val - min_val) + min_val
            print("denormalized_value", denormalized_value)
            
            # 결과를 denormalized_predictions 배열에 저장
            denormalized_predictions[i, j] = np.array([index, predicted_value, denormalized_value])
            print(denormalized_predictions[i, j])

            print(denormalized_predictions.shape)
            print(denormalized_predictions)
        
    return denormalized_predictions



def denormalize_predictions_global(predictions_index, min_val, max_val):
    # 디노멀라이징 결과를 저장할 배열 생성
    # 디노멀라이징 결과를 저장할 배열 생성
    pre_index = np.array(predictions_index)
    denormalized_predictions = np.zeros((pre_index.shape[0], pre_index.shape[1], 3), dtype=float)

    #print(pre_index.shape)
    #print(pre_index)
    #print(denormalized_predictions.shape)

    #print(pre_index.shape[0])
    #print(pre_index.shape[1])

    #print("min", min_val, "max", max_val)

    #print(min_max_y_test[0])

    #print(range(pre_index.shape[0]))

    # pre_index 순회하며 디노멀라이징 계산
    for i in range(pre_index.shape[0]):  # 예측 횟수 순회
        for j in range(pre_index.shape[1]):  # 예측 길이 순회
            index = int(pre_index[i, j, 0])
            predicted_value = pre_index[i, j, 1]

            #print("index", index)
            #print("predicted_value", predicted_value)
            
            # 디노멀라이징 계산
            denormalized_value = predicted_value * (max_val - min_val) + min_val
            #print("denormalized_value", denormalized_value)
            
            # 결과를 denormalized_predictions 배열에 저장
            denormalized_predictions[i, j] = np.array([index, predicted_value, denormalized_value])
            #print(denormalized_predictions[i, j])

            #print(denormalized_predictions.shape)
            #print(denormalized_predictions)
        
    return denormalized_predictions