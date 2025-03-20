from flask import Flask, jsonify, send_from_directory
import tensorflow as tf
import pandas as pd
from datetime import datetime
import random
import numpy as np
from numpy import newaxis
import lstm_helper_ori as lstm

app = Flask(__name__)

# 모델 로드
MODEL_FILE = "fartcoin_20250317.keras"
model = tf.keras.models.load_model(MODEL_FILE)

# 데이터 파일 로드
CSV_FILE = "extracted_close4000.csv"
df = pd.read_csv(CSV_FILE)

#print(df.iloc[:200])

# 미리 종가 값만 분리하여 저장
close_values = df["close"].values  # NumPy 배열로 변환

index = 100  # 현재 데이터의 시작점, 초기값은 100부터 시작
WINDOW_SIZE = 50  # 한 번에 표시할 데이터 개수
PREDICTION_LEN = 50 # 예측 배열


@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.htm')

@app.route('/get_data')
def get_data():
    global index
    start = index
    #end = min(index + WINDOW_SIZE, len(df))
    end = len(df)
    
    data = [
        {
            "Date": row["timestamp"], 
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"]
        }
        for _, row in df.iloc[start:end].iterrows()
    ]
    #print(data)
    return jsonify(data)

@app.route('/move/<direction>')
def move(direction):
    global index
    if direction == "next":
        index = min(index + 1, len(df) - PREDICTION_LEN)
    elif direction == "prev":
        index = max(index - 1, 100)
    elif direction == "next10x":
        index = min(index + 10, len(df) - PREDICTION_LEN)
    elif direction == "prev10x":
        index = max(index - 10, 100)
    elif direction == "next100x":
        index = min(index + 100, len(df) - PREDICTION_LEN)
    elif direction == "prev100x":
        index = max(index - 100, 100)

    #print("넘어온 인덱스", index)

    start = max(index + 50, 0)  # 왜 50을 더해야 하지?!!!
    end = index + 50 + WINDOW_SIZE
    #print("구간정보", start, end)
 
    # 종가 값만 가져오기
    window_close_values = close_values[start:end]
    window_data = [window_close_values.tolist()]
    result, mm = lstm.min_max_normalize(window_data)

    result = np.array(result)[..., np.newaxis]  # (1, 50, 1) 형태로 변환

    # 예측을 진행한다.
    predictions_index = lstm.predict_sequences_multiple_index(model, result, 50, 50)
    #print("예측값", predictions_index.shape)
    #print("예측값", predictions_index)

    # 예측값을 디노멀라이징 한다.
    dnormalize_prediction = lstm.denormalize_predictions_global(predictions_index, mm[0][0], mm[0][1])

    #print("예측값 디노멀라이징", dnormalize_prediction.shape)
    #print("예측값 디노멀라이징", dnormalize_prediction)

    # PREDICTION_LEN 길이만큼 랜덤한 값(0.3 ~ 0.4)으로 채운 배열 생성
    #random_values = [round(random.uniform(0.3, 0.4), 4) for _ in range(PREDICTION_LEN)]
    random_values = dnormalize_prediction[0, :, -1].tolist()
    #print("random_values", random_values)

    return jsonify({
        "status": "ok",
        "index": index,
        "prediction": random_values  # 랜덤 값 배열 추가
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8100, debug=True)
