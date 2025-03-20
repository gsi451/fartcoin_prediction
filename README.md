# fartcoin_prediction
fartcoin 코인 데이터를 사용해서 LSTM 학습한 소스

설명자료

1. stock.ipynb : lstm 학습 소스코드입니다.
2. webserver 폴더는 웹서버를 구동하고 웹화면에서 실제 예측한 값을 도식화 합니다.
3. 코인 데이터를 학습을 하고 예측한 데이터를 실제 주식 차트에 그려보고 싶어서 구현해본 소스 입니다.
4. 웹화면
   ![예측(그린)라인 표시](https://github.com/gsi451/fartcoin_prediction/blob/main/newplot.png)
   ![예측(그린)라인 표시](https://github.com/gsi451/fartcoin_prediction/blob/main/newplot_zoom.png)
   - 웹 화면에서 왼쪽, 오른쪽 위치를 변경하면서 해당 위치마다 예측치가 어떻게 변하는지 볼 수 있다.
6. 추가적으로 더 해볼만한 부분들
   - 여러 종류의 코인 데이터를 수집해서 학습에 적용해서 모델을 일반화 시켜보기
   - 예측한 데이터를 강화학습에 접목
   - 예측에 대한 데이터가 효율이 있는지 실전 매매를 데모형태로 구현해보기
