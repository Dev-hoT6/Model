**Train folder**
* Image folder에는 다음과 같은 코드를 가짐
  * YOLOv8 모델을 사용하여 옷 카테고리 및 사람 인식하는 모델 코드
  * YOLOv8 모델을 ONNX로 변환하는 코드
* Review folder에는 다음과 같은 코드를 가짐
  * SBERT를 사용하여 데이터 레이블링하는 코드
  * Classifier 모델을 사용하여 리뷰의 품질을 판단하는 모델의 코드
  * Classifier 모델을 ONNX로 변환하는 코드

**preprocessing folder**
* 데이터 전처리에 관련된 코드를 가짐
![image](https://github.com/Dev-hoT6/Model/assets/97217295/99a58095-7a79-4a84-905b-b54847ccd88b)
<br>

**전처리 변환시 텍스트 모델 성능**
![image](https://github.com/Dev-hoT6/Model/assets/97217295/6e263eca-8af6-4150-901c-1197fae691fa)
<br>

**텍스트 모델 ONNX 변환시 성능**
![image](https://github.com/Dev-hoT6/Model/assets/97217295/f062e610-030b-437d-8406-466d108198c9)
<br>

**최종 텍스트 모델 성능**
![image](https://github.com/Dev-hoT6/Model/assets/97217295/e433975c-ac7b-4ed1-9f89-1802b11eac9a)
<br>

**최종 이미지 모델 성능**
* 왼쪽 그래프 - 이미지 모델 정확도 
* 오른쪽 그래프 - 사람 인식 정확도 
![image](https://github.com/Dev-hoT6/Model/assets/97217295/9ef56f98-fd89-4486-b168-1e02a3731e91)
