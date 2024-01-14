모델은 train29/weights/best.pt로 불러오시면 됩니다.
크롤링 데이터는 따로 말씀 없으셔서 일단 학습 코드만 올립니다.
학습 코드는 2단계에서 확인하실 수 있으시고, model.train 이후의 셀이 테스트 하는 구간입니다.
model = YOLO("C:/Users/USER/runs/detect/train27/weights/best.pt")의 주석을 푸시고 weight 위치에 맞게 수정하셔서 불러오시면 됩니다.
test 할 것도 본인의 컴퓨터 위치에 맞게 수정하셔서 실행하시면 코드가 돌아갈 것입니다.
