# 디렉토리 설명
RTR 프로젝트의 Review 품질 평가 모델 학습 및 최적화를 위한 디렉토리

# Requirements
```
(Google Colaboratory)

onnx==1.15.0
onnxruntime==1.16.3
pytorch==2.1.0+cu121
```

# 파일 리스트
- `Review Model_Baseline.py` : 리뷰 평가 모델 학습을 위한 베이스라인
- `Classifier_123.ipynb` : 리뷰 품질 평가 모델(1-3단계) 학습 및 평가
- `SBERT_optimization_ONNX.ipynb` : 리뷰 벡터화 모델 최적화 및 ONNX 변환
- `Classifier_123_ONNX_export_test.ipynb` : 리뷰 품질 평가 모델(1-3단계) ONNX 변환 및 평가