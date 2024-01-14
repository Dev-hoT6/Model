코드의 위치를 A 폴더 내에 있다고 지칭하고 파일 위치가 다음과 같을 때 코드가 작성되었음을 알려드립니다.<br>
* 파일 다운로드 순서
  * 각 단계별 코드
  * 각 카테고리별 이미지 파일(img_data/product_img, img_data/review_img)
  * classes 파일, 모든 이미지 라벨링 파일(All_img_Label)
* 파일 다운로드 위치
  * 각 카테고리별 이미지 파일은 아래 구글 드라이브에서 다운 받으실 수 있습니다.<br>
    * https://drive.google.com/file/d/1DNs33kf2S2a2v6qwS0bJ85PK1mKku30t/view
  * All_img_Label은 All_img_Label.zip 파일을 다운 받으시고 zip을 푸시면 됩니다.
  * 각 단계별 코드, classes 파일은 이 곳의 파일을 다운 받으시면 됩니다.

* 파일 위치는 다음과 같이 변경해주시길 바랍니다. 
* 각 파일 위치(0, 1, 2단계 코드 실행 전에 절취선 기준 위쪽 폴더, 파일, 코드는 존재하여야 함)
  * 0단계 코드 : A 폴더/
  * 1단계 코드 : A 폴더/
  * 2단계 코드 : A 폴더/
  * 모든 상품 이미지 파일(카테고리 별 폴더가 있음) : A 폴더/img_data/product_img
  * 모든 리뷰 이미지 파일(카테고리 별 폴더가 있음) : A 폴더/img_data/review_img
  * 이미지 폴더 : A 폴더/img_data/
  * classes 파일 : A 폴더/img_data/
  * 모든 이미지 라벨링 파일 : A 폴더/img_data/All_img_Label/
    
    ---------------------------------------------------------------------------------------------------
  * 모든 이미지 파일(0단계 코드 실행 시 생성됨, 모든 카테고리가 통합되는 폴더) : A 폴더/img_data/All_img/
  * train image dataset 파일(1단계 코드 실행 시 생성됨) : A 폴더/img_data/train/images/
  * train label dataset 파일(1단계 코드 실행 시 생성됨) : A 폴더/img_data/train/labels/
  * validation image dataset 파일(1단계 코드 실행 시 생성됨) : A 폴더/img_data/val/images/
  * validation label dataset 파일(1단계 코드 실행 시 생성됨) : A 폴더/img_data/val/labels/
  * data.yaml 파일(1단계 코드 실행 시 생성됨) : A 폴더/
<br>

* 파일 내용
  * 1단계 코드 파일
    1. All_img_Label에 들어있는 파일을 토대로 이미지 리스트를 불러오기
    2. 이미지 리스트를 shuffle하고 8:2로 train image dataset, val image dataset, train label dataset, val label dataset 만들기
    3. 이미지 폴더에 train 폴더, val 폴더가 생성되고 각각의 폴더 내에 images, labels 폴더가 생성된다.
    4. 각각의 폴더에 train image dataset, val image dataset, train label dataset, val label dataset 저장하기
    5. classes 파일을 읽어 이미지 클래스 값을 불러오고 각각의  {'train': train_img_dir, 'val': val_img_dir, 'nc': nc, 'names': names} 값을 data.yaml 파일로 저장하기
  
  * 2단계 코드 파일
    1. 1단계에서 저장한 data.yaml 파일 불러오기
    2. model 설정 후 학습하기
    3. 학습된 모델을 가지고 각각의 카테고리 별로 테스트하기

