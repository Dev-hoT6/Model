# preprocess.py
# Review의 Text 데이터 전처리

아래의 파이썬 코드는 한국어 텍스트 데이터의 자연어 처리를 위한 전처리 도구를 제공합니다. 
또한, 1) 네이버의 맞춤법 검사기 API를 사용하여 맞춤법 및 문법 검사를 하는 클래스와 2) 다양한 전처리 작업들을 위한 클래스를 포함하고 있습니다.

## 기능

1. **NLP_Preprocessor 클래스**: 텍스트 전처리를 위한 다양한 메서드를 포함
2. **Naver 클래스**: 네이버의 맞춤법 및 문법 검사기를 사용하여 한국어 텍스트를 교정


## 사용 방법

### NLP_Preprocessor Class

#### 정제(Cleansing) Method
- `delete_welcome(context)`: 텍스트에서 일반적인 오타를 수정
- `delete_stopwords(context)`: 텍스트에서 불용어를 제거
- `delete_html_tag(context)`: 텍스트에서 HTML 태그를 제거
- `delete_email(context)`: 텍스트에서 이메일 주소를 제거
- `delete_hashtag(context)`: 텍스트에서 해시태그(#)를 제거
- `delete_emoticons(context)`: 텍스트에서 이모티콘을 제거

#### 정규화(Normalization) Method
- `repeat_char_normalizer(context)`: 반복되는 문자를 정규화
- `repeated_spacing_normalizer(context)`: 공백을 정규화
- `duplicated_sentence_normalizer(context)`: 중복된 문장을 제거
- `min_max_filter(context)`: 문장의 길이에 따라 필터링

#### 전처리(preprocess) Method
- `preprocess_first(context)`: 사용자의 NLP Task에 맞는 위의 정제 및 정규화 Method들을 커스터마이징하여 사용할 수 있는 전처리 Method


### Naver Class

- `convert_spelling(text)`: 입력 텍스트 한 문장의 맞춤법 및 문법을 교정
- `Convert_spelling(sentences)`: convert_spelling의 복수의 문장을 처리하고 교정할 수 있는 version


## 요구 사항

- Python 3
- 필요한 라이브러리: `requests`, `json`, `time`, `sys`, `collections`, `xml`, `re`, `urllib`, `pandas`, `soynlp`

## 패키지 설치

모든 필요한 라이브러리들이 설치되어 있는지 확인하세요. 다음 명령어를 통해 패키지를 설치할 수 있습니다.:

```
pip install requests pandas soynlp
```

## 예시

위 두 개의 클래스들의 사용 방법을 보여주는 간단한 예시입니다.:

```python
# 클래스 인스턴스 생성
preprocessor = NLP_Preprocessor()
naver = Naver()

# 샘플 리뷰의 텍스트 데이터 (데이터 타입 : pd.Series)
review_text_data = DataFrame['review']

# 텍스트 전처리
preprocess_first_Completed = preprocessor.preprocess_first(review_text_data)

# Naver 클래스를 사용하여 맞춤법 검사
naver_Completed = naver.Convert_spelling(preprocess_first_Completed)


```

## 참고 사항

- Naver Class는 네이버 API와 통신하기 때문에 반드시 인터넷 연결이 필요합니다.
- 전처리 Method는 사용자의 요구 사항에 따라 커스터마이징 할 수 있습니다.

&emsp;

# How_to_use.py
# How to use "preprocess.py"

아래의 파이썬 코드는 preprocess.py를 사용하는 방법을 알려드리기 위해 제작되었습니다.

## 기능

1. **GitHub 데이터 클론**: 특정 GitHub 저장소에서 데이터를 클론
2. **패키지 설치**: 필요한 Python 패키지를 설치
3. **데이터 불러오기**: 다양한 카테고리의 패션 리뷰 데이터를 로드
4. **데이터 전처리**: NLP 전처리 및 네이버 맞춤법 검사를 수행
5. **CSV 파일로 저장**: 전처리된 데이터를 CSV 파일로 저장

## 사용 방법

1. **GitHub에서 데이터 클론하기**:
    - TrainData 및 Model의 Github repository를 클론

2. **필요한 패키지 설치하기**:
    - soynlp 패키지를 설치

3. **데이터 불러오기**:
    - 카테고리 별로 CSV 파일을 로드

4. **데이터 전처리하기**:
    - `NLP_Preprocessor`와 `Naver` Class를 사용하여 데이터를 전처리함
    - '1차 전처리' 이후 '네이버 맞춤법 검사기'를 수행

5. **CSV 파일로 저장하기**:
    - 전처리된 데이터를 새로운 CSV 파일로 저장

6. **전처리된 데이터 검사하기**:
    - 저장된 CSV 파일에서 리뷰 데이터를 검사

## 예시

```python
# GitHub 저장소 클론
!git clone https://{Github_User_Name}:{Github_Token}@github.com/Dev-hoT6/TrainData.git
!git clone https://{Github_User_Name}:{Github_Token}@github.com/Dev-hoT6/Model.git

# 패키지 설치
!pip install soynlp

# 데이터 불러오기
import pandas as pd
Onepiece = pd.read_csv("/content/TrainData/List Data/Onepiece_List.csv")
# ... (다른 카테고리 데이터 불러오기 생략)

# 전처리
from Model.preprocessing.preprocess import NLP_Preprocessor, Naver

preprocessor = NLP_Preprocessor()
naver = Naver()

preprocess_first_Completed = preprocessor.preprocess_first(Onepiece['review'])
naver_Completed = naver.Convert_spelling(preprocess_first_Completed)

# 갈아끼기
Onepiece['review'] = naver_Completed

# 데이터 저장
Onepiece.to_csv("Preprocessed_Onepiece.csv", index=False)

# 전처리된 데이터 검사
Preprocessed_Onepiece = pd.read_csv("Preprocessed_Onepiece.csv")
print(Preprocessed_Onepiece['review'])
```


## 참고 사항

- GitHub repository에서 클론하는 과정에서 `{Github_User_Name}`과 `{Github_Token}`은 사용자의 GitHub User_name과 토큰으로 대체해주셔야 합니다.
