# preprocess.py
# Review의 Text 데이터 전처리

이 파이썬 코드는 한국어 텍스트 데이터의 자연어 처리를 위한 도구를 제공합니다. 특히, 네이버의 API를 사용하여 맞춤법 및 문법 검사를 하는 클래스와 다양한 전처리 작업을 위한 클래스를 포함하고 있습니다.

## 기능

1. **Naver 클래스**: 네이버의 맞춤법 및 문법 검사기를 사용하여 한국어 텍스트를 교정합니다.
2. **NLP_Preprocessor 클래스**: 텍스트 전처리를 위한 다양한 메서드를 포함합니다.

## 사용 방법

### Naver 클래스

- `convert_spelling(text)`: 입력 텍스트의 맞춤법 및 문법을 교정합니다.
- `Convert_spelling(sentences)`: 여러 문장을 처리하고 교정합니다.

### NLP_Preprocessor 클래스

#### 정제(Cleansing) 메서드
- `delete_welcome(context)`: 텍스트에서 일반적인 오타를 수정합니다.
- `delete_stopwords(context)`: 텍스트에서 불용어를 제거합니다.
- `delete_html_tag(context)`: 텍스트에서 HTML 태그를 제거합니다.
- `delete_email(context)`: 텍스트에서 이메일 주소를 제거합니다.
- `delete_hashtag(context)`: 텍스트에서 해시태그(#)를 제거합니다.
- `delete_emoticons(context)`: 텍스트에서 이모티콘을 제거합니다.

#### 정규화(Normalization) 메서드
- `repeat_char_normalizer(context)`: 반복되는 문자를 정규화합니다.
- `repeated_spacing_normalizer(context)`: 공백을 정규화합니다.
- `duplicated_sentence_normalizer(context)`: 중복된 문장을 제거합니다.
- `min_max_filter(context)`: 문장의 길이에 따라 필터링합니다.

#### 전처리 함수
- `preprocess_first(context)`: 첫 번째 단계의 텍스트 전처리를 수행합니다.


## 요구 사항

- Python 3
- 필요 라이브러리: `requests`, `json`, `time`, `sys`, `collections`, `xml`, `re`, `urllib`, `pandas`, `soynlp`

## 설치

모든 필요 라이브러리가 설치되어 있는지 확인하십시오. 다음 명령어를 사용하여 의존성을 설치하십시오:

```
pip install requests pandas soynlp
```

## 예시

다음은 이 클래스들의 사용 방법을 보여주는 간단한 예시입니다:

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

- Naver 클래스는 네이버 API와 통신하기 위해 인터넷 연결이 필요합니다.
- 전처리 메서드는 특정 요구 사항에 따라 조정하거나 확장할 수 있습니다.

&emsp;

# How_to_use.py
# How to use preprocess.py

이 코드는 GitHub에서 데이터를 클론하고, 패션 관련 리뷰 데이터에 대한 전처리를 수행하는 방법을 제공합니다.

## 기능

1. **GitHub 데이터 클론**: 특정 GitHub 저장소에서 데이터를 클론합니다.
2. **패키지 설치**: 필요한 Python 패키지를 설치합니다.
3. **데이터 불러오기**: 다양한 카테고리의 패션 리뷰 데이터를 불러옵니다.
4. **데이터 전처리**: NLP 전처리 및 네이버 맞춤법 검사를 수행합니다.
5. **CSV 파일로 저장**: 전처리된 데이터를 CSV 파일로 저장합니다.

## 사용 방법

1. **GitHub에서 데이터 클론하기**:
    - TrainData 및 Model 저장소를 클론합니다.

2. **필요한 패키지 설치하기**:
    - soynlp 패키지를 설치합니다.

3. **데이터 불러오기**:
    - 각 카테고리 별로 CSV 파일을 불러옵니다.

4. **데이터 전처리하기**:
    - `NLP_Preprocessor`와 `Naver` 클래스를 사용하여 데이터를 전처리합니다.
    - 첫 번째 단계 전처리 후 네이버 맞춤법 검사를 수행합니다.

5. **CSV 파일로 저장하기**:
    - 전처리된 데이터를 새로운 CSV 파일로 저장합니다.

6. **전처리된 데이터 검사하기**:
    - 저장된 CSV 파일에서 리뷰 데이터를 확인합니다.

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
preprocess_first_Completed = preprocessor.preprocess_first(review_data)
naver_Completed = naver.Convert_spelling(preprocess_first_Completed)

# 데이터 저장
Fashion_list[index].to_csv("Preprocessed_XXX.csv", index=False)

# 전처리된 데이터 검사
Test = pd.read_csv("Preprocessed_XXX.csv")
print(Test['review'])
```

## 참고 사항

- GitHub 저장소에서 클론하는 과정에서 `{Github_User_Name}`과 `{Github_Token}`은 사용자의 GitHub 계정 이름과 토큰으로 대체해야 합니다.
- 전처리 메서드는 프로젝트의 요구에 따라 조정될 수 있습니다.
