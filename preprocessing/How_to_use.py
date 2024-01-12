!git clone https://{Github_User_Name}:{Github_Token}@github.com/Dev-hoT6/TrainData.git
!git clone https://{Github_User_Name}:{Github_Token}@github.com/Dev-hoT6/Model.git



# 패키지 설치하기
!pip install soynlp



# 데이터 불러오기
import pandas as pd

Onepiece = pd.read_csv("/content/TrainData/List Data/Onepiece_List.csv")
Outer = pd.read_csv("/content/TrainData/List Data/Outer_List.csv")
Pants = pd.read_csv("/content/TrainData/List Data/Pants_List.csv")
Skirt = pd.read_csv("/content/TrainData/List Data/Skirt_List.csv")
Top = pd.read_csv("/content/TrainData/List Data/Top_List.csv")
Sport_Outer = pd.read_csv("/content/TrainData/List Data/Sport_Outer_List.csv")
Sport_Pants = pd.read_csv("/content/TrainData/List Data/Sport_Pants_List.csv")
Sport_Top = pd.read_csv("/content/TrainData/List Data/Sport_Top_List.csv")



Fashion_list = [Onepiece, Outer, Pants, Skirt, Top, Sport_Outer, Sport_Pants, Sport_Top]

index = 0 # Index 주의
review_data = Fashion_list[index]['review']



# 전처리

# 클래스
from Model.preprocessing.preprocess import NLP_Preprocessor, Naver
preprocessor = NLP_Preprocessor()
naver = Naver()

# NLP_Preprocessor 클래스 사용법
preprocess_first_Completed = preprocessor.preprocess_first(review_data)
# Naver 클래스 사용법
naver_Completed = naver.Convert_spelling(preprocess_first_Completed)



# 갈아끼기
Fashion_list[index]['review'] = naver_Completed



# CSV 파일로 내보내기
Fashion_list[index].to_csv("Preprocessed_XXX.csv", index=False) # 파일명 주의



# 검사하기
Test = pd.read_csv("Preprocessed_XXX.csv") # 파일명 주의
Test['review']
