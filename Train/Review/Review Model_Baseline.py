#!/usr/bin/env python
# coding: utf-8

# # Sentence Transformers 학습과 활용

# 본 노트북에서는 `klue/roberta-base` 모델을 **KLUE** 내 **STS** 데이터셋을 활용하여 모델을 훈련하는 예제를 다루게 됩니다.
# 
# 학습을 통해 얻어질 `sentence-klue-roberta-base` 모델은 입력된 문장의 임베딩을 계산해 유사도를 예측하는데 사용할 수 있게 됩니다.
# 
# 학습 과정 이후에는 간단한 예제 코드를 통해 모델이 어떻게 활용되는지도 함께 알아보도록 할 것입니다.
# 
# 모든 소스 코드는 [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers) 원 라이브러리를 참고하였습니다.
# 
# 먼저, 노트북을 실행하는데 필요한 라이브러리를 설치합니다. 모델 훈련을 위해서는 `sentence-transformers`가, 학습 데이터셋 로드를 위해서는 `datasets` 라이브러리의 설치가 필요합니다.

# In[1]:


# !pip install sentence-transformers datasets
# pip install git+https://github.com/huggingface/transformers


# ## Sentence Transformers 학습

# 노트북을 실행하는데 필요한 라이브러리들을 모두 임포트합니다.

# In[2]:


import os
import gc
import math
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from transformers import pipeline #, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification


# In[3]:


# batch size 크기를 줄여가면서 gpu 캐시를 비워주는 코드
gc.collect()
torch.cuda.empty_cache()
# torch.empty_cache()


# 학습 경과를 지켜보는데 사용될 *logger* 를 초기화합니다.

# In[4]:


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


# 학습에 필요한 정보를 변수로 기록합니다.
# 
# 본 노트북에서는 `klue-roberta-base` 모델을 활용하지만, https://huggingface.co/klue 페이지에서 더 다양한 사전학습 언어 모델을 확인하실 수 있습니다.

# In[5]:


model_name = "ddobokki/klue-roberta-base-nli-sts-ko-en" #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"#"klue/roberta-base"#"amberoad/bert-multilingual-passage-reranking-msmarco"#""#"microsoft/phi-2"#"klue/roberta-base"#"amberoad/bert-multilingual-passage-reranking-msmarco"#'klue/paraphrase-MiniLM-L6-v2'#"klue/roberta-base"# mistralai/Mixtral-8x7B-Instruct-v0.1


# 모델 정보 외에도 학습에 필요한 하이퍼 파라미터를 정의합니다.

# In[6]:


train_batch_size = 64
num_epochs = 8
model_save_path = "output/training_klue_sts_8_"+ str(train_batch_size)+ model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 앞서 정의한 사전학습 언어 모델을 로드합니다.
# 
# `sentence-transformers`는 HuggingFace의 `transformers`와 호환이 잘 이루어지고 있기 때문에, [모델 허브](https://huggingface.co/models)에 올라와있는 대부분의 언어 모델을 임베딩을 추출할 *Embedder* 로 사용할 수 있습니다.

# In[7]:


embedding_model = models.Transformer(model_name,max_seq_length=256)
#아래 error를 방지하기 위해 torch만 있는 가상환경을 새로 만듦
    #if you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.


# *Embedder* 에서 추출된 토큰 단위 임베딩들을 가지고 문장 임베딩을 어떻게 계산할 것인지를 결정하는 *Pooler* 를 정의합니다.
# 
# 여러 Pooling 기법이 있겠지만, 예제 노트북에서는 **Mean Pooling**을 사용하기로 합니다.
# 
# **Mean Pooling**이란 모델이 반환한 모든 토큰 임베딩을 더해준 후, 더해진 토큰 개수만큼 나누어 문장을 대표하는 임베딩으로 사용하는 기법을 의미합니다.

# In[8]:


pooler = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)


# *Embedder* 와 *Pooler* 를 정의했으므로, 이 두 모듈로 구성된 하나의 모델을 정의합니다.
# 
# `modules`에 입력으로 들어가는 모듈이 순차적으로 임베딩 과정에 사용이 된다고 생각하시면 됩니다.

# In[9]:


model = SentenceTransformer(modules=[embedding_model, pooler])
# model = SentenceTransformer(model_name)


# 이제 학습에 사용될 KLUE STS 데이터셋을 다운로드 및 로드합니다.

# In[10]:


datasets = load_dataset("kor_nli", "multi_nli")


# 다운로드 혹은 로드 후 얻어진 `datasets` 객체를 살펴보면, 훈련 데이터와 검증 데이터가 포함되어 있는 것을 확인할 수 있습니다.

# In[11]:


datasets


# 각 예시 데이터는 아래와 같이 두 개의 문장과 두 문장의 유사도를 라벨로 지니고 있습니다.

# In[12]:


datasets["train"][0]


# 이제 테스트에 활용할 데이터를 얻어야 할 차례입니다.
# 
# 위에서 살펴본 바와 같이 KLUE 내 STS 데이터셋은 테스트 데이터셋을 포함하고 있지 않습니다.
# 
# 따라서 실습의 원활한 진행을 위해 다른 벤치마크 STS 데이터셋인 KorSTS 데이터셋을 다운로드 및 로드하여 사용하도록 하겠습니다.
# 
# (\* 두 데이터셋은 제작 과정이 엄밀히 다르므로, KLUE STS 데이터에 대해 학습된 모델이 KorSTS 테스트셋에 대해 기록하는 점수은 사실상 큰 의미가 없을 수 있습니다. 전체적인 훈련 프로세스의 이해를 돕기 위해 사용한다고 생각해주시는게 좋습니다.)

# In[13]:


testsets = load_dataset("kor_nli", "xnli")


# KorSTS 데이터셋은 훈련, 검증 그리고 테스트셋을 지니고 있습니다.

# In[14]:


testsets


# KorSTS의 예시 데이터도 마찬가지로 두 문장과 두 문장 간 유사도를 지니고 있습니다.

# In[15]:


testsets["test"][0]


# 이제 앞서 얻어진 데이터셋을 `sentence-transformers` 훈련 양식에 맞게 변환해주는 작업을 거쳐야 합니다.
# 
# 두 데이터 모두 0점에서 5점 사이의 값으로 유사도가 기록되었기 때문에, 0.0 ~ 1.0 스케일로 정규화를 시켜주는 작업을 거치게 됩니다.
# 
# (\* KorSTS 내 테스트셋의 경우 `None`으로 기록된 문장이 몇 개 존재하여, `None`을 걸러주는 조건이 추가되었습니다.)

# In[16]:


train_samples = []
dev_samples = []
test_samples = []
# 우리 입장에서 sentence1 = premise, sentence2 = hypothesis, labels=label
# KLUE STS 내 훈련, 검증 데이터 예제 변환
for phase in ["train", "validation"]:
    if phase=="train":
        examples = datasets[phase]
    elif phase=="validation":
        examples = testsets[phase]

    for example in examples:
        score = float(example["label"]) / 2.0  # 0.0 ~ 1.0 스케일로 유사도 정규화

        inp_example = InputExample(
            texts=[example["premise"], example["hypothesis"]],
            label=score,
        )

        if phase == "validation":
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# KorSTS 내 테스트 데이터 예제 변환
for example in testsets["test"]:
    score = float(example["label"]) / 2.0

    if example["premise"] and example["hypothesis"]:
        inp_example = InputExample(
            texts=[example["premise"], example["hypothesis"]],
            label=score,
        )

    test_samples.append(inp_example)


# 앞선 로직을 통해 각 데이터 예제는 다음과 같이 `InputExample` 객체로 변환되게 됩니다.

# In[17]:


train_samples[0].texts, train_samples[0].label


# In[18]:


test_samples[0].texts, test_samples[0].label


# 이제 학습에 사용될 `DataLoader`와 **Loss**를 설정해주도록 합니다.
# 
# `CosineSimilarityLoss`는 입력된 두 문장의 임베딩 간 코사인 유사도와 골드 라벨 간 차이를 통해 계산되게 됩니다.

# In[19]:


train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=train_batch_size,
)
train_loss = losses.CosineSimilarityLoss(model=model)


# 모델 검증에 활용할 **Evaluator** 를 정의해줍니다.
# 
# 앞서 얻어진 검증 데이터를 활용하여, 모델의 문장 임베딩 간 코사인 유사도가 얼마나 골드 라벨에 가까운지 계산하는 역할을 수행합니다.

# In[20]:


evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples,
    name="sts-dev",
)


# 모델 학습에 사용될 **Warm up Steps**를 설정합니다.
# 
# 다양한 방법으로 스텝 수를 결정할 수 있겠지만, 예제 노트북에서는 원 예제 코드를 따라 훈련 배치 수의 10% 만큼으로 값을 설정합니다.

# In[21]:


warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")


# 이제 앞서 얻어진 객체, 값들을 가지고 모델의 훈련을 진행합니다.
# 
# `sentence-transformers`에서는 다음과 같이 `fit` 함수를 통해 간단히 모델의 훈련과 검증이 가능합니다.
# 
# 훈련 과정을 통해 매 에폭 마다 얻어지는 체크포인트에 대해 *Evaluator* 가 학습된 모델의 코사인 유사도와 골드 라벨 간 피어슨, 스피어만 상관 계수를 계산해 기록을 남기게 됩니다.

# In[22]:


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model.to(device)
# Train 시킬 때 주석 풀기
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=10000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)


# In[ ]:


# model_save_path = "output/training_klue_sts_kykim-bert-kor-base-2023-12-31_15-22-38"# output/training_klue_sts_sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2-2024-01-07_23-02-55
# print(model_save_path)


# 학습이 완료되었다면 이제 학습된 모델을 테스트 할 시간입니다.
# 
# 앞서 KorSTS 데이터를 활용해 구축한 테스트 데이터셋을 앞서와 마찬가지로 *Evaluator* 로 초기화해주도록 합니다.

# In[ ]:

print(model_save_path)
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')


# 이제 테스트 *Evaluator* 를 활용하여 테스트셋에 대해 각 상관 계수를 계산하도록 합니다.

# In[ ]:


test_evaluator(model, output_path=model_save_path)


# 역시 검증 데이터에 비해 좋지 않은 점수를 기록하였습니다.
# 
# KLUE 내 검증 데이터셋 중 일부를 샘플링하여 테스트셋으로 활용하는 방안도 있겠지만,
# 
# 본 노트북은 전체 훈련 프로세스를 파악하는데 초점을 맞추었으므로 실험을 마치도록 합니다.

# ## Sentence Transformers 활용

# In[ ]:


path = os.getcwd()
data_path = os.path.abspath(os.path.join(os.path.join(path, "SU"),"crawling code"))
print(path)
print(data_path)


# In[ ]:


# 3단계(150개 상품 = 15개, 100개 상품 = 10개, 50개 상품 = 5개, 25개 상품 = 3개)
Top = [1307, 1386, 1554, 2315, 3153, 3248, 3605, 3821, 3924, 3931, 4253, 4301, 4376, 4385, 4478]
Outer = [692, 848, 1154, 1372, 2115, 3130, 3272, 3549, 3624, 3852, 3893, 4136, 4183, 4320, 4489]
Pants = [1863, 2614, 2619, 2924, 3075, 3222, 3400, 3520, 3565, 3657, 3930, 4112, 4182, 4410, 4474]
Skirt = [1936, 2335, 2582, 2650, 2729, 2767, 2812, 2897, 2958, 2996]
Onepiece = [303, 1299, 1367, 1406, 1496]
Sport_Top = [4, 219, 1145, 1346, 1470]
Sport_Outer = [31, 1159, 1233, 1368, 1470]
Sport_Pants = [4, 550, 715]
# 0단계(각 상품당(Sport 제외) 3개씩 가져오기)
# Zero_Top = [1559, 2092, 4330]
# Zero_Outer = [654, 1616, 2908]
# Zero_Pants = [593, 3737, 4091]
# Zero_Skirt = [22, 1700, 2829]
# Zero_Onepiece = [347, 1046]
# 1단계(150개 상품 = 15개 이상, 100개 상품 = 10개 이상, 50개 상품 = 5개 이상, 25개 상품 = 3개 이상)
One_Top = [42, 162, 282, 296, 345, 1036, 1045, 1376, 1677, 1692, 1717, 2403, 2765, 2769, 3293, 3574, 3579, 3582, 3586, 3976]
One_Outer = [934, 1344, 1399, 1404, 1448, 1616, 1683, 2464, 2898, 2533,  3028, 3477, 3591, 4287] # 2692
One_Pants = [20, 69, 122, 379, 382, 386, 559, 560, 595, 737, 760, 2533, 2743, 4188]# 892
One_Skirt = [22, 71, 128, 232, 472, 684, 838, 1077, 1246, 1250, 1374]
One_Onepiece = [178, 118, 271, 563, 951]
One_Sport_Top = [55, 200, 452, 666, 962]
One_Sport_Outer = [682, 1013, 1230, 1418]# 1244
One_Sport_Pants = [59, 87, 190, 299]# 52

List = ["Top", "Outer","Pants","Skirt","Onepiece","Sport_Top","Sport_Outer","Sport_Pants"]
Zero_List = ["Top", "Outer","Pants","Skirt","Onepiece"]
review = []
zero_review = []#["옷"]#["맛있어요","앜ㅋㅋㅋㅋㅋㅋ존잼ㅠㅠ","최고의 음식이에요","감동적인 책이에요","자세한 상담을 원하시면 연락주시기 바랍니다.","새싹이 돋아나고 있어요", "웅성웅성 수근수근","외계인이 침략했어요","나나투어 많관부","우주랑 하마를 합치면 우주하마","목도리랑 도마뱀을 합치면 목도리도마뱀","흔들리는 꽃들 속에서 네 샴푸향이 느껴진거야","와 샌즈 아시는구나! 겁나 어.렵.습.니.다!","누가 코딩해주실 분?"]
one_review = []
test = []
# spacing = Spacing()
for category in List:
    globals()[f"text_{category}"] = pd.read_csv(os.path.join(data_path,f"{category}_List.csv"),header=0)
    category_num = globals()[f"text_{category}"].shape[0]
    review_temp = [globals()[f"text_{category}"].iloc[index,6] for index in globals()[f"{category}"]]
    one_review_temp = [globals()[f"text_{category}"].iloc[index,6] for index in globals()[f"One_{category}"]]
    test_temp = [globals()[f"text_{category}"].iloc[index,6] for index in range(category_num)]    
    review.extend(review_temp)
    one_review.extend(one_review_temp)
    test.extend(test_temp)
    # if category in Zero_List:
    #     zero_review_temp = [spacing(globals()[f"text_{category}"].iloc[index,6]) for index in globals()[f"Zero_{category}"]]
    #     zero_review.extend(zero_review_temp)


# In[ ]:


# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# Tokenize sentences
review_0 = []
review_1 = []
review_2 = []
review_3 = []
Min_List = []
Mean_List = []
Max_List = []
one_Mean_List = []
one_Max_List = []

embedding_review = model.encode(review)
embedding_one_review = model.encode(one_review)
for index, value in enumerate(test):
    # print(index, value)
    # for i in review:
    embedding_value = model.encode(value)
    # embedding_zero_review = model.encode(zero_review)
    top_k = 20

    # 입력 문장 - 문장 후보군 간 코사인 유사도 계산 후,
    # cos_scores_0 = util.pytorch_cos_sim(embedding_value, embedding_zero_review)[0]
    cos_scores_1 = util.pytorch_cos_sim(embedding_value, embedding_one_review)[0]
    cos_scores_3 = util.pytorch_cos_sim(embedding_value, embedding_review)[0]

    # 코사인 유사도 순으로 `top_k` 개 문장 추출
    # top_results_0 = np.mean(np.array(torch.topk(cos_scores_0, k=len(zero_review))[0]))# 전부 가져오기
    top_results_1 = torch.topk(cos_scores_1, k=len(one_review))# 전부 가져오기
    top_results_3 = torch.topk(cos_scores_3, k=top_k)#len(review))
    MIN_num = min(top_results_3[0])
    MEAN_num = np.mean(np.array(top_results_3[0]))
    MAX_num = max(top_results_3[0])
    One_MEAN_num = np.mean(np.array(top_results_1[0]))
    One_MAX_num = max(top_results_1[0])
    Min_List.append(MIN_num)
    Mean_List.append(MEAN_num)
    Max_List.append(MAX_num)
    one_Mean_List.append(One_MEAN_num)
    one_Max_List.append(One_MAX_num)
    if One_MEAN_num<=0.35 or MEAN_num<=0.31:
        print(value)
        print(f"{index} 1 : (Mean : %.4f) \t 2, 3(Min: %.4f \t Mean: %.4f \t Max: %.4f)"%(One_MEAN_num, MIN_num,MEAN_num,MAX_num))
    if One_MAX_num>=0.995:
        review_1.append(value)
    elif MEAN_num>=0.997 and len(value)>=96:
        review_3.append(value)
    elif MEAN_num>=0.36:
        review_2.append(value)
    else:
        review_0.append(value)


# In[ ]:


plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.hist(Min_List,bins=15,label = "Min")
plt.xlim(0,1)
plt.title("Min histogram")
plt.legend()
plt.grid()
plt.subplot(1,3,2)
plt.hist(Mean_List,bins=15,label = "Mean")
plt.xlim(0,1)
plt.title("Mean histogram")
plt.legend()
plt.grid()
plt.subplot(1,3,3)
plt.hist(Max_List,bins=15,label = "Max")
plt.xlim(0,1)
plt.title("Max histogram")
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.hist(one_Mean_List,bins=15,label = "Mean")
plt.xlim(0,1)
plt.title("One histogram")
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.hist(one_Max_List,bins=15,label = "Max")
plt.xlim(0,1)
plt.title("One histogram")
plt.legend()
plt.grid()
plt.show()
print(np.percentile(Mean_List, 0.1))
print(np.percentile(Mean_List, 50))
print(np.percentile(Mean_List, 95))
print(np.percentile(one_Mean_List, 0.1))
print(np.percentile(one_Max_List, 96))


# In[ ]:


for i in range(4):
    print("*"*100)
    print("%d단계: 총 %d개"%(i,len(globals()[f"review_{i}"])))
    print("*"*100)
    for index,value in enumerate(globals()[f"review_{i}"]):
        print(index, value)
        print()


# In[ ]:




