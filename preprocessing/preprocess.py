# <라이브러리 호출>

import requests
import json
import time
import sys
from collections import namedtuple, OrderedDict
import xml.etree.ElementTree as ET
import re
from urllib import parse
import pandas as pd
import urllib.request
from soynlp.normalizer import *



# <네이버 맞춤법 검사기>
# 설명 : 1) 오타 수정, 2) 띄어쓰기 수정

class Naver:
    base_url = 'https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy'
    _agent = requests.Session()
    PY3 = sys.version_info[0] == 3

    class CheckResult:
        PASSED = 0
        WRONG_SPELLING = 1
        WRONG_SPACING = 2
        AMBIGUOUS = 3
        STATISTICAL_CORRECTION = 4

    _Checked = namedtuple('Checked', ['result', 'original', 'checked', 'errors', 'words', 'time'])

    def __init__(self):
        self.token = self.read_token()

    def read_token(self):
        try:
            with open("token.txt", "r") as f:
                return f.read()
        except:
            return "-"

    def update_token(self):
        html = self._agent.get(url='https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=맞춤법검사기')
        match = re.search('passportKey=([a-zA-Z0-9]+)', html.text)
        if match is not None:
            self.token = parse.unquote(match.group(1))
            with open("token.txt", "w") as f:
                f.write(self.token)

    def _remove_tags(self, text):
        text = u"<content>{}</content>".format(text).replace("<br>", "\n")
        if not self.PY3:
            text = text.encode("utf-8")
        return "".join(ET.fromstring(text).itertext())

    def _get_data(self, text):
        payload = {
            "_callback": "window.__jindo2_callback._spellingCheck_0",
            "q": text,
            "color_blindness": 0,
            "passportKey": self.token
        }
        headers = {
            "Host": "m.search.naver.com",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.9 Safari/537.36",
            "referer": "https://search.naver.com/",
            "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept": "*/*"
        }
        start_time = time.time()
        response = self._agent.get(self.base_url, params=payload, headers=headers)
        passed_time = time.time() - start_time
        response_data = response.text[42:-2]
        return passed_time, json.loads(response_data)

# 네이버 맞춤법 검사에 사용할 함수
    def convert_spelling(self, text):
        if isinstance(text, list):
            return [self.convert_spelling(item) for item in text]

        if len(text) > 500:
            return self._Checked(result=False, original=text, checked='', errors=0, words=[], time=0.0)

        passed_time, data = self._get_data(text)
        if "error" in data["message"]:
            self.update_token()
            passed_time, data = self._get_data(text)
            if "error" in data["message"]:
                return self._Checked(result=False, original=text, checked='', errors=0, words=[], time=0.0)

        html = data["message"]["result"]["html"]
        checked_text = self._remove_tags(html)
        errors = data["message"]["result"]["errata_count"]
        words = self._process_words(html)
        return self._Checked(result=True, original=text, checked=checked_text, errors=errors, words=words, time=passed_time)



# 네이버 맞춤법 검사에 사용할 함수를
# 우리 조의 Task에 맞는 전처리 방법으로 수정한 함수 
    def Convert_spelling(self, sentences):
        sentences = list(sentences)

        result = []
        for i, sentence in enumerate(sentences):

            if i % 100 == 0:
                print(f"{i}번째 Index 진행중")
                time.sleep(1)            
            
            try:
                checked_sentence = self.convert_spelling(sentence).checked
                result.append(checked_sentence)

            except Exception as e:
                print(f"Error: {e}", f"/ 설명 : {i}번째 Index에서 서버 오류가 발생하였습니다. 5초 후, {i+1}번째 Index부터 다시 시작하도록 하겠습니다!")
                result.append(sentence)
                time.sleep(5)
                continue

        
        return pd.Series(result)



    def _process_words(self, html):
        words_dict = OrderedDict()
        html = html.replace("<span class=\"green_text\">", "<green>") \
                   .replace("<span class=\"red_text\">", "<red>") \
                   .replace("<span class=\"purple_text\">", "<purple>") \
                   .replace("<span class=\"blue_text\">", "<blue>") \
                   .replace("</span>", "<end>")
        items = html.split(" ")
        temp_tag = ""
        for word in items:
            if temp_tag == "" and word[:1] == "<":
                pos = word.find(">") + 1
                temp_tag = word[:pos]
            elif temp_tag != "":
                word = f"{temp_tag}{word}"

            if word.endswith("<end>"):
                word = word.replace("<end>", "")
                temp_tag = ""

            check_result = self.CheckResult.PASSED
            if word.startswith("<red>"):
                check_result = self.CheckResult.WRONG_SPELLING
                word = word.replace("<red>", "")
            elif word.startswith("<green>"):
                check_result = self.CheckResult.WRONG_SPACING
                word = word.replace("<green>", "")
            elif word.startswith("<purple>"):
                check_result = self.CheckResult.AMBIGUOUS
                word = word.replace("<purple>", "")
            elif word.startswith("<blue>"):
                check_result = self.CheckResult.STATISTICAL_CORRECTION
                word = word.replace("<blue>", "")
            words_dict[word] = check_result

        return words_dict



class NLP_Preprocessor:

# 정제

    # 함수 : 오타 수정 (채택)
    def delete_welcome(self, context):
        preprocessed_text = []
        for text in context:
          text = re.sub(r'ㄱㅊ', '괜찮', text).strip()
          text = re.sub(r'미춋어', '미쳤어', text).strip()
          text = re.sub(r'젹당', '적당', text).strip()
          text = re.sub(r'댱', '다', text).strip()
          text = re.sub(r'냤습', '났습', text).strip()
          text = re.sub(r'괜춘|괜츦|괜추니|괜탆|괘않', '괜찮', text).strip()
          text = re.sub(r'후즐', '후줄', text).strip()
          text = re.sub(r'급니|슺니', '습니', text).strip()
          text = re.sub(r'늨', '는', text).strip()
          text = re.sub(r'느껨', '느낌', text).strip()
          text = re.sub(r'진짲', '진짜', text).strip()
          text = re.sub(r'정망', '정말', text).strip()
          text = re.sub(r'줳', '줬', text).strip()
          text = re.sub(r'귀염|기욤|귀얍|귀욤', '귀여움', text).strip()
          text = re.sub(r'재딜', '재질', text).strip()
          text = re.sub(r'페딩', '패딩', text).strip()
          text = re.sub(r'뻡뻐', '뽀뽀', text).strip()
          text = re.sub(r'포잊트', '포인트', text).strip()
          text = re.sub(r'욬|욯', '요', text).strip()
          text = re.sub(r'융', '용', text).strip()
          text = re.sub(r'댱', '당', text).strip()
          text = re.sub(r'입읗수', '입을수', text).strip()
          text = re.sub(r'완젘', '완전', text).strip()
          text = re.sub(r'듷', '듯', text).strip()
          text = re.sub(r'삿', '샀', text).strip()
          text = re.sub(r'슺', '습', text).strip()
          text = re.sub(r'따둣|따뚯|뜨뜻', '따뜻', text).strip()
          text = re.sub(r'윽시', '역시', text).strip()
          text = re.sub(r'쥼', '줌', text).strip()
          text = re.sub(r'햤', '했', text).strip()
          text = re.sub(r'읗', '을', text).strip()
          text = re.sub(r'완존', '완전', text).strip()
          text = re.sub(r'럈', '랐', text).strip()
          text = re.sub(r'쩗', '짧', text).strip()
          text = re.sub(r'젛', '좋', text).strip()
          text = re.sub(r'귣|귯', '굿', text).strip()
          text = re.sub(r'죵|죻', '죠', text).strip()
          text = re.sub(r'앖|웂|옶', '없', text).strip()
          text = re.sub(r'쥬아', '좋아', text).strip()
          text = re.sub(r'조음', '좋음', text).strip()
          text = re.sub(r'예뿝', '예쁩', text).strip()
          text = re.sub(r'이뽀|애쁘|에쁘', '예쁘', text).strip()
          text = re.sub(r'앆', '았', text).strip()
          text = re.sub(r'웈동', '운동', text).strip()
          text = re.sub(r'댜', '다', text).strip()
          text = re.sub(r'댕겨', '다녀', text).strip()
          text = re.sub(r'엔간', '웬만', text).strip()
          text = re.sub(r'엇', '었', text).strip()
          text = re.sub(r'근대', '근데', text).strip()
          text = re.sub(r'믄', '는', text).strip()
          text = re.sub(r'뺠', '뺄', text).strip()
          text = re.sub(r'하쥬', '하죠', text).strip()
          text = re.sub(r'시로', '싫어', text).strip()
          if text:
            preprocessed_text.append(text)
        return preprocessed_text



    # 함수 : 불용어 제거 (채택)
    def delete_stopwords(self, context):
        stopwords =  ['ㄱ',
                      'ㄴ',
                      'ㄷ',
                      'ㄹ',
                      'ㅁ',
                      'ㅂ',
                      'ㅅ',
                      'ㅇ',
                      'ㅈ',
                      'ㅊ',
                      'ㅋ',
                      'ㅌ',
                      'ㅍ',
                      'ㅎ',
                      'ㅏ',
                      'ㅑ',
                      'ㅓ',
                      'ㅕ',
                      'ㅗ',
                      'ㅛ',
                      'ㅜ',
                      'ㅠ',
                      'ㅡ',
                      'ㅣ',
                      '😂',
                      '🥺',
                      '💕',
                      '🧸',
                      '💡',
                      '😉',
                      '🥰',
                      '😅',
                      '\U0001fa76',
                      '😋',
                      '\U0001faf0',
                      '😄',
                      '🤔',
                      '🙉',
                      '\U0001fa77',
                      '💚',
                      '🚛',
                      '😘',
                      '👏',
                      '😆',
                      '😏',
                      '🥶',
                      '💜',
                      '😁',
                      '🤗',
                      '👉',
                      '🔥',
                      '😀',
                      '😊',
                      '😙',
                      '💓',
                      '🙃',
                      '😭',
                      '💞',
                      '😃',
                      '🥲',
                      '🌀',
                      '🤧',
                      '💗',
                      '\U0001fae0',
                      '😍',
                      '💖',
                      '\U0001fae8',
                      '🏼',
                      '😎',
                      '🙏',
                      '\U0001f979',
                      '🥵',
                      '🤍',
                      '🍀',
                      '💙',
                      '😹',
                      '🤭',
                      '🤣',
                      '🏾',
                      '🤤',
                      '👀',
                      '🙂',
                      '\U0001fa75',
                      '🏻',
                      '😻',
                      '\U0001fae3',
                      '😽',
                      '🖤',
                      '\U0001faf6',
                      '👍',
                      '⭐️',
                      '✅',
                      '&',
                      '🥦',
                      '🤎',
                      '💛',
                      '🤍',
                      '\U0001faf6🏻',
                      '🐻',
                      'ㄲ',
                      'ㄸ',
                      'ㅃ',
                      'ㅆ',
                      'ㅉ',
                      'ㅐ',
                      'ㅔ',
                      'ㅒ',
                      'ㅖ',
                      'ㅘ',
                      'ㅚ',
                      'ㅙ',
                      'ㅞ',
                      'ㅟ',
                      'ㅝ',
                      'ㅢ',
                      # 특수문자
                      '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
                      # 1차
                      '◠', '‿', '◜', 'ᴗ', '´', 'ㆅ', 'ゞ', 'ദ്ദി', '˶', '˙', 'ᵕ', '´', '༎ຶ',
                      'ٹ', '⌓', '.', '<', 'ʺ̤', '♡', '北', '⸌', '☻', '⸍', '‼', '◡̎', '༡',
                      '˃', 'ꇴ', '˂', '๑', 'ꉂ', '◟', '´', '◦', 'ω', '➰', 'ᐟ', '⑉', '꒦ິ', ';',
                      '☆', '•', 'ᆢ'

                      ]

        preprocessed_text = []
        for text in context:
            for stopword in stopwords:
                text = text.replace(stopword, '')
            preprocessed_text.append(text)
        return preprocessed_text



    # 함수 : HTML 태그를 제거
    def delete_html_tag(self, context):
      preprcessed_text = []
      for text in context:
          text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text).strip()
          if text:
              preprcessed_text.append(text)
      return preprcessed_text



    # 함수 : 이메일 제거
    def delete_email(self, context):
      preprocessed_text = []
      for text in context:
        text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
        if text:
          preprocessed_text.append(text)
      return preprocessed_text



    # 함수 : 해시태그(#) 제거 (채택)
    def delete_hashtag(self, context):
      preprocessed_text = []
      for text in context:
        text = re.sub(r"#\S+", "", text).strip()
        if text:
          preprocessed_text.append(text)
      return preprocessed_text



    # 함수 : 이모티콘 제거 (채택)
    def delete_emoticons(self, context):
      preprocessed_text = []
      for text in context:
        text = re.sub(r'[\U00010000-\U0010ffff]', "", text).strip()
        if text:
          preprocessed_text.append(text)
      return preprocessed_text



# 정규화

    # 함수 : 반복 횟수 정규화 (채택)
    def repeat_char_normalizer(self, context):
        normalized_text = []
        for text in context:
            text = repeat_normalize(text, num_repeats=2).strip()
            if text:
                normalized_text.append(text)
        return normalized_text



    # 함수 : 띄어쓰기 정규화 (채택)
    def repeated_spacing_normalizer(self, context):
        normalized_text = []
        for text in context:
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                normalized_text.append(text)
        return normalized_text



    # 함수 : 중복 문장 정규화
    def duplicated_sentence_normalizer(self, context):
        from collections import OrderedDict
        context = list(OrderedDict.fromkeys(context))
        return context



    # 함수 : 문장을 최대, 최소 길이로 필터링하는 정규화
    def min_max_filter(self, context):
        preprocessed_text = []
        for text in context:
          if 0 < len(text) and len(text) < 250: # 문장 최대, 최소 길이 지정해주기
            preprocessed_text.append(text)
        return preprocessed_text



# 전처리 함수

    # 함수 : 첫번째 전처리
    def preprocess_first(self, context):
        context = list(context)
        # 정제
        context1 = self.delete_welcome(context)
        context2 = self.delete_stopwords(context1)
        context3 = self.delete_emoticons(context2)
        context4 = self.delete_hashtag(context3)
        # 정규화
        context5 = self.repeat_char_normalizer(context4)
        context6 = self.repeated_spacing_normalizer(context5)

        return pd.Series(context6)



    # 함수 : 두번째 전처리
    def preprocess_second(self, context):

        return context


