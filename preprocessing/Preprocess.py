# -*- coding: utf-8 -*-
"""Preprocess

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XYsTYZT4KkqTk2KkfCKZxtpdlJea0iAX
"""

import requests
import json
import time
import sys
from collections import OrderedDict, namedtuple
import xml.etree.ElementTree as ET
import re
from urllib import parse
import pandas as pd
import urllib.request

"""## 네이버 맞춤법 코드

"""

# 조사와 어미도 단어로 처리함. 마땅한 영단어가 생각이 안 나서..
_checked = namedtuple('Checked',
    ['result', 'original', 'checked', 'errors', 'words', 'time'])


class Checked(_checked):
    def __new__(cls, result=False, original='', checked='', errors=0, words=[], time=0.0):
        return super(Checked, cls).__new__(
            cls, result, original, checked, errors, words, time)

    def as_dict(self):
        d = {
            'result': self.result,
            'original': self.original,
            'checked': self.checked,
            'errors': self.errors,
            'words': self.words,
            'time': self.time,
        }
        return d

    def only_checked(self):
        return self.checked

class CheckResult:
    PASSED = 0
    WRONG_SPELLING = 1
    WRONG_SPACING = 2
    AMBIGUOUS = 3
    STATISTICAL_CORRECTION = 4

base_url = 'https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy'

_agent = requests.Session()
PY3 = sys.version_info[0] == 3

def read_token():
    try :
      with open("token.txt", "r") as f:
          TOKEN = f.read()
      return TOKEN
    except :
      return "-"

def update_token(agent):
    """update passportkey
    from https://gist.github.com/AcrylicShrimp/4c94db38b7d2c4dd2e832a7d53654e42
    """

    html = agent.get(url='https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=맞춤법검사기')

    match = re.search('passportKey=([a-zA-Z0-9]+)', html.text)
    if match is not None:
        TOKEN = parse.unquote(match.group(1))
        with open("token.txt", "w") as f:
            f.write(TOKEN)

    return TOKEN

def _remove_tags(text):
    text = u"<content>{}</content>".format(text).replace("<br>","\n")
    if not PY3:
      text = text.encode("utf-8")

    result = "".join(ET.fromstring(text).itertext())

    return result

def _get_data(text, token):
    payload = {
        "_callback": "window.__jindo2_callback._spellingCheck_0",
        "q": text,
        "color_blindness": 0,
        "passportKey": token
    }
    headers = {
        "Host": "m.search.naver.com",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.9 Safari/537.36",
        "referer": "https://search.naver.com/",
        "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept": "*/*"
    }
    start_time = time.time()
    r = _agent.get(base_url, params=payload, headers=headers)
    passed_time = time.time() - start_time
    r = r.text[42:-2]
    data = json.loads(r)
    return passed_time, data

def naver_covert_spelling(text):
    """
    매개변수로 입력받은 한글 문장의 맞춤법을 체크합니다.
    """
    if isinstance(text, list):
        result = []
        for item in text:
            checked = naver_covert_spelling(item)
            result.append(checked)
        return result

    # 최대 500자까지 가능.
    if len(text) > 500:
        return Checked(result=False)

    TOKEN = read_token()
    passed_time, data = _get_data(text, TOKEN)
    if "error" in data["message"].keys():
        TOKEN = update_token(_agent)
        passed_time, data = _get_data(text, TOKEN)
        if "error" in data["message"].keys():
            return Checked(result=False)
    html = data["message"]["result"]["html"]
    result = {
        "result": True,
        "original": text,
        "checked": _remove_tags(html),
        "errors": data["message"]["result"]["errata_count"],
        "time": passed_time,
        "words": OrderedDict(),
    }

    # 띄어쓰기로 구분하기 위해 태그는 일단 보기 쉽게 바꿔둠.
    # ElementTree의 iter()를 써서 더 좋게 할 수 있는 방법이 있지만
    # 이 짧은 코드에 굳이 그렇게 할 필요성이 없으므로 일단 문자열을 치환하는 방법으로 작성.
    html = html.replace("<span class=\"green_text\">", "<green>") \
               .replace("<span class=\"red_text\">", "<red>") \
               .replace("<span class=\"purple_text\">", "<purple>") \
               .replace("<span class=\"blue_text\">", "<blue>") \
               .replace("</span>", "<end>")
    items = html.split(" ")
    words = []
    tmp = ""
    for word in items:
        if tmp == "" and word[:1] == "<":
            pos = word.find(">") + 1
            tmp = word[:pos]
        elif tmp != "":
            word = u"{}{}".format(tmp, word)

        if word[-5:] == "<end>":
            word = word.replace("<end>", "")
            tmp = ""

        words.append(word)

    for word in words:
        check_result = CheckResult.PASSED
        if word[:5] == "<red>":
            check_result = CheckResult.WRONG_SPELLING
            word = word.replace("<red>", "")
        elif word[:7] == "<green>":
            check_result = CheckResult.WRONG_SPACING
            word = word.replace("<green>", "")
        elif word[:8] == "<purple>":
            check_result = CheckResult.AMBIGUOUS
            word = word.replace("<purple>", "")
        elif word[:6] == "<blue>":
            check_result = CheckResult.STATISTICAL_CORRECTION
            word = word.replace("<blue>", "")
        result["words"][word] = check_result

    result = Checked(**result)

    return result.checked

"""## 부산대 맞춤법 코드

"""

def busan_covert_spelling(text):
  response = requests.post('http://164.125.7.61/speller/results', data={'text1': text})
  data = response.text.split('data = [', 1)[-1].rsplit('];', 1)[0]
  # JSON 디코딩 시도
  data = json.loads(data)

  # 이후에 수행할 작업 추가
  for error_text in data['errInfo']:
    err_word = error_text["orgStr"]
    corr_word = error_text["candWord"].split("|")[0]
    text = text.replace(err_word,corr_word)

  return text


from soynlp.normalizer import repeat_normalize

"""*기타 전처리*"""

my_stopwords = []

# 불용어 사전에 한국어 자음, 모음 추가
korean_consonants = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ"
korean_vowels = "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"

my_stopwords.extend(korean_consonants)
my_stopwords.extend(korean_vowels)

my_stopwords.extend(['😂',
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
 '&'])

my_stopwords = list(set(my_stopwords))
my_stopwords



class NLP_Preprocessor:

    # 함수 : 숫자는 [number]로 변환 +  + 불용어 제거
    def delete_stopwords(self, context):

        context = re.sub(r'ㄱㅊ', '괜찮', context)
        context = re.sub(r'\d+', '[number]', context)

        for stopword in my_stopwords:
            context = context.replace(stopword, '')
        return context


    # 함수 : 위 4개의 함수 한번에 실행
    def preprocess(self, context):
        context = re.sub(r'ㄱㅊ', '괜찮', context)
        context = re.sub(r'\d+', '[number]', context)

        context = self.delete_stopwords(context)

        context = repeat_normalize(context, num_repeats=2).strip()
        context = re.sub(r"\s+", " ", context).strip()
        return context

