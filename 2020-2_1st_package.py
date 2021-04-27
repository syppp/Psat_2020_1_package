#!/usr/bin/env python
# coding: utf-8

# In[1]:


#0번
import os
import pandas
os.chdir("C:/Users/samsung/Desktop/대학교/3학년 2학기/피셋/패키지/2주차 패키지_파이썬")


# In[2]:


#1번
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
path = "C://Users/samsung//Desktop//대학교//3학년 2학기//피셋//패키지//2주차 패키지_파이썬//chromedriver.exe"
driver = webdriver.Chrome(path)


# In[3]:


import time

#music
music_title = []
music_sub = []
music_article = []

for k in range(1,3):
    url = "http://www.koreaherald.com/list.php?ct=020403000000&np=%d&mp=1"%k
    driver.get(url)
    for i in range(1,16):
        url = "http://www.koreaherald.com/list.php?ct=020403000000&np=%d&mp=1"%k
        driver.get(url)
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i))
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i)).click()
        time.sleep(1)
        source = driver.page_source
        soup = BeautifulSoup(source,'html.parser')
        music_title.append(soup.select('body > div.main > div > div.view_bg > div.view > h1'))
        music_sub.append(soup.select('body > div.main > div > div.view_bg > div.view > h2'))
        music_article.append(soup.select('#articleText'))
        
music = pd.DataFrame(list(zip(music_title,music_sub,music_article)),columns = ['title','sub','article'])


# In[14]:


#politics
pol_title = []
pol_sub = []
pol_article = []
for k in range(1,3):
    url = "http://www.koreaherald.com/list.php?ct=020101000000&np=%d&mp=1"%k
    driver.get(url)
    for i in range(1,16):
        url = "http://www.koreaherald.com/list.php?ct=020101000000&np=%d&mp=1"%k
        driver.get(url)
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i))
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i)).click()
        time.sleep(1)
        source = driver.page_source
        soup = BeautifulSoup(source,'html.parser')
        pol_title.append(soup.select('body > div.main > div > div.view_bg > div.view > h1'))
        pol_sub.append(soup.select('body > div.main > div > div.view_bg > div.view > h2'))
        pol_article.append(soup.select('#articleText'))
        
pol = pd.DataFrame(list(zip(pol_title,pol_sub,pol_article)),columns = ['title','sub','article'])
pol


# In[16]:


#film
film_title = []
film_sub = []
film_article = []
for k in range(1,3):
    url = "http://www.koreaherald.com/list.php?ct=020401000000&np=%d&mp=1"%k
    driver.get(url)
    for i in range(1,16):
        url = "http://www.koreaherald.com/list.php?ct=020401000000&np=%d&mp=1"%k
        driver.get(url)
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i))
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i)).click()
        time.sleep(1)
        source = driver.page_source
        soup = BeautifulSoup(source,'html.parser')
        film_title.append(soup.select('body > div.main > div > div.view_bg > div.view > h1'))
        film_sub.append(soup.select('body > div.main > div > div.view_bg > div.view > h2'))
        film_article.append(soup.select('#articleText'))
        
film = pd.DataFrame(list(zip(film_title,film_sub,film_article)),columns = ['title','sub','article'])        
film


# In[17]:


#economics
eco_title = []
eco_sub = []
eco_article = []
for k in range(1,3):
    url = "http://www.koreaherald.com/list.php?ct=021901000000&np=%d&mp=1"%k
    driver.get(url)
    for i in range(1,16):
        url = "http://www.koreaherald.com/list.php?ct=021901000000&np=%d&mp=1"%k
        driver.get(url)
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i))
        driver.find_element_by_xpath("/html/body/div[3]/div/div[1]/ul/li[%d]/a"%(i)).click()
        time.sleep(1)
        source = driver.page_source
        soup = BeautifulSoup(source,'html.parser')
        eco_title.append(soup.select('body > div.main > div > div.view_bg > div.view > h1'))
        eco_sub.append(soup.select('body > div.main > div > div.view_bg > div.view > h2'))
        eco_article.append(soup.select('#articleText'))
        
eco = pd.DataFrame(list(zip(eco_title,eco_sub,eco_article)),columns = ['title','sub','article'])        
eco


# In[18]:


#2번
import pandas as pd
data = pd.read_csv("data.csv",encoding='utf-8')
type(data)
data.head(n=5)
data.tail(n=5)


# In[20]:


#3번
#3-1 (영어가 아닌 텍스트 제거하기, df.str.replace() 함수와 정규식을 사용해보세요)
import re

data['article'] = data['article'].str.replace(pat=r'[^a-zA-Z]',repl= r' ',regex=True)


# In[21]:


# 3-2 글자가 3자이하인 단어들을 모두 삭제하세요
#힌트: split함수를 이용하여 문장을 단어로 쪼개준 후 3개 이하인 단어들을 제거한 후 join함수를 이용해 다시 문장으로 만들어주세요.
list1 =[]
for i in range(0,4789):
    list1 =[]
    data['article'][i] = str(data['article'][i]).split()
    for j in data['article'][i]:
        if(len(j) > 3):
            list1.append(str(j))
            
    data['article'][i] = list1
    data['article'][i] = " ".join(data['article'][i])
data.head(n=5)


# In[22]:


#3-3. 모든 단어들을 소문자로 바꿔주세요.
for i in range(0,4789):
    data['article'][i] = data['article'][i].lower()
    


# In[435]:


#data.to_csv("data2.csv")

