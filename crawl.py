import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from serpapi import GoogleSearch
import re

api_key = "c4401cac8b81c69e25c103b98d3c9ea1631d1ed46f82ca9f06290d880849b121" 

def get_webpage_info(url):
   
    text = []
    
    response = requests.get(url)
    response.raise_for_status()  # Kiểm tra lỗi HTTP
    soup = BeautifulSoup(response.content, "html.parser")

    target_tags = ['p', 'h1', 'span', 'li', 'a', 'td', 'th']  
    text_tags = soup.find_all(target_tags)
    for tag in text_tags:
        cleaned_text = re.sub(r'\s+', ' ', tag.text.strip())
        text.append(cleaned_text)
    return text
  
    



def search_google(query):
    params = {
        "q": query,  # Truy vấn tìm kiếm
        "api_key": api_key  # API key của bạn
    }

    search = GoogleSearch(params)
    results = search.get_dict()  # Lấy kết quả dưới dạng từ điển

    return results

def crawl_data(query,max_page = 5):
    text = []
    search_results = search_google(query)
    for result in search_results["organic_results"][:max_page]:
        link = result['link']
        try:
            text += get_webpage_info(link) 
        except:
            pass

    # save data to temporary_file.txt
    if len(text) != 0:
        with open('temporary_data.txt','w+') as f:
            f.write(" ".join(text))


