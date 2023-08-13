import requests
from bs4 import BeautifulSoup

url = "https://www.gov.uk/find-local-council"

payload={'postcode': 'RG6 7DD'}
files=[

]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

soup = BeautifulSoup(response.text, 'html.parser' )

if soup.find_all("div", {"class": "district-result group"}):
    res = soup.find_all("div", {"class": "district-result group"})
    data = str(res[0].find_all("h3")[0].text).strip()
    print(data)

elif soup.find_all("div", {"class": "unitary-result group"}):
    res = soup.find_all("div", {"class": "unitary-result group"})
    data = str(res[0].find_all("strong")[0].text).strip()
    print(data)