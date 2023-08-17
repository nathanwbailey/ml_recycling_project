import requests
from bs4 import BeautifulSoup
import json

def get_council_data(postcode):
    url = "https://www.gov.uk/find-local-council"

    payload={'postcode': postcode}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    soup = BeautifulSoup(response.text, 'html.parser' )
    data = None
    if soup.find_all("div", {"class": "district-result group"}):
        res = soup.find_all("div", {"class": "district-result group"})
        data = str(res[0].find_all("h3")[0].text).strip()

    elif soup.find_all("div", {"class": "unitary-result group"}):
        res = soup.find_all("div", {"class": "unitary-result group"})
        data = str(res[0].find_all("strong")[0].text).strip()

    return data

def get_from_database(council_name):
    data = {}
    with open('council_database.json', 'r') as f:
        data = json.load(f)
    data_entries = data['entries']
    council_entry = None
    for entry in data_entries:
        if entry['council'] == council_name:
            council_entry = entry.copy()
            del council_entry['council']     
    return council_entry

# data_out = get_from_database(get_council_data('RG6 7DD'))
# print(data_out)