from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults

from multiprocessing import Pool
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests


def scrape_url(url):
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache",
            },
        )
    except Exception as e:
        response = None
    if response is not None and response.status_code == 200:
        html = response.text
    else:
        return url, "Access Denied"
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    filtered_chunks = list(chunks)
    filtered_chunks = list(filter(lambda x: len(x) > 100, filtered_chunks))
    # drop blank lines
    text = "\n".join(chunk for chunk in filtered_chunks if chunk)

    return url, text


def detailed_search(tool, query, num_results=10):
    ddg_results = tool.api_wrapper.results(query, num_results)
    pool = Pool(processes=len(ddg_results))
    urls = [result["link"] for result in ddg_results]
    results = pool.map(scrape_url, urls)

    all_results = {}
    for link, scraped in results:
        all_results[link] = scraped
    final_results = []
    result_idx = 0
    for result in ddg_results:
        if all_results[result["link"]] == "Access Denied":
            continue
        result["text"] = all_results[result["link"]]
        result["id"] = result_idx
        final_results.append(result)
        result_idx += 1
    return final_results


class DuckDuckGoSearcher:
    def __init__(self):
        self.searcher = DuckDuckGoSearchResults()

    def search(self, query, num_results=10):
        results = detailed_search(self.searcher, query, num_results=num_results)
        results = {i: x for i, x in enumerate(results)}
        return results
