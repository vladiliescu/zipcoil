import html
import json
import os
import re
import urllib
from html.parser import HTMLParser

import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI

from smith import tool

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
openai_chatmodel = os.getenv("AZURE_OPENAI_CHAT_MODEL")


GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_search_results_for(query):
    encoded_query = urllib.parse.urlencode({"q": query})
    url = f"https://html.duckduckgo.com/html?q={encoded_query}"

    request = urllib.request.Request(url)
    request.add_header(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    )

    raw_response = urllib.request.urlopen(request).read()
    html = raw_response.decode("utf-8")

    soup = BeautifulSoup(html, "html.parser")
    a_results = soup.select("a.result__a")

    links = []
    for a_result in a_results:
        # print(a_result)
        url = a_result.attrs["href"]
        title = a_result.text
        links.append({"title": title, "url": url})

    return links


class HTMLToTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.ignore_tags = {"script", "style", "meta", "link"}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.text.append(f"\n{'#' * int(tag[1])} ")
        elif tag == "p":
            self.text.append("\n\n")
        elif tag == "br":
            self.text.append("\n")
        elif tag == "li":
            self.text.append("\n- ")

    def handle_endtag(self, tag):
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.text.append("\n")
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.ignore_tags:
            self.text.append(data.strip())

    def get_text(self):
        result = "".join(self.text)
        # Clean up extra whitespace
        result = re.sub(r"\n\s*\n", "\n\n", result)
        return html.unescape(result.strip())


def html_to_simple_text(html_content):
    parser = HTMLToTextParser()
    parser.feed(str(html_content))
    return parser.get_text()


@tool
def load_page_content(url: str) -> str:
    """Returns the content of a particular webpage.

    Args:
        url: Url of the webpage for which to retrieve the content
    """
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        },
    )
    page_content = response.content.decode("utf-8")

    soup = BeautifulSoup(page_content, "html.parser")
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    return html_to_simple_text(soup)


@tool
def get_homepage(blogger: str) -> str:
    """Returns the homepage of a particular blogger.

    Args:
        blogger: Blogger name
    """
    return "https://vladiliescu.net"
    # return get_search_results_for(f"{blogger} homepage")[0]["url"]


tools = [get_homepage, load_page_content]


def call_function(name, args):
    if name == "get_homepage":
        return get_homepage(**args)
    if name == "load_page_content":
        return load_page_content(**args)


query = "What is the homepage of Vlad Iliescu? What is the content of that page?"
messages = [{"role": "user", "content": query}]

total_input_token_count = 0
total_output_token_count = 0

while True:
    completion = client.chat.completions.create(
        model=openai_chatmodel, messages=messages, tools=[tool._tool_schema for tool in tools]
    )

    total_input_token_count += completion.usage.prompt_tokens
    total_output_token_count += completion.usage.completion_tokens

    if completion.choices[0].finish_reason == "stop":
        print(f"{BOLD}Final answer: {completion.choices[0].message.content}{RESET}")
        break
    elif completion.choices[0].finish_reason == "tool_calls":
        messages.append(completion.choices[0].message)
        for tool_call in completion.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            result = call_function(name, args)
            print(f"Called {BOLD}{name}({args}){RESET} and it returned {GRAY}{result[:300]}{RESET}")

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    else:
        raise Exception("We're not supposed to be here")
