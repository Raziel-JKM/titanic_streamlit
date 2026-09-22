import io
import os
import re
from html.parser import HTMLParser

html = io.open("titanic_report.html", encoding="utf-8").read()

srcs = re.findall(r'<img[^>]*src="([^"]+)"', html)
missing = [s for s in srcs if not os.path.exists(s)]
print("img 참조:", len(srcs), "| 누락:", missing)

VOID = {"img", "br", "hr", "meta", "link", "input", "source"}
stack, bad = [], []


class P(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if stack and stack[-1] == tag:
            stack.pop()
        else:
            bad.append((tag, list(stack[-3:])))


P().feed(html)
print("닫히지 않은 태그:", stack)
print("짝 안 맞는 종료 태그:", bad)

hrefs = re.findall(r'href="#([^"]+)"', html)
ids = set(re.findall(r'id="([^"]+)"', html))
print("앵커 누락:", [h for h in hrefs if h not in ids])

alt_missing = len(re.findall(r"<img(?![^>]*\balt=)", html))
print("alt 없는 img:", alt_missing)
print("크기(KB):", round(len(html.encode("utf-8")) / 1024, 1))
