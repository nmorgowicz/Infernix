"""Forum HTML to roleplay text: DOM parsers, post filter, thread filters and the excerpt cut.

The behaviour of this module is the corpus definition (``pins.FILTER_VERSION``): the rendering
keeps the whitespace of the markup (no space is inserted or removed at inline tags), drops quotes,
spoilers, signatures, link previews, scripts and images, and the filters below decide which posts and
threads are eligible. The minors-risk screen is a conservative keyword screen, not a content guarantee.
"""

from __future__ import annotations

import re
import unicodedata

import lxml.html

MIN_EXCERPT, MAX_EXCERPT = 2400, 3000
MIN_CUT = 2000
SKIP_POSTS = 2
MIN_THREAD_POSTS = 8
MIN_POST, MAX_POST = 150, 6000

PAGE_MARK = "<!-- [NEW PAGE STARTS BELOW] -->"
SKIP_TAGS = {"script", "style", "noscript", "iframe", "form", "button", "svg", "select", "input", "textarea", "head"}
PARAGRAPH_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "pre", "center", "hr"}
LINE_TAGS = {"div", "li", "tr", "dl", "dt", "dd", "blockquote", "td"}
DROP_CLASSES = (
    "bbCodeBlock--quote",
    "bbCodeBlock--spoiler",
    "bbCodeBlock--unfurl",
    "js-selectToQuoteEnd",
    "bbCodeBlock--screenLimited",
    "message-signature",
    "quoteheader",
    "quotefooter",
)
TITLE_DROP = re.compile(
    r"\b(ooc|out of character|interest check|sign[- ]?ups?|character sheets?|cs|rules?|lore|discussion|"
    r"recruit(?:ing|ment)?|applications?|plot ideas?|ideas?|requests?|search|looking for|wanted|resources?|help|"
    r"questions?|advertis\w+)\b",
    re.I,
)
MINORS = re.compile(
    r"\b(loli\w*|shota\w*|underage|pre-?teen\w*|child ?porn\w*|(?:[0-9]|1[0-7])[- ]?(?:years?|yrs?)[- ]?old)\b", re.I
)
AGE_CONTEXT = re.compile(
    r"\b(school\w*|high ?school|middle school|junior high|classroom|homeroom|teachers?|students?|teen\w*|"
    r"minors?|brat|little girl|little boy)\b",
    re.I,
)
SEXUAL = re.compile(
    r"\b(naked|nude|cleavage|breasts?|nipples?|cocks?|pussy|clit\w*|fuck\w*|cum|cumming|moan\w*|aroused|"
    r"arousal|erect\w*|orgasm\w*|thrust\w*|panties|bra|lingerie|sex|sexual)\b",
    re.I,
)
OOC_POST = re.compile(r"\booc\b|\(\(|\)\)|\[ooc|ooc:", re.I)
DICE = re.compile(r"\b\d+d\d+\b", re.I)
STOPWORDS = frozenset(
    "the of and to a in is it that was for on with as he she his her they i you at be this had not are but from "
    "by or have an were we my me so if what there when them their him one would about out up".split()
)
ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
TYPOGRAPHY = "\u2018\u2019\u201c\u201d\u2013\u2014\u2026"

POST_REASONS = ("length", "ooc", "url", "dice", "non_english", "sheet")

RULES = """\
Parse: threads are split at the page marker into posts (Elliquiy print pages: dt.postheader + dd.postbody; XenForo: article.message--post,
div.bbWrapper). Text is rendered from the DOM without inserting or removing whitespace at inline tags; <br> is a newline, block tags are
line or paragraph breaks; quotes (blockquote on Elliquiy, bbCodeBlock--quote), spoilers, signatures, link previews, scripts, images are
dropped; NFC, NBSP to space, zero-width characters removed, blanks collapsed, 3+ newlines to 2.
Post filter: 150-6000 characters; no OOC marker or double parenthesis; no URL; no dice notation; at most 3 % non-ASCII characters
(typographic quotes, dashes and ellipsis excepted); English by stopword rate (at least 25 words, at least 30 % from a fixed 50-word list); not
a character sheet (six or more short label: value lines). Usernames and timestamps are never rendered.
Thread filter: title not matching the OOC/administrative pattern; at least 8 posts and 8 eligible posts; minors-risk keyword screen over the
whole thread: (a) loli*, shota*, underage, pre-teen*, child porn*, or an age 0-17 written as N years/yrs old; (b) a school/teen/student/minor
term together with any sexual term. A conservative keyword screen, not a content guarantee.
Excerpt: the first run of consecutive eligible posts after the first two eligible posts, with at least 2 posts, 2 distinct authors and
2,400 characters, cut to at most 3,000 characters at the last blank line (at least 2,000 kept) or else the last sentence end.
"""


def render(node, skip_blockquote: bool) -> str:
    """Text of a node: markup whitespace is kept, ``<br>`` is a newline, block tags are line or paragraph breaks."""
    parts: list[str] = []

    def walk(element) -> None:
        tag = element.tag if isinstance(element.tag, str) else None
        if tag is None:
            return
        tag = tag.lower()
        classes = element.get("class", "")
        if tag in SKIP_TAGS or any(c in classes for c in DROP_CLASSES) or (skip_blockquote and tag == "blockquote"):
            return
        if tag == "br":
            parts.append("\n")
            return
        if tag == "img":
            return
        pad = "\n\n" if tag in PARAGRAPH_TAGS else "\n" if tag in LINE_TAGS else ""
        parts.append(pad)
        if element.text:
            parts.append(element.text)
        for child in element:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        parts.append(pad)

    walk(node)
    return "".join(parts)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r", "").replace("\xa0", " ")
    text = ZERO_WIDTH.sub("", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_elliquiy(contents: str) -> list[tuple[str, str]]:
    """(author, text) of every post of an Elliquiy print-page thread."""
    posts = []
    for page in contents.split(PAGE_MARK):
        doc = lxml.html.fromstring(page)
        for header in doc.xpath('//dt[contains(@class,"postheader")]'):
            body = header.getnext()
            if body is None or "postbody" not in body.get("class", ""):
                continue
            author = header.xpath(".//strong")
            author = author[1].text_content().strip() if len(author) > 1 else ""
            posts.append((author, clean(render(body, skip_blockquote=True))))
    return posts


def parse_xenforo(contents: str) -> list[tuple[str, str]]:
    """(author, text) of every post of a XenForo thread."""
    posts = []
    for page in contents.split(PAGE_MARK):
        doc = lxml.html.fromstring(page)
        for article in doc.xpath('//article[contains(@class,"message--post")]'):
            body = article.xpath('.//article[contains(@class,"message-body")]//div[contains(@class,"bbWrapper")]')
            if body:
                posts.append((article.get("data-author", ""), clean(render(body[0], skip_blockquote=False))))
    return posts


PARSERS = {"elliquiy": parse_elliquiy, "xenforo": parse_xenforo}


def post_reason(text: str) -> str | None:
    """None when the post is eligible, else the first failing filter."""
    if not MIN_POST <= len(text) <= MAX_POST:
        return "length"
    if OOC_POST.search(text):
        return "ooc"
    if "http" in text.lower() or "www." in text.lower():
        return "url"
    if DICE.search(text):
        return "dice"
    if sum(ord(c) > 127 and c not in TYPOGRAPHY for c in text) > 0.03 * len(text):
        return "non_english"
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 25 or sum(w in STOPWORDS for w in words) < 0.30 * len(words):
        return "non_english"
    lines = [line for line in text.split("\n") if line]
    if sum(1 for line in lines if len(line) <= 40 and ":" in line) >= 6:
        return "sheet"
    return None


def minors_risk(text: str) -> bool:
    """The conservative keyword screen over a whole thread's text."""
    return bool(MINORS.search(text) or (AGE_CONTEXT.search(text) and SEXUAL.search(text)))


def excerpt_of(posts: list[tuple[str, str]], tally: dict[str, int]):
    """First run of >= 2 consecutive eligible posts (after the first SKIP_POSTS eligible ones) with >= 2 authors.

    Returns ``(text, info)`` or None; ``tally`` counts each post by its reason (``eligible`` when none).
    """
    reasons = [post_reason(text) for _, text in posts]
    for reason in reasons:
        tally[reason or "eligible"] = tally.get(reason or "eligible", 0) + 1
    eligible_seen, run = 0, []
    best = None
    for index, ((_, text), reason) in enumerate(zip(posts, reasons)):
        if reason is None:
            eligible_seen += 1
            if eligible_seen > SKIP_POSTS:
                run.append(index)
        else:
            run = []
        if run:
            joined = "\n\n".join(posts[i][1] for i in run)
            if len(joined) >= MIN_EXCERPT and len({posts[i][0] for i in run}) >= 2 and len(run) >= 2:
                best = list(run)
                break
    if best is None:
        return None
    text = "\n\n".join(posts[i][1] for i in best)
    if len(text) > MAX_EXCERPT:
        cut = text.rfind("\n\n", 0, MAX_EXCERPT)
        if cut < MIN_CUT:
            ends = [m.end() for m in re.finditer(r"[.!?][\'\"\u201d\u2019)]*(?=\s)", text[:MAX_EXCERPT])]
            cut = ends[-1] if ends else -1
        if cut < MIN_CUT:
            return None
        text = text[:cut].rstrip()
    authors = len({posts[i][0] for i in best})
    used, covered = 0, 0
    for i in best:
        covered += len(posts[i][1]) + 2
        used += 1
        if covered >= len(text):
            break
    return text, {"first_post_index": best[0], "posts": used, "authors": authors}
