# =============================================================================
# HEPCoverageKG: generic HTML harvester
# Fetches and parses a single fixed HTML page, source-aware where needed.
# First source verified against: arXiv's HTML5 rendering (LaTeXML/ar5iv
# output), e.g. https://arxiv.org/html/<arxiv_id> - see 2307.01094 (ATLAS
# SUSY search) as the reference page this parser was built against.
#
# requests.Session + rotating User-Agent + retry/backoff conventions ported
# from DeepCollector's uci_harvester.py.
# =============================================================================
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from hepcoveragekg.harvesting.base_harvester import BaseHarvester

try:
    import lxml  # noqa: F401

    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

# Top-level content sections to index; subsections nest inside these, so
# taking the parent's full text already includes them. Bibliography is
# excluded - citation lists aren't useful for field extraction.
SECTION_CLASSES = {"ltx_section"}
HEADING_TAGS = ["h1", "h2", "h3", "h4"]

# LaTeXML renders an unresolved custom macro (e.g. ATLAS/CMS's own
# \AtlasTitle, \AtlasAbstract commands) as literal leftover text like
# "\AtlasAbstract" inline with real content, instead of a clean tag boundary.
UNDEFINED_MACRO_SPLIT = re.compile(r"\\[A-Za-z]+(?:Title|Abstract)\b")


@dataclass
class HarvestedSection:
    heading: str
    text: str


@dataclass
class HarvestedPaper:
    source_url: str
    title: str
    abstract: str
    sections: list[HarvestedSection] = field(default_factory=list)


class HTMLHarvester(BaseHarvester):
    """Fetches a single fixed HTML page and parses it into a HarvestedPaper."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 5.0
    REQUEST_TIMEOUT = 30

    def __init__(self, config: Any):
        super().__init__(config)
        self.session = requests.Session()

    def fetch(self, url: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                headers = {"User-Agent": random.choice(USER_AGENTS), "Accept": "text/html"}
                response = self.session.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.text
            except requests.RequestException as error:
                last_error = error
                if self.verbosity >= 1:
                    print(f"[HTMLHarvester] fetch failed (attempt {attempt}/{self.MAX_RETRIES}) for {url}: {error}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"Failed to fetch {url} after {self.MAX_RETRIES} attempts") from last_error

    def parse(self, html: str, source_url: str) -> HarvestedPaper:
        soup = BeautifulSoup(html, BS_PARSER)
        self._strip_math_annotations(soup)
        title, abstract = self._extract_title_and_abstract(soup)
        sections = self._extract_sections(soup)
        if self.verbosity >= 2:
            print(f"[HTMLHarvester] parsed '{title[:60]}...' - {len(sections)} sections")
        return HarvestedPaper(source_url=source_url, title=title, abstract=abstract, sections=sections)

    def harvest(self, url: str) -> HarvestedPaper:
        html = self.fetch(url)
        return self.parse(html, url)

    def _extract_title_and_abstract(self, soup: BeautifulSoup) -> tuple[str, str]:
        # Preferred path: standard LaTeXML tagging.
        title_tag = soup.find(class_="ltx_title_document") or soup.find("h1")
        abstract_tag = soup.find(class_="ltx_abstract")
        if title_tag and abstract_tag:
            return self._clean_text(title_tag.get_text(" ")), self._clean_text(abstract_tag.get_text(" "))

        # Fallback: custom-macro papers (e.g. ATLAS/CMS \AtlasTitle/\AtlasAbstract)
        # collapse title+abstract into the article's first paragraph, with the
        # undefined macro name left inline as a separator between the two.
        article = soup.find("article", class_="ltx_document")
        if article is None:
            return "", ""
        first_para = article.find("div", class_="ltx_para")
        if first_para is None:
            return "", ""

        text = self._clean_text(first_para.get_text(" "))
        parts = [p.strip() for p in UNDEFINED_MACRO_SPLIT.split(text) if p.strip()]
        title = parts[0] if parts else ""
        abstract = parts[1] if len(parts) > 1 else ""
        return title, abstract

    def _extract_sections(self, soup: BeautifulSoup) -> list[HarvestedSection]:
        sections = []
        for section_tag in soup.find_all("section"):
            classes = set(section_tag.get("class", []))
            if not classes & SECTION_CLASSES:
                continue
            if section_tag.find_parent("section") is not None:
                continue  # nested subsection - already covered by its parent's text
            heading_tag = section_tag.find(HEADING_TAGS)
            heading = self._clean_text(heading_tag.get_text(" ")) if heading_tag else ""
            body_text = self._clean_text(section_tag.get_text(" "))
            if heading and body_text.startswith(heading):
                body_text = body_text[len(heading):].strip()
            sections.append(HarvestedSection(heading=heading, text=body_text))
        return sections

    @staticmethod
    def _strip_math_annotations(soup: BeautifulSoup) -> None:
        # Each <math> carries the visible MathML rendering plus redundant
        # fallback encodings (TeX source, arXiv's "llamapun" plain-text
        # annotation) - get_text() would otherwise concatenate all of them,
        # e.g. "139 fb^-1" -> "139 fb − 1 1 {}^{-1} start_FLOATSUPERSCRIPT...".
        for tag in soup.find_all(["annotation", "annotation-xml"]):
            tag.decompose()

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
