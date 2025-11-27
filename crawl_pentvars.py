import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser
import uuid

import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
START_URL = "https://pentvars.edu.gh/"
COLLECTION_NAME = "pentvars_site"

USER_AGENT = "Mozilla/5.0 (compatible; PentvarsCrawler/1.0; +https://pentvars.edu.gh)"
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.5


def get_chroma_collection():
    """Get or create the Chroma collection used for Pentvars website content."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)
    return collection


def get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer model used for embeddings.

    We reuse the same model as in app.py: all-MiniLM-L6-v2.
    """
    # Lazy-load to avoid heavy import at module import time
    # (simple singleton pattern)
    global _EMBEDDING_MODEL  # type: ignore[assignment]
    try:
        model = _EMBEDDING_MODEL  # type: ignore[name-defined]
    except NameError:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _EMBEDDING_MODEL = model  # type: ignore[assignment]
    return model


def init_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def init_robots(start_url: str) -> Optional[RobotFileParser]:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        resp = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            return rp
        return None
    except Exception:
        return None


def is_allowed(url: str, rp: Optional[RobotFileParser]) -> bool:
    if rp is None:
        return True
    try:
        return rp.can_fetch("*", url)
    except Exception:
        return True


def normalize_url(url: str, base_url: str) -> str:
    # Convert relative -> absolute, drop fragments
    joined = urljoin(base_url, url)
    joined, _ = urldefrag(joined)
    return joined


def same_domain(url: str, base_url: str) -> bool:
    a = urlparse(url).netloc.lower()
    b = urlparse(base_url).netloc.lower()
    # Treat www.pentvars.edu.gh and pentvars.edu.gh as the same
    if a.startswith("www."):
        a = a[4:]
    if b.startswith("www."):
        b = b[4:]
    return a == b


def fetch_html(session: requests.Session, url: str) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            print(f"[error] Attempt {attempt + 1} to fetch {url} failed: {e}")
            time.sleep(2 * (attempt + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to fetch {url} for unknown reasons")


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def clean_soup(soup: BeautifulSoup) -> None:
    # Remove obvious non-content elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()


def extract_links(soup: BeautifulSoup, base_url: str) -> Set[str]:
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        norm = normalize_url(href, base_url)
        if norm.startswith("http"):
            links.add(norm)
    return links


def extract_sections(soup: BeautifulSoup, url: str) -> List[Dict[str, Optional[str]]]:
    """Extract logical content sections based on headers (h1–h3).

    Each section is anchored by a header and includes text from its following
    siblings up to (but not including) the next header of the same or higher level.
    """
    sections: List[Dict[str, Optional[str]]] = []

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    body = soup.body or soup
    headers = body.find_all(["h1", "h2", "h3"])

    # Fallback: treat the whole page as one section
    if not headers:
        full_text = body.get_text(" ", strip=True)
        if full_text:
            sections.append(
                {
                    "url": url,
                    "page_title": page_title,
                    "header_level": None,
                    "header_text": page_title or url,
                    "text": full_text,
                }
            )
        return sections

    for header in headers:
        header_text = header.get_text(" ", strip=True)
        header_level = header.name

        texts: List[str] = []
        for sibling in header.next_siblings:
            name = getattr(sibling, "name", None)
            if name in {"h1", "h2", "h3"}:
                break
            if name is None:
                continue
            texts.append(sibling.get_text(" ", strip=True))

        section_text = " ".join(t for t in texts if t).strip()
        if section_text:
            sections.append(
                {
                    "url": url,
                    "page_title": page_title,
                    "header_level": header_level,
                    "header_text": header_text,
                    "text": section_text,
                }
            )

    return sections


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Split long text into overlapping character-based chunks."""
    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def index_page(
    collection,
    model: SentenceTransformer,
    url: str,
    soup: BeautifulSoup,
    max_chars: int = 1000,
    overlap: int = 200,
) -> int:
    """Extract sections from a page, chunk them, embed, and add to Chroma.

    Returns the number of chunks stored for this page.
    """
    sections = extract_sections(soup, url)
    if not sections:
        return 0

    documents: List[str] = []
    metadatas: List[Dict[str, Optional[str]]] = []
    ids: List[str] = []

    for section_index, section in enumerate(sections):
        section_text = section.get("text") or ""
        if not section_text.strip():
            continue

        chunks = chunk_text(section_text, max_chars=max_chars, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "url": section["url"],
                    "page_title": section["page_title"],
                    "header": section["header_text"],
                    "header_level": section["header_level"],
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                }
            )
            ids.append(str(uuid.uuid4()))

    if not documents:
        return 0

    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(documents)


def crawl_and_index(
    start_url: str = START_URL,
    max_pages: int = 100,
    max_chars: int = 1000,
    overlap: int = 200,
) -> None:
    """BFS crawl within the pentvars.edu.gh domain and index content into Chroma."""
    session = init_session()
    rp = init_robots(start_url)
    collection = get_chroma_collection()
    model = get_embedding_model()

    queue: deque[str] = deque([start_url])
    seen: Set[str] = set()

    pages_processed = 0
    total_chunks = 0

    print(f"Starting crawl from {start_url} (max_pages={max_pages})")

    while queue and pages_processed < max_pages:
        url = queue.popleft()

        if url in seen:
            continue
        seen.add(url)

        if not same_domain(url, start_url):
            continue

        if not is_allowed(url, rp):
            print(f"[robots] Skipping disallowed URL: {url}")
            continue

        try:
            html = fetch_html(session, url)
        except Exception as e:
            print(f"[error] Failed to fetch {url}: {e}")
            continue

        soup = make_soup(html)
        clean_soup(soup)

        chunks_added = index_page(
            collection,
            model,
            url,
            soup,
            max_chars=max_chars,
            overlap=overlap,
        )

        pages_processed += 1
        total_chunks += chunks_added

        print(f"[{pages_processed}] {url} -> {chunks_added} chunks (total={total_chunks})")

        # Enqueue new links for BFS
        for link in extract_links(soup, url):
            if link not in seen:
                queue.append(link)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"Done. Pages processed: {pages_processed}, chunks stored: {total_chunks}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crawl pentvars.edu.gh and index content into a Chroma collection.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        default=START_URL,
        help="Starting URL for crawling (default: https://pentvars.edu.gh/)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of pages to crawl (default: 100)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive chunks (default: 200)",
    )

    args = parser.parse_args()

    crawl_and_index(
        start_url=args.start_url,
        max_pages=args.max_pages,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
