"""Retrieve and search public SEC filing text using standard-library HTTP."""
import re
from html.parser import HTMLParser
from urllib.parse import quote
from urllib.request import Request, urlopen

SEC_ARCHIVES = 'https://www.sec.gov/Archives/edgar/data'
USER_AGENT = 'SECResearchAgent/1.0 research contact research@example.org'

class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag in ('p', 'div', 'tr', 'li', 'h1', 'h2', 'h3', 'br'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1
        if tag in ('p', 'div', 'tr', 'li', 'h1', 'h2', 'h3'):
            self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

def filing_url(cik, accession, primary_document):
    cik_digits = str(cik).lstrip('0') or '0'
    if not cik_digits.isdigit() or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
        raise ValueError('Invalid CIK or accession')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', primary_document) or primary_document in ('.', '..'):
        raise ValueError('Invalid primary document filename')
    return f'{SEC_ARCHIVES}/{cik_digits}/{accession.replace("-", "")}/{quote(primary_document)}'

def get_filing_text(cik, accession, primary_document, *, max_bytes=12_000_000):
    """Fetch one SEC primary filing document; return text and canonical source URL."""
    url = filing_url(cik, accession, primary_document)
    req = Request(url, headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
    with urlopen(req, timeout=45) as response:
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError('Filing exceeds maximum retrieval size')
        encoding = response.headers.get_content_charset() or 'utf-8'
    content = raw.decode(encoding, errors='replace')
    if primary_document.lower().endswith(('.htm', '.html')):
        parser = _Text()
        parser.feed(content)
        content = ''.join(parser.parts)
    content = re.sub(r'[ \t]+', ' ', content)
    content = re.sub(r'\n\s*\n+', '\n\n', content)
    return {'url': url, 'accession': accession, 'text': content}

def search_filing(text, terms, *, window=450, max_hits=12):
    """Return bounded, verbatim excerpts with offsets and matched terms."""
    if isinstance(text, dict):
        text = text['text']
    if isinstance(terms, str):
        terms = [terms]
    terms = [t for t in terms if isinstance(t, str) and t.strip()]
    if not terms:
        raise ValueError('Supply at least one search term')
    hits = []
    for match in re.finditer('|'.join(re.escape(t) for t in terms), text, re.IGNORECASE):
        a, b = max(0, match.start()-window), min(len(text), match.end()+window)
        hits.append({'term': match.group(), 'offset': match.start(), 'excerpt': text[a:b]})
        if len(hits) >= max_hits:
            break
    return hits

DEFAULT_RESEARCH_TERMS = ('Project Jupiter', 'force majeure', 'New Mexico',
                          'data center', 'lease commitments', 'liquidity')


def research_filings(cik, forms=('10-K', '10-Q', '8-K'), terms=DEFAULT_RESEARCH_TERMS,
                     max_filings=30, *, filings=None, fetch=None, max_hits_per_filing=8):
    """Search actual primary SEC documents, not filing metadata.

    A retrieval failure is recorded rather than interpreted as a negative search.
    `filings` and `fetch` permit deterministic offline tests and caching.
    """
    if not isinstance(max_filings, int) or not 1 <= max_filings <= 100:
        raise ValueError('max_filings must be between 1 and 100')
    if not forms or not terms:
        raise ValueError('forms and terms must be nonempty')
    if filings is None:
        from sec_data import get_recent_filings
        filings = get_recent_filings(cik)
    fetch = fetch or get_filing_text
    allowed = set(forms)
    candidates = [f for f in filings if f.get('form') in allowed
                  and f.get('accessionNumber') and f.get('primaryDocument')]
    candidates.sort(key=lambda f: f.get('filingDate', ''), reverse=True)
    sources, findings, failures = [], [], []
    for filing in candidates[:max_filings]:
        accn = filing['accessionNumber']
        try:
            document = fetch(cik, accn, filing['primaryDocument'])
            if not isinstance(document, dict) or not document.get('text') or not document.get('url'):
                raise ValueError('Empty or invalid filing document')
            hits = search_filing(document['text'], terms, max_hits=max_hits_per_filing)
        except (OSError, ValueError, KeyError, TimeoutError) as exc:
            failures.append({'accession': accn, 'error': f'{type(exc).__name__}: {exc}'})
            continue
        sources.append({'accession': accn, 'url': document['url'],
                        'form': filing['form'], 'filing_date': filing['filingDate']})
        for hit in hits:
            findings.append({'statement': f"Filing mentions {hit['term']}",
                             'accession': accn, 'excerpt': hit['excerpt'],
                             'term': hit['term']})
    return {'sources': sources, 'findings': findings, 'failures': failures,
            'searched': len(sources), 'attempted': min(len(candidates), max_filings),
            'limitations': ([f'{len(failures)} filing retrieval(s) failed; absence of matches is not proof of absence.']
                            if failures else [])}


# Evidence preparation and synthesis checks (formerly evidence_synthesis.py)
"""Typed, compact evidence preparation for model interpretation."""

def prepare_evidence(evidence, trends=None, *, max_excerpts=12):
    """Return a predictable schema; raw search hits are NOT analytical findings."""
    if not isinstance(evidence, dict):
        raise TypeError('research_filings must return a dictionary')
    if trends is not None and (not isinstance(trends, dict) or not isinstance(trends.get('rows'), list)):
        raise TypeError("financial_trends must return {'rows': [...], 'changes': [...]}")
    snippets, seen = [], set()
    for item in evidence.get('findings', []):
        if not isinstance(item, dict):
            continue
        excerpt = item.get('excerpt', '')
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        key = (item.get('accession'), re.sub(r'\s+', ' ', excerpt).strip()[:160].lower())
        if key in seen:
            continue
        seen.add(key)
        snippets.append({'accession':item.get('accession'), 'term':item.get('term',''),
                         'excerpt':excerpt[:1300]})
        if len(snippets) >= max_excerpts:
            break
    return {'snippets':snippets, 'sources':evidence.get('sources', []),
            'retrieval_failures':evidence.get('failures', []),
            'trends':trends if trends is not None else {'rows':[], 'changes':[]},
            'instructions':('Interpret selected excerpts; do not copy raw tables or report keyword matches. '
                            'Quantitative claims must identify value, unit, period and accession. '
                            'An SEC mention of financial constraints is not direct proof of event causation.')}

def validate_synthesis(result):
    """Cheap structural checks, not semantic verification of financial interpretations."""
    errors = []
    for field in ('findings', 'financial_indicators', 'direct_evidence'):
        for i, item in enumerate(result.get(field, [])):
            if not isinstance(item, dict):
                continue
            statement = item.get('statement','')
            if not isinstance(statement,str):
                continue
            if len(statement) > 650 or '\n' in statement:
                errors.append(f'{field}[{i}] must be one concise prose statement, not a raw filing excerpt.')
            if re.search(r'\b(?:filing|report|document)\s+(?:mentions|contains|references)\b',statement,re.IGNORECASE):
                errors.append(f'{field}[{i}] is a keyword-match description, not a material finding.')
            if re.search(r'\b(?:caused|precipitated)\b',statement,re.IGNORECASE) and result.get('conclusion') == 'insufficient_evidence':
                errors.append(f'{field}[{i}] claims causation inconsistent with the conclusion.')
    return errors
