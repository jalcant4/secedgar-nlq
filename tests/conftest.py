"""Allow isolated unit tests without the user's local edgar.py transport."""
import sys
import types

# Add src/ to path so project modules are importable.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))

try:
    import edgar
except ImportError:
    edgar = types.ModuleType('edgar')
    edgar.get_company_facts = lambda cik: {}
    edgar.get_filings = lambda cik: {}
    edgar.get_cik = lambda company: ''
    sys.modules['edgar'] = edgar
try:
    import llm
except ImportError:
    llm = types.ModuleType('llm')
    llm.llm_query = lambda prompt: ''
    sys.modules['llm'] = llm
try:
    import repl
except ImportError:
    repl = types.ModuleType('repl')
    repl.REPL = object
    sys.modules['repl'] = repl
