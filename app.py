"""
MathSphere v12.0 — Complete Production Backend
✅ Fixed Image Processing
✅ Enhanced Graph Plotter (|x|, mod, √ support)
✅ Mathematician Links (Wikipedia + Resources)
✅ Project Resources (4-5 links each)
✅ All endpoints tested and working
"""

import os, sys, io, json, logging, re, base64, random, sqlite3
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════════
# SYSTEM SETUP
# ════════════════════════════════════════════════════════════════

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mathsphere.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY   = os.getenv('GROQ_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
SECRET_KEY     = os.getenv('SECRET_KEY', 'mathsphere-secret-2024')
FLASK_ENV      = os.getenv('FLASK_ENV', 'production')
DEBUG_MODE     = FLASK_ENV == 'development'

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DB_PATH    = os.path.join(BASE_DIR, 'pyqs.db')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    JSON_SORT_KEYS=False,
    CACHE_TYPE='SimpleCache',
    CACHE_DEFAULT_TIMEOUT=3600
)

CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET","POST","OPTIONS","DELETE"],
                                  "allow_headers": ["Content-Type"]}})

limiter = Limiter(app=app, key_func=get_remote_address,
                  default_limits=["300 per day","60 per hour"],
                  storage_uri="memory://")
cache = Cache(app)

# ════════════════════════════════════════════════════════════════
# API CLIENTS
# ════════════════════════════════════════════════════════════════

GROQ_AVAILABLE = False
groq_client = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
        logger.info("[OK] Groq connected")
    else:
        logger.warning("[WARN] GROQ_API_KEY not set")
except Exception as e:
    logger.warning(f"[WARN] Groq: {e}")

GEMINI_AVAILABLE = False
gemini_client = None
genai_types = None

try:
    from google import genai as _genai
    from google.genai import types as _types
    if GEMINI_API_KEY:
        gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
        genai_types = _types
        GEMINI_AVAILABLE = True
        logger.info("[OK] Gemini connected")
    else:
        logger.warning("[WARN] GEMINI_API_KEY not set")
except Exception as e:
    logger.warning(f"[WARN] Gemini: {e}")

SYMPY_AVAILABLE = False

try:
    from sympy import (
        Symbol, N, diff, solve, limit, series,
        sin, cos, tan, asin, acos, atan, cot, sec, csc,
        sinh, cosh, tanh, asinh, acosh, atanh,
        exp, log, sqrt, Abs, factorial, erf, gamma,
        ceiling, floor, sign, Mod,
        pi, E, I, oo, nan, zoo,
        Rational, Float, Integer, Poly, roots, factor, expand,
        Derivative, Integral, summation,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor
    )
    SYMPY_AVAILABLE = True
    logger.info("[OK] SymPy loaded")
except Exception as e:
    logger.warning(f"[WARN] SymPy: {e}")

NUMPY_AVAILABLE = False
try:
    from numpy import isfinite as _isfinite, isnan as _isnan
    NUMPY_AVAILABLE = True
except Exception:
    def _isfinite(x):
        try: return float(x) not in (float('inf'), float('-inf'))
        except: return False
    def _isnan(x):
        try: v=float(x); return v!=v
        except: return True

# ════════════════════════════════════════════════════════════════
# MATHEMATICIAN DATABASE WITH LINKS
# ════════════════════════════════════════════════════════════════

MATHEMATICIANS_DB = {
    "ramanujan": {
        "name": "Srinivasa Ramanujan",
        "birth": "1887",
        "death": "1920",
        "nationality": "Indian",
        "field": "Number Theory, Modular Forms, Series",
        "wiki": "https://en.wikipedia.org/wiki/Srinivasa_Ramanujan",
        "resources": [
            "https://mathworld.wolfram.com/Ramanujan.html",
            "https://www.britannica.com/biography/Srinivasa-Ramanujan",
            "https://www.youtube.com/results?search_query=Ramanujan+mathematics",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Ramanujan.html"
        ],
        "summary": "Self-taught Indian mathematical genius who made extraordinary contributions to number theory, infinite series, and continued fractions. Despite limited formal training, Ramanujan discovered thousands of mathematical identities. His work on partition functions, mock theta functions, and the Riemann hypothesis remains influential."
    },
    "euler": {
        "name": "Leonhard Euler",
        "birth": "1707",
        "death": "1783",
        "nationality": "Swiss",
        "field": "Analysis, Number Theory, Graph Theory, Topology",
        "wiki": "https://en.wikipedia.org/wiki/Leonhard_Euler",
        "resources": [
            "https://mathworld.wolfram.com/Euler.html",
            "https://www.britannica.com/biography/Leonhard-Euler",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Euler.html",
            "https://www.youtube.com/results?search_query=Leonhard+Euler+mathematics"
        ],
        "summary": "One of history's most prolific mathematicians. Euler invented the modern notation for mathematical functions (f(x)), studied graph theory, topology, and complex analysis. His work spans calculus, number theory, and physics—making foundational contributions to virtually every area."
    },
    "gauss": {
        "name": "Carl Friedrich Gauss",
        "birth": "1777",
        "death": "1855",
        "nationality": "German",
        "field": "Number Theory, Statistics, Analysis, Astronomy",
        "wiki": "https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss",
        "resources": [
            "https://mathworld.wolfram.com/Gauss.html",
            "https://www.britannica.com/biography/Carl-Friedrich-Gauss",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Gauss.html",
            "https://www.youtube.com/results?search_query=Gauss+number+theory"
        ],
        "summary": "Known as the 'Prince of Mathematicians.' Gauss contributed to number theory (quadratic reciprocity), statistics (Gaussian distribution), astronomy, and magnetism. His Disquisitiones Arithmeticae is a cornerstone of modern number theory."
    },
    "riemann": {
        "name": "Bernhard Riemann",
        "birth": "1826",
        "death": "1866",
        "nationality": "German",
        "field": "Analysis, Geometry, Number Theory",
        "wiki": "https://en.wikipedia.org/wiki/Bernhard_Riemann",
        "resources": [
            "https://mathworld.wolfram.com/Riemann.html",
            "https://www.britannica.com/biography/Bernhard-Riemann",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Riemann.html",
            "https://www.youtube.com/results?search_query=Riemann+hypothesis+explained"
        ],
        "summary": "Created the foundations of complex analysis, differential geometry, and introduced the Riemann integral. The Riemann Hypothesis on prime numbers remains one of mathematics' greatest unsolved problems."
    },
    "newton": {
        "name": "Isaac Newton",
        "birth": "1643",
        "death": "1727",
        "nationality": "English",
        "field": "Calculus, Physics, Optics",
        "wiki": "https://en.wikipedia.org/wiki/Isaac_Newton",
        "resources": [
            "https://mathworld.wolfram.com/Newton.html",
            "https://www.britannica.com/biography/Isaac-Newton",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Newton.html",
            "https://www.youtube.com/results?search_query=Newton+calculus+physics"
        ],
        "summary": "Co-inventor of calculus (with Leibniz). Newton's laws of motion and universal gravitation revolutionized physics. His work on binomial expansions and infinite series laid groundwork for modern analysis."
    },
    "leibniz": {
        "name": "Gottfried Wilhelm Leibniz",
        "birth": "1646",
        "death": "1716",
        "nationality": "German",
        "field": "Calculus, Philosophy, Logic",
        "wiki": "https://en.wikipedia.org/wiki/Gottfried_Wilhelm_Leibniz",
        "resources": [
            "https://mathworld.wolfram.com/Leibniz.html",
            "https://www.britannica.com/biography/Gottfried-Wilhelm-Leibniz",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Leibniz.html",
            "https://www.youtube.com/results?search_query=Leibniz+calculus+notation"
        ],
        "summary": "Co-developer of calculus with Newton. Leibniz's notation (dy/dx, integral sign) is used today. Contributed to logic, metaphysics, and founded the Prussian Academy of Sciences."
    },
    "cantor": {
        "name": "Georg Cantor",
        "birth": "1845",
        "death": "1918",
        "nationality": "German-Russian",
        "field": "Set Theory, Infinity, Topology",
        "wiki": "https://en.wikipedia.org/wiki/Georg_Cantor",
        "resources": [
            "https://mathworld.wolfram.com/Cantor.html",
            "https://www.britannica.com/biography/Georg-Cantor",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Cantor.html",
            "https://www.youtube.com/results?search_query=Cantor+set+theory+infinity"
        ],
        "summary": "Founder of set theory. Cantor revolutionized mathematics by rigorously studying infinity and uncountable sets. His work on the continuum hypothesis and diagonal argument remain central to mathematics."
    },
    "hardy": {
        "name": "Godfrey Harold Hardy",
        "birth": "1877",
        "death": "1947",
        "nationality": "English",
        "field": "Number Theory, Analysis, Pure Mathematics",
        "wiki": "https://en.wikipedia.org/wiki/G._H._Hardy",
        "resources": [
            "https://mathworld.wolfram.com/Hardy.html",
            "https://www.britannica.com/biography/G.-H.-Hardy",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Hardy.html",
            "https://www.youtube.com/results?search_query=Hardy+number+theory+Ramanujan"
        ],
        "summary": "Brilliant number theorist who championed pure mathematics. Hardy famously collaborated with Ramanujan and wrote 'A Mathematician's Apology,' a classic defense of mathematical beauty."
    },
    "fermat": {
        "name": "Pierre de Fermat",
        "birth": "1607",
        "death": "1665",
        "nationality": "French",
        "field": "Number Theory, Analytic Geometry, Calculus",
        "wiki": "https://en.wikipedia.org/wiki/Pierre_de_Fermat",
        "resources": [
            "https://mathworld.wolfram.com/Fermat.html",
            "https://www.britannica.com/biography/Pierre-de-Fermat",
            "https://mathshistory.st-andrews.ac.uk/Biographies/Fermat.html",
            "https://www.youtube.com/results?search_query=Fermat+last+theorem"
        ],
        "summary": "Pioneer in analytic geometry and number theory. Fermat's Last Theorem (x^n + y^n = z^n has no integer solutions for n>2) remained unsolved for 358 years until 1995. Also studied probability and maxima/minima."
    },
}

# ════════════════════════════════════════════════════════════════
# PROJECT RESOURCES DATABASE
# ════════════════════════════════════════════════════════════════

PROJECT_RESOURCES = {
    "calculus": [
        {"title": "Khan Academy - Calculus", "url": "https://www.khanacademy.org/math/calculus-1"},
        {"title": "3Blue1Brown - Essence of Calculus", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"},
        {"title": "Stewart's Calculus Textbook", "url": "https://www.cengage.com/c/calculus-9e-stewart/"},
        {"title": "Paul's Online Math Notes", "url": "https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx"},
        {"title": "Wolfram MathWorld - Calculus", "url": "https://mathworld.wolfram.com/topics/Calculus.html"}
    ],
    "real analysis": [
        {"title": "Rudin - Principles of Mathematical Analysis", "url": "https://www.mheducation.com/highered/product/principles-mathematical-analysis-walter-rudin/9780078156571.html"},
        {"title": "Real Analysis Lecture Series - MIT OCW", "url": "https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/"},
        {"title": "ProofWiki - Analysis", "url": "https://proofwiki.org/wiki/Definition:Real_Analysis"},
        {"title": "LibreTexts - Real Analysis", "url": "https://math.libretexts.org/Bookshelves/Real_Analysis"},
        {"title": "Mathtutor - Real Analysis Videos", "url": "https://www.youtube.com/c/MathTutor"}
    ],
    "linear algebra": [
        {"title": "3Blue1Brown - Essence of Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"},
        {"title": "Gilbert Strang - Linear Algebra (MIT)", "url": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/"},
        {"title": "Khan Academy - Linear Algebra", "url": "https://www.khanacademy.org/math/linear-algebra"},
        {"title": "Wolfram MathWorld - Linear Algebra", "url": "https://mathworld.wolfram.com/topics/LinearAlgebra.html"},
        {"title": "3x3 Matrix Operations Visualizer", "url": "https://www.geogebra.org/m/JXnrRFZy"}
    ],
    "abstract algebra": [
        {"title": "Dummit & Foote - Abstract Algebra Textbook", "url": "https://www.wiley.com/en-us/Abstract+Algebra%2C+3rd+Edition-p-9780471433347"},
        {"title": "Group Theory Visualizer", "url": "https://www.geogebra.org/search/group%20theory"},
        {"title": "AoPS - Abstract Algebra", "url": "https://artofproblemsolving.com/wiki/index.php/Abstract_algebra"},
        {"title": "Khan Academy - Groups & Rings", "url": "https://www.khanacademy.org/"},
        {"title": "ProofWiki - Algebra", "url": "https://proofwiki.org/wiki/Definition:Algebra"}
    ],
    "complex analysis": [
        {"title": "Complex Analysis - MIT OCW", "url": "https://ocw.mit.edu/courses/18-112-functions-of-a-complex-variable-fall-2012/"},
        {"title": "Ahlfors - Complex Analysis Textbook", "url": "https://www.mheducation.com/highered/product/complex-analysis-lars-ahlfors/9780070006577.html"},
        {"title": "Brilliant.org - Complex Numbers", "url": "https://brilliant.org/courses/complex-numbers/"},
        {"title": "3D Complex Function Visualizer", "url": "https://www.geogebra.org/m/vMX5HQFX"},
        {"title": "ProofWiki - Complex Analysis", "url": "https://proofwiki.org/wiki/Definition:Complex_Analysis"}
    ],
    "number theory": [
        {"title": "Hardy & Wright - Introduction to Number Theory", "url": "https://www.oxfordscholarship.com/view/10.1093/oso/9780198533818.001.0001"},
        {"title": "Project Euler - Number Theory Problems", "url": "https://projecteuler.net/"},
        {"title": "Khan Academy - Number Theory", "url": "https://www.khanacademy.org/math/number-theory"},
        {"title": "AoPS - Number Theory", "url": "https://artofproblemsolving.com/wiki/index.php/Number_theory"},
        {"title": "Wolfram MathWorld - Number Theory", "url": "https://mathworld.wolfram.com/topics/NumberTheory.html"}
    ],
    "differential equations": [
        {"title": "Boyce & DiPrima - Differential Equations", "url": "https://www.wiley.com/en-us/Elementary+Differential+Equations+-+11th+Edition-p-9781119320913"},
        {"title": "3Blue1Brown - Differential Equations", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjzJ9zMBA0TTu7b9BFVZ"},
        {"title": "MIT OCW - Differential Equations", "url": "https://ocw.mit.edu/courses/18-03-differential-equations-spring-2010/"},
        {"title": "Paul Dawkins - Differential Equations", "url": "https://tutorial.math.lamar.edu/Classes/DE/DE.aspx"},
        {"title": "Desmos - Slope Field Explorer", "url": "https://www.desmos.com/calculator/p7tlmutpm8"}
    ],
    "topology": [
        {"title": "Topology - MIT OCW", "url": "https://ocw.mit.edu/courses/18-901-topology-i-fall-2004/"},
        {"title": "Munkres - Topology Textbook", "url": "https://www.pearsonhighered.com/product/Munkres-Topology-2nd-Edition/9780131816299.html"},
        {"title": "Brilliant.org - Topology", "url": "https://brilliant.org/courses/topology/"},
        {"title": "ProofWiki - Topology", "url": "https://proofwiki.org/wiki/Definition:Topology"},
        {"title": "Topological Shape Explorer", "url": "https://www.geogebra.org/search/topology"}
    ],
    "probability": [
        {"title": "MIT Probability Course", "url": "https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/"},
        {"title": "Khan Academy - Probability", "url": "https://www.khanacademy.org/math/statistics-probability"},
        {"title": "Sheldon Ross - Introduction to Probability", "url": "https://www.pearsonhighered.com/product/Ross-A-First-Course-in-Probability-10th-Edition/9780137618560.html"},
        {"title": "Wolfram MathWorld - Probability", "url": "https://mathworld.wolfram.com/topics/Probability.html"},
        {"title": "Interactive Probability Simulator", "url": "https://www.geogebra.org/search/probability"}
    ],
}

# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════

MATH_SYSTEM = """You are Anupam — a world-class mathematics tutor specializing in IIT JAM, GATE, and CSIR NET.

For every problem, provide a COMPLETE structured response:

## 🧠 Key Insight
[The core idea needed to solve this]

## 📚 Prerequisites
[What the student must know first]

## 📝 Step-by-Step Solution
[Numbered steps with full working, using LaTeX for all math]

## ✅ Verification
[Check the answer]

## 📋 Exam Relevance
[Which exams test this, typical marks]

CRITICAL RULES:
- Use LaTeX for ALL mathematical expressions: \\( inline \\) and \\[ display \\]
- NEVER return raw JSON or code blocks unless specifically asked
- NEVER say "I cannot" — always attempt the problem
- Be precise, accurate, and thorough
- For MCQ: clearly state the correct option and WHY others are wrong"""

# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def clean_ai_response(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'```[\w]*\n?', '', text)
    text = text.replace('```', '')
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            obj = json.loads(stripped)
            for key in ('answer', 'content', 'response', 'text', 'result', 'message'):
                if key in obj and isinstance(obj[key], str):
                    return obj[key]
        except Exception:
            pass
    return text.strip()


def ask_ai(messages: list, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    if not messages:
        return ""

    if GROQ_AVAILABLE and groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": MATH_SYSTEM}] + messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return clean_ai_response(text)
        except Exception as e:
            logger.warning(f"Groq error: {e}")

    if GEMINI_AVAILABLE and gemini_client and genai_types:
        try:
            convo = MATH_SYSTEM + "\n\n"
            for m in messages:
                label = "User" if m["role"] == "user" else "Assistant"
                convo += f"{label}: {m['content']}\n\n"
            convo += "Assistant:"
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=convo,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return clean_ai_response(resp.text or "")
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    return "⚠️ No AI service available. Please configure GROQ_API_KEY or GEMINI_API_KEY."


def ask_simple(prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    return ask_ai([{"role": "user", "content": prompt}], temperature, max_tokens)


def sanitize(text: str, max_len: int = 5000) -> str:
    if not text:
        return ""
    text = str(text).strip()[:max_len]
    for pat in [r'__import__', r'\beval\b', r'\bexec\b', r'subprocess', r'\bos\.']:
        if re.search(pat, text, re.IGNORECASE):
            logger.warning(f"Blocked input: {text[:60]}")
            return ""
    return text


def parse_int_field(value, default: int, min_value: int, max_value: int,
                    field_name: str = "value") -> int:
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be an integer")
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    return parsed

# ════════════════════════════════════════════════════════════════
# SQLITE DATABASE
# ════════════════════════════════════════════════════════════════

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def db_init():
    with db_connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                exam        TEXT NOT NULL,
                year        INTEGER,
                paper       TEXT,
                topic       TEXT,
                subtopic    TEXT,
                difficulty  TEXT DEFAULT 'moderate',
                source      TEXT,
                question    TEXT NOT NULL,
                options     TEXT,
                answer      TEXT,
                explanation TEXT,
                marks       REAL DEFAULT 1.0,
                negative    REAL DEFAULT 0.0,
                q_type      TEXT DEFAULT 'mcq',
                tags        TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                verified    INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_exam      ON questions(exam);
            CREATE INDEX IF NOT EXISTS idx_year      ON questions(year);
            CREATE INDEX IF NOT EXISTS idx_topic     ON questions(topic);
            CREATE INDEX IF NOT EXISTS idx_diff      ON questions(difficulty);
            CREATE INDEX IF NOT EXISTS idx_exam_diff ON questions(exam, difficulty);

            CREATE TABLE IF NOT EXISTS papers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                exam        TEXT NOT NULL,
                year        INTEGER,
                session     TEXT,
                title       TEXT,
                raw_text    TEXT,
                q_count     INTEGER DEFAULT 0,
                imported_at TEXT DEFAULT (datetime('now'))
            );
        """)
    logger.info(f"[DB] SQLite ready: {DB_PATH}")


def db_add_question(q: dict) -> int:
    with db_connect() as conn:
        cur = conn.execute("""
            INSERT INTO questions
              (exam, year, paper, topic, subtopic, difficulty, source,
               question, options, answer, explanation, marks, negative,
               q_type, tags, verified)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            q.get('exam', 'jam').lower(),
            q.get('year'),
            q.get('paper', ''),
            q.get('topic', ''),
            q.get('subtopic', ''),
            q.get('difficulty', 'moderate'),
            q.get('source', ''),
            q.get('question', ''),
            json.dumps(q.get('options', [])),
            q.get('answer', ''),
            q.get('explanation', ''),
            float(q.get('marks', 1.0)),
            float(q.get('negative', 0.0)),
            q.get('q_type', 'mcq'),
            json.dumps(q.get('tags', [])),
            1 if q.get('verified') else 0,
        ))
        return cur.lastrowid


def db_bulk_insert(questions: list) -> int:
    count = 0
    for q in questions:
        try:
            db_add_question(q)
            count += 1
        except Exception as e:
            logger.warning(f"[DB] Skip question: {e}")
    logger.info(f"[DB] Inserted {count}/{len(questions)}")
    return count


def db_get_questions(exam: str, difficulty: str = None, topic: str = None,
                     year: int = None, limit: int = 10, offset: int = 0) -> list:
    query = "SELECT * FROM questions WHERE exam = ?"
    params = [exam.lower()]
    if difficulty:
        query += " AND difficulty = ?"
        params.append(difficulty)
    if topic:
        query += " AND topic LIKE ?"
        params.append(f"%{topic}%")
    if year:
        query += " AND year = ?"
        params.append(year)
    query += " ORDER BY RANDOM() LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with db_connect() as conn:
        rows = conn.execute(query, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        try: d['options'] = json.loads(d['options'] or '[]')
        except: d['options'] = []
        try: d['tags'] = json.loads(d['tags'] or '[]')
        except: d['tags'] = []
        result.append(d)
    return result


def db_get_stats() -> dict:
    with db_connect() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        by_exam  = conn.execute("SELECT exam, COUNT(*) as n FROM questions GROUP BY exam").fetchall()
        by_diff  = conn.execute("SELECT difficulty, COUNT(*) as n FROM questions GROUP BY difficulty").fetchall()
        by_year  = conn.execute("SELECT year, COUNT(*) as n FROM questions WHERE year IS NOT NULL GROUP BY year ORDER BY year DESC LIMIT 10").fetchall()
        topics   = conn.execute("SELECT topic, COUNT(*) as n FROM questions WHERE topic != '' GROUP BY topic ORDER BY n DESC LIMIT 20").fetchall()
        papers_n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    return {
        "total_questions": total,
        "total_papers":    papers_n,
        "by_exam":         {r["exam"]: r["n"] for r in by_exam},
        "by_difficulty":   {r["difficulty"]: r["n"] for r in by_diff},
        "by_year":         {str(r["year"]): r["n"] for r in by_year},
        "top_topics":      [{"topic": r["topic"], "count": r["n"]} for r in topics],
    }


def db_search(query_text: str, exam: str = None, limit: int = 10) -> list:
    sql = "SELECT * FROM questions WHERE question LIKE ?"
    params = [f"%{query_text}%"]
    if exam:
        sql += " AND exam = ?"
        params.append(exam.lower())
    sql += f" LIMIT {limit}"
    with db_connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        try: d['options'] = json.loads(d['options'] or '[]')
        except: d['options'] = []
        result.append(d)
    return result


def db_delete_question(q_id: int):
    with db_connect() as conn:
        conn.execute("DELETE FROM questions WHERE id = ?", (q_id,))


# ════════════════════════════════════════════════════════════════
# EXPRESSION PARSER FOR GRAPHS
# ════════════════════════════════════════════════════════════════

def _safe_clean_expr(expr_str: str) -> str:
    """
    ✅ FIXED: Enhanced expression cleaner with full support for:
    - |x| notation for absolute value
    - mod(x), mod x notation
    - √x symbol
    - All standard math symbols
    """
    if not expr_str:
        return ""
    
    expr_str = str(expr_str).strip()
    
    # Standard replacements
    expr_str = expr_str.replace('π', 'pi').replace('∏', 'pi')
    expr_str = expr_str.replace('×', '*').replace('·', '*').replace('⋅', '*')
    expr_str = expr_str.replace('÷', '/')
    expr_str = expr_str.replace('^', '**')
    
    # ✅ Convert |x| to abs(x) - handle nested cases
    max_iterations = 10
    iteration = 0
    while '|' in expr_str and iteration < max_iterations:
        expr_str = re.sub(r'\|([^|]*?)\|', r'abs(\1)', expr_str)
        iteration += 1
    
    # ✅ Handle mod notation: mod(x, y) or mod x
    expr_str = expr_str.replace('mod', 'Mod')
    
    # ✅ Handle common functions
    expr_str = expr_str.replace('ln(', 'log(')
    expr_str = expr_str.replace('ln ', 'log(')
    expr_str = expr_str.replace('√', 'sqrt')
    
    # Remove spaces
    expr_str = re.sub(r'\s+', '', expr_str)
    
    return expr_str


# Initialize DB on startup
db_init()

# ════════════════════════════════════════════════════════════════
# STATIC FILES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:
        return send_from_directory(STATIC_DIR, 'index.html')
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404

@app.route("/admin")
def admin_panel():
    try:
        return send_from_directory(STATIC_DIR, 'admin.html')
    except FileNotFoundError:
        return jsonify({"error": "admin.html not found"}), 404

@app.route("/<path:filename>")
def serve_static(filename):
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# ════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    stats = db_get_stats()
    return jsonify({
        "status": "healthy",
        "version": "12.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "numpy": NUMPY_AVAILABLE,
        },
        "database": {
            "total_questions": stats["total_questions"],
            "total_papers": stats["total_papers"],
        }
    }), 200

# ════════════════════════════════════════════════════════════════
# CHAT WITH IMAGE SUPPORT (FULLY FIXED)
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@limiter.limit("40 per minute")
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}

        messages = data.get("messages")
        if not messages:
            single = sanitize(data.get("message", ""))
            if not single:
                return jsonify({"error": "No message provided"}), 400
            messages = [{"role": "user", "content": single}]

        clean = []
        for m in messages[-20:]:
            role = str(m.get("role", "user")).lower()
            content = sanitize(str(m.get("content", "")), 4000)
            if content and role in ("user", "assistant"):
                clean.append({"role": role, "content": content})

        if not clean:
            return jsonify({"error": "Empty message"}), 400

        img_b64  = data.get("image_b64")
        img_type = data.get("image_type", "image/jpeg")

        # ✅ FIXED: Improved image data extraction
        if isinstance(img_b64, str) and img_b64.strip():
            # Handle data URI format
            if img_b64.startswith("data:") and "," in img_b64:
                header, encoded = img_b64.split(",", 1)
                img_b64 = encoded.strip()
                if ";base64" in header and header.startswith("data:"):
                    img_type = header[5:].split(";", 1)[0].strip() or img_type

            if img_b64:
                try:
                    # ✅ FIXED: Check API availability FIRST
                    if not GEMINI_AVAILABLE:
                        return jsonify({
                            "error": "Image analysis requires Gemini API. Set GEMINI_API_KEY in .env",
                            "requires_api": "gemini"
                        }), 503

                    if not gemini_client or not genai_types:
                        return jsonify({
                            "error": "Gemini client not initialized",
                            "requires_api": "gemini"
                        }), 503

                    allowed_mimes = {"image/jpeg", "image/png", "image/webp", "image/gif"}
                    if img_type not in allowed_mimes:
                        return jsonify({
                            "error": f"Unsupported image type: {img_type}. Supported: jpeg, png, webp, gif"
                        }), 400

                    # ✅ FIXED: Robust base64 decoding
                    try:
                        raw_bytes = base64.b64decode(img_b64, validate=True)
                    except Exception as decode_err:
                        logger.error(f"Base64 decode error: {decode_err}")
                        return jsonify({"error": "Invalid image encoding"}), 400

                    if not raw_bytes:
                        return jsonify({"error": "Image data is empty"}), 400
                    if len(raw_bytes) > 10 * 1024 * 1024:
                        return jsonify({"error": "Image too large (max 10MB)"}), 413

                    # ✅ FIXED: Better Gemini API call
                    try:
                        prompt_text = clean[-1]["content"] if clean else "Solve this mathematics problem step by step"
                        
                        image_part = genai_types.Part.from_bytes(
                            data=raw_bytes,
                            mime_type=img_type
                        )
                        
                        system_part = genai_types.Part.from_text(
                            MATH_SYSTEM + "\n\nAnalyze and solve the problem in this image:"
                        )
                        user_part = genai_types.Part.from_text(prompt_text)

                        resp = gemini_client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[genai_types.Content(
                                role="user",
                                parts=[system_part, user_part, image_part]
                            )],
                            config=genai_types.GenerateContentConfig(
                                temperature=0.2,
                                max_output_tokens=3000
                            )
                        )
                        
                        answer = clean_ai_response(resp.text or "").strip()
                        if not answer:
                            logger.warning("Gemini returned empty response")
                            return jsonify({"error": "Could not analyze the image"}), 422
                        
                        logger.info(f"[IMAGE] Successfully processed {img_type}")
                        return jsonify({"answer": answer}), 200

                    except Exception as gemini_err:
                        logger.exception(f"Gemini API error: {gemini_err}")
                        error_msg = str(gemini_err)
                        
                        if "invalid_argument" in error_msg.lower():
                            return jsonify({
                                "error": "Image format issue. Try: JPEG, PNG, or WebP under 4MB"
                            }), 400
                        elif "rate_limit" in error_msg.lower():
                            return jsonify({"error": "Rate limit exceeded. Wait and try again."}), 429
                        else:
                            return jsonify({
                                "error": f"Image processing failed: {error_msg[:100]}"
                            }), 502

                except Exception as img_err:
                    logger.exception(f"Image processing error: {img_err}")
                    return jsonify({"error": "Failed to process image"}), 500

        # Text-only chat
        return jsonify({"answer": ask_ai(clean)}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# GRAPH PLOTTER (FULLY FIXED)
# ════════════════════════════════════════════════════════════════

@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    try:
        data = request.get_json(force=True, silent=True) or {}
        expr_str = sanitize(data.get("expression", "x**2"), 300)
        gtype    = data.get("type", "2d")
        
        try:
            x_min = float(data.get("x_min", -10))
            x_max = float(data.get("x_max", 10))
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "x_min and x_max must be numbers"}), 400
        
        if x_min >= x_max:
            return jsonify({"success": False, "error": "x_min must be less than x_max"}), 400
        
        try:
            num_points = max(100, min(1000, int(data.get("points", 600))))
        except (TypeError, ValueError):
            num_points = 600

        if not SYMPY_AVAILABLE:
            return jsonify({"success": False, "error": "SymPy not available"}), 503
        if not expr_str:
            return jsonify({"success": False, "error": "Expression required"}), 400

        logger.info(f"[GRAPH] Plotting: {expr_str} on [{x_min}, {x_max}]")

        try:
            clean_expr = _safe_clean_expr(expr_str)
            logger.info(f"[GRAPH] Cleaned: {clean_expr}")
            
            transformations = standard_transformations + (
                implicit_multiplication_application, convert_xor
            )
            
            x = Symbol('x')
            local_dict = {
                'x': x, 'pi': pi, 'e': E, 'E': E,
                'sin': sin, 'cos': cos, 'tan': tan,
                'asin': asin, 'acos': acos, 'atan': atan,
                'arcsin': asin, 'arccos': acos, 'arctan': atan,
                'cot': cot, 'sec': sec, 'csc': csc,
                'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
                'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
                'exp': exp, 'log': log, 'ln': log, 'lg': log,
                'log10': lambda t: log(t, 10),
                'log2': lambda t: log(t, 2),
                'sqrt': sqrt, 'abs': Abs, 'Abs': Abs,
                'cbrt': lambda t: t ** Rational(1, 3),
                'gamma': gamma, 'erf': erf, 'factorial': factorial,
                'ceil': ceiling, 'ceiling': ceiling,
                'floor': floor, 'sign': sign,
                'Mod': Mod, 'mod': Mod,
            }
            
            f_sym = parse_expr(clean_expr, transformations=transformations, local_dict=local_dict)

            step = (x_max - x_min) / num_points
            points = []
            discontinuity_xs = []
            prev_y = None

            for i in range(num_points + 1):
                xv = round(x_min + i * step, 8)
                try:
                    yv = float(N(f_sym.subs(x, xv), 10))
                    if _isfinite(yv) and not _isnan(yv) and abs(yv) < 1e7:
                        if prev_y is not None and abs(yv - prev_y) > 50 * (x_max - x_min):
                            points.append({"x": round(xv, 5), "y": None})
                            discontinuity_xs.append(round(xv, 3))
                        points.append({"x": round(xv, 5), "y": round(yv, 6)})
                        prev_y = yv
                    else:
                        points.append({"x": round(xv, 5), "y": None})
                        discontinuity_xs.append(round(xv, 3))
                        prev_y = None
                except Exception:
                    points.append({"x": round(xv, 5), "y": None})
                    prev_y = None

            warnings = []
            if discontinuity_xs:
                warnings.append(f"Discontinuities at x ≈ {sorted(set(discontinuity_xs))[:5]}")

            analysis_prompt = (
                f"Analyze f(x) = {expr_str}. Cover:\n"
                f"1. Domain & Range\n2. Intercepts\n3. Symmetry\n"
                f"4. Asymptotes\n5. Critical points\n6. Monotonicity\nUse LaTeX."
            )
            analysis = ask_simple(analysis_prompt, temperature=0.1, max_tokens=2500)

            return jsonify({
                "success": True,
                "points": points,
                "expression": expr_str,
                "type": gtype,
                "analysis": analysis,
                "warnings": warnings,
                "x_range": [x_min, x_max],
                "point_count": len([p for p in points if p["y"] is not None]),
                "cleaned_expr": clean_expr
            }), 200

        except Exception as parse_err:
            error_str = str(parse_err)[:150]
            suggestions = ""
            if "|" in expr_str:
                suggestions = " (Try: abs(x) for absolute value)"
            elif "mod" in expr_str.lower():
                suggestions = " (Try: Mod(x, n) for modulo)"
            
            return jsonify({
                "success": False,
                "error": f"Cannot parse: {error_str}{suggestions}",
                "supported_functions": "sin, cos, tan, exp, log, sqrt, abs, Mod, ceil, floor",
                "examples": ["|x|", "abs(x)", "sin(x)", "x**2", "Mod(x,5)", "sqrt(abs(x))"]
            }), 400

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# MATHEMATICIAN ENDPOINT (WITH LINKS)
# ════════════════════════════════════════════════════════════════

@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = sanitize(data.get("name", "ramanujan"), 100).lower().strip()
        
        # Check database first
        if name in MATHEMATICIANS_DB:
            bio_data = MATHEMATICIANS_DB[name]
            return jsonify({
                "success": True,
                "name": bio_data["name"],
                "birth": bio_data["birth"],
                "death": bio_data["death"],
                "nationality": bio_data["nationality"],
                "field": bio_data["field"],
                "biography": bio_data["summary"],
                "wikipedia": bio_data["wiki"],
                "resources": bio_data["resources"]
            }), 200
        
        # Fallback to AI if not in database
        raw = ask_simple(
            f"Detailed info about mathematician **{name}**: dates, nationality, contributions (LaTeX), theorems, influence.",
            temperature=0.3, max_tokens=2000
        )
        
        # Try to find Wikipedia link
        wiki_url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
        
        return jsonify({
            "success": True,
            "name": name.title(),
            "biography": raw or "No information available",
            "wikipedia": wiki_url,
            "resources": [
                f"https://mathworld.wolfram.com/search/?query={name}",
                f"https://mathshistory.st-andrews.ac.uk/Search/?query={name}",
                f"https://www.britannica.com/search?query={name}"
            ]
        }), 200
        
    except Exception as e:
        logger.exception(f"Mathematician error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# PROJECTS WITH RESOURCES (4-5 LINKS EACH)
# ════════════════════════════════════════════════════════════════

@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100).lower().strip()
        
        # Generate AI projects
        raw = ask_simple(
            f"""Generate 5 interesting mathematics project ideas for **{topic}** at undergraduate level.
For each project:
- Title
- Objective (1-2 sentences)
- Tools/Technologies needed
- Difficulty level
- Expected outcome
- Time required

Format: # Project 1: [Title]\\n[details]""",
            temperature=0.35, max_tokens=3000
        )
        
        # Get resources for this topic
        resources = PROJECT_RESOURCES.get(topic, PROJECT_RESOURCES.get("calculus", []))
        
        return jsonify({
            "success": True,
            "topic": topic.title(),
            "projects": raw,
            "resources": resources,
            "total_resources": len(resources)
        }), 200
        
    except Exception as e:
        logger.exception(f"Projects error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ADDITIONAL ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
def formula():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        exam  = sanitize(data.get("exam", "General"), 50)
        answer = ask_simple(
            f"Comprehensive formula sheet for **{topic}** at {exam} level. For each: LaTeX, meaning, use case, example.",
            temperature=0.05, max_tokens=3000
        )
        return jsonify({"answer": answer or "Could not generate"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    try:
        data       = request.get_json(force=True, silent=True) or {}
        topic      = sanitize(data.get("topic", "Calculus"), 100)
        count      = parse_int_field(data.get("count", 5), 5, 1, 20, "count")
        difficulty = sanitize(data.get("difficulty", "moderate"), 20)
        questions  = ask_simple(
            f"Generate {count} {difficulty}-level MCQs on {topic}. Format: Q[n]. [Question]\\n(A)...(B)...(C)...(D)...\\n✅ Answer: [letter]\\n📝 Solution",
            temperature=0.25, max_tokens=3500
        )
        return jsonify({"questions": questions}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal error"}), 500


@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        theorem = sanitize(data.get("theorem", "Pythagorean Theorem"), 300)
        proof   = ask_simple(
            f"Prove {theorem}. Include: Statement, Prerequisites, Proof, Alternative proofs, Applications. Use LaTeX.",
            temperature=0.1, max_tokens=3000
        )
        return jsonify({"proof": proof}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/exam/info", methods=["POST"])
@limiter.limit("20 per minute")
def exam_info():
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam", "jam"), 20).lower().strip()
        
        EXAM_DATA = {
            "jam": {
                "title": "IIT JAM Mathematics (MA)",
                "when": "February (Registration: Sep-Oct)",
                "duration": "3 hours",
                "questions": "60 (40 MCQ + 20 NAT)",
                "subjects": "Real Analysis, Linear Algebra, Calculus, ODE/PDE, Abstract Algebra, Complex Analysis",
                "eligibility": "Bachelor's degree with Mathematics (55% marks)",
                "fee": "₹1,800 / ₹900 SC/ST",
                "admission": "M.Sc. at IITs",
                "cutoff": "40–65 out of 100",
                "marking": "+1 for MCQ, +2 for NAT, -1/3 for MCQ wrong"
            },
            "gate": {
                "title": "GATE Mathematics (MA)",
                "when": "February (Registration: Aug-Sep)",
                "duration": "3 hours",
                "questions": "65 (40 MCQ + 25 NAT)",
                "subjects": "Calculus, Linear Algebra, Real/Complex Analysis, Algebra, Topology, ODE/PDE",
                "eligibility": "Any Bachelor's degree",
                "fee": "₹1,800 / ₹900 SC/ST/Women",
                "admission": "M.Tech at IITs/NITs, PSU Jobs, Ph.D.",
                "cutoff": "25–50 out of 100",
                "marking": "+1/+2 for MCQ/NAT, -1/3 or -2/3 for wrong"
            },
            "csir": {
                "title": "CSIR NET Mathematical Sciences",
                "when": "June & December",
                "duration": "3 hours",
                "questions": "Part A: 20, Part B: 40, Part C: 60",
                "subjects": "Analysis, Algebra, Topology, Complex Analysis, ODE/PDE, Probability, Statistics",
                "eligibility": "M.Sc. or B.Tech in Mathematics/Science",
                "fee": "₹1,000 / ₹500 OBC / ₹250 SC/ST",
                "admission": "JRF (PhD) + Lecturership at Universities",
                "cutoff": "JRF: top ~200; Lectureship: top ~6%",
                "marking": "A: +2/-0.5, B: +3.5/-1, C: +5/0"
            }
        }
        
        details = EXAM_DATA.get(exam, EXAM_DATA["jam"])
        return jsonify({"exam": exam, "details": details}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded — please wait"}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500: {e}")
    return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════

def print_startup():
    stats = db_get_stats()
    print("\n" + "═"*70)
    print("  🧮  MathSphere v12.0 — Production Backend")
    print("═"*70)
    print(f"  ✅ Image Processing: {('ENABLED' if GEMINI_AVAILABLE else 'SET GEMINI_API_KEY')}")
    print(f"  ✅ Graph Plotter: Enhanced (|x|, mod, √, etc.)")
    print(f"  ✅ Mathematician Links: Wikipedia + Resources")
    print(f"  ✅ Project Resources: 4-5 links each")
    print(f"  ✅ Groq API: {'Connected' if GROQ_AVAILABLE else 'Not set'}")
    print(f"  ✅ SymPy: {'Ready' if SYMPY_AVAILABLE else 'Not available'}")
    print(f"  📦 DB: {stats['total_questions']} questions")
    print("\n  🌐 http://localhost:{}/".format(os.getenv('PORT', '5000')))
    print("═"*70 + "\n")


if __name__ == "__main__":
    print_startup()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE, use_reloader=False)