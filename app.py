"""
MathSphere v11.1 — Complete Production Backend
All-in-one file: Flask app + SQLite PYQ database + AI extraction + Admin routes
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
        ceiling, floor, sign,
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
# SQLITE PYQ DATABASE
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


# AI extraction
EXTRACTION_PROMPT = """Extract ALL multiple choice questions from the text below.
Return ONLY a JSON array. No markdown, no code fences, nothing else.
Start with [ and end with ].

Each object must have:
- "exam": "jam" | "gate" | "csir"  (guess from context, default "jam")
- "year": integer or null
- "topic": e.g. "Real Analysis"
- "difficulty": "easy" | "moderate" | "difficult"
- "question": full question text (keep any LaTeX)
- "options": exactly 4 strings ["A) ...", "B) ...", "C) ...", "D) ..."]
- "answer": correct option string
- "explanation": brief explanation
- "q_type": "mcq"

TEXT:
{text}"""


def ai_extract_questions(raw_text: str) -> list:
    MAX_CHUNK = 4000
    chunks = [raw_text[i:i+MAX_CHUNK] for i in range(0, len(raw_text), MAX_CHUNK)]
    all_questions = []
    for i, chunk in enumerate(chunks):
        logger.info(f"[EXTRACT] Chunk {i+1}/{len(chunks)}")
        prompt = EXTRACTION_PROMPT.replace('{text}', chunk)
        raw = ask_simple(prompt, temperature=0.1, max_tokens=3000)
        try:
            clean = re.sub(r'```(?:json)?|```', '', raw or '').strip()
            s = clean.find('[')
            e = clean.rfind(']') + 1
            if s >= 0 and e > s:
                qs = json.loads(clean[s:e])
                all_questions.extend(qs)
                logger.info(f"[EXTRACT] Chunk {i+1}: got {len(qs)} questions")
        except Exception as ex:
            logger.warning(f"[EXTRACT] Parse error chunk {i+1}: {ex}")
    return all_questions


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
        return jsonify({"error": "admin.html not found — put it in static/"}), 404

@app.route("/<path:filename>")
def serve_static(filename):
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# ════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    stats = db_get_stats()
    return jsonify({
        "status": "healthy", "version": "11.1",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": GROQ_AVAILABLE, "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE, "numpy": NUMPY_AVAILABLE,
        },
        "database": {
            "total_questions": stats["total_questions"],
            "total_papers":    stats["total_papers"],
        }
    }), 200

# ════════════════════════════════════════════════════════════════
# CHAT
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

        if isinstance(img_b64, str) and img_b64.startswith("data:") and "," in img_b64:
            header, encoded = img_b64.split(",", 1)
            img_b64 = encoded
            if ";base64" in header and header.startswith("data:"):
                img_type = header[5:].split(";", 1)[0] or img_type

        if img_b64:
            try:
                allowed_mimes = {"image/jpeg", "image/png", "image/webp", "image/gif"}
                if img_type not in allowed_mimes:
                    return jsonify({"error": f"Unsupported image type: {img_type}"}), 400
                try:
                    raw_bytes = base64.b64decode(img_b64, validate=True)
                except Exception:
                    return jsonify({"error": "Invalid image payload"}), 400
                if not raw_bytes or len(raw_bytes) > 10 * 1024 * 1024:
                    return jsonify({"error": "Invalid or too large image"}), 400
                if not (GEMINI_AVAILABLE and gemini_client and genai_types):
                    return jsonify({"error": "Image analysis requires Gemini API key"}), 503

                prompt_text = clean[-1]["content"] if clean else "Solve this mathematics problem step by step"
                image_part  = genai_types.Part.from_bytes(data=raw_bytes, mime_type=img_type)
                system_part = genai_types.Part.from_text(MATH_SYSTEM + "\n\nSolve the problem in this image:")
                user_part   = genai_types.Part.from_text(prompt_text)

                resp = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[genai_types.Content(role="user", parts=[system_part, user_part, image_part])]
                )
                answer = clean_ai_response(resp.text or "").strip()
                if not answer:
                    return jsonify({"error": "Could not read image"}), 422
                return jsonify({"answer": answer}), 200
            except Exception as img_err:
                logger.exception(f"Image error: {img_err}")
                return jsonify({"error": "Image processing failed"}), 502

        return jsonify({"answer": ask_ai(clean)}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# GRAPH PLOTTER
# ════════════════════════════════════════════════════════════════

def _safe_clean_expr(expr_str: str) -> str:
    expr_str = expr_str.replace('π', 'pi').replace('×', '*').replace('÷', '/')
    expr_str = expr_str.replace('^', '**')
    expr_str = re.sub(r'\s+', '', expr_str)
    return expr_str


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
                'exp': exp, 'log': log, 'ln': log,
                'log10': lambda t: log(t, 10), 'log2': lambda t: log(t, 2),
                'sqrt': sqrt, 'abs': Abs, 'Abs': Abs,
                'cbrt': lambda t: t ** Rational(1, 3),
                'gamma': gamma, 'erf': erf, 'factorial': factorial,
                'ceil': ceiling, 'ceiling': ceiling,
                'floor': floor, 'sign': sign,
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
                f"4. Asymptotes\n5. Critical points\n6. Inflection points\n"
                f"7. Monotonicity\n8. Behavior as x→±∞\nUse LaTeX."
            )
            analysis = ask_simple(analysis_prompt, temperature=0.1, max_tokens=2500)

            return jsonify({
                "success": True, "points": points, "expression": expr_str,
                "type": gtype, "analysis": analysis, "warnings": warnings,
                "x_range": [x_min, x_max],
                "point_count": len([p for p in points if p["y"] is not None])
            }), 200

        except Exception as parse_err:
            return jsonify({
                "success": False,
                "error": f"Cannot parse: {str(parse_err)[:120]}. Use: sin(x), x**2, exp(x), log(x)"
            }), 400

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# HARDCODED PYQ FALLBACK
# ════════════════════════════════════════════════════════════════

REAL_PYQS = {
    "jam": {
        "easy": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Real Analysis",
                "question": "Let \\(f: \\mathbb{R} \\to \\mathbb{R}\\) be defined by \\(f(x) = \\begin{cases} x^2 & x \\in \\mathbb{Q} \\\\ 0 & x \\notin \\mathbb{Q} \\end{cases}\\). At which point(s) is \\(f\\) continuous?",
                "options": ["A) Nowhere", "B) At x = 0 only", "C) At all rational x", "D) Everywhere"],
                "answer": "B) At x = 0 only",
                "explanation": "At x=0: |f(x)-f(0)| ≤ x² → 0. For x≠0: neighbourhood contains rationals and irrationals giving different limits."
            },
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Linear Algebra",
                "question": "The rank of \\(A = \\begin{pmatrix}1&2&3\\\\2&4&6\\\\3&6&9\\end{pmatrix}\\) is:",
                "options": ["A) 0", "B) 1", "C) 2", "D) 3"],
                "answer": "B) 1",
                "explanation": "R₂=2R₁ and R₃=3R₁. Only one linearly independent row. Rank=1."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Calculus",
                "question": "Evaluate \\(\\int_0^{\\pi/2} \\sin^2 x\\, dx\\).",
                "options": ["A) π/2", "B) π/4", "C) 1", "D) π"],
                "answer": "B) π/4",
                "explanation": "sin²x=(1-cos2x)/2. Integral=[x/2-sin2x/4]₀^{π/2}=π/4."
            },
            {
                "year": 2021, "source": "IIT JAM Mathematics", "topic": "Calculus",
                "question": "The value of \\(\\lim_{x \\to 0} \\frac{e^x - 1 - x}{x^2}\\) is:",
                "options": ["A) 0", "B) 1", "C) 1/2", "D) ∞"],
                "answer": "C) 1/2",
                "explanation": "By Taylor: eˣ=1+x+x²/2+… so (eˣ-1-x)/x²→1/2."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Complex Analysis",
                "question": "The residue of \\(f(z) = \\frac{e^z}{(z-1)^2}\\) at \\(z = 1\\) is:",
                "options": ["A) e", "B) e/2", "C) 2e", "D) 1"],
                "answer": "A) e",
                "explanation": "Pole of order 2: Res = d/dz[eᶻ]|_{z=1} = e."
            },
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Abstract Algebra",
                "question": "In \\(\\mathbb{Z}_{12}\\), the order of element 8 is:",
                "options": ["A) 2", "B) 3", "C) 4", "D) 6"],
                "answer": "B) 3",
                "explanation": "ord(8)=12/gcd(8,12)=12/4=3."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Differential Equations",
                "question": "General solution of \\(y'' - 3y' + 2y = 0\\) is:",
                "options": ["A) \\(c_1 e^x + c_2 e^{2x}\\)", "B) \\(c_1 e^{-x}+c_2 e^{-2x}\\)", "C) \\((c_1+c_2 x)e^x\\)", "D) \\(c_1\\cos x+c_2\\sin x\\)"],
                "answer": "A) \\(c_1 e^x + c_2 e^{2x}\\)",
                "explanation": "Char. eq: r²-3r+2=0 → r=1,2."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Real Analysis",
                "question": "The Wronskian of \\(y_1=e^x\\) and \\(y_2=e^{-x}\\) at \\(x=0\\) is:",
                "options": ["A) 0", "B) 1", "C) -2", "D) 2"],
                "answer": "C) -2",
                "explanation": "W=y₁y₂'-y₁'y₂=eˣ(-e⁻ˣ)-eˣ(e⁻ˣ)=-1-1=-2."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Abstract Algebra",
                "question": "Number of group homomorphisms from \\(\\mathbb{Z}_{12}\\) to \\(\\mathbb{Z}_8\\):",
                "options": ["A) 4", "B) 2", "C) 8", "D) 1"],
                "answer": "A) 4",
                "explanation": "Count = gcd(12,8) = 4."
            },
        ]
    },
    "gate": {
        "easy": [
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Calculus",
                "question": "\\(\\lim_{x \\to 0} \\dfrac{\\sin x}{x}\\) equals:",
                "options": ["A) 0", "B) 1", "C) ∞", "D) undefined"],
                "answer": "B) 1",
                "explanation": "Standard fundamental limit."
            },
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Linear Algebra",
                "question": "Eigenvalues of \\(\\begin{pmatrix}0&1\\\\-1&0\\end{pmatrix}\\):",
                "options": ["A) 0,0", "B) 1,-1", "C) i,-i", "D) 1,1"],
                "answer": "C) i,-i",
                "explanation": "λ²+1=0 → λ=±i."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Complex Analysis",
                "question": "Evaluate \\(\\oint_{|z|=2} \\frac{dz}{z^2+1}\\):",
                "options": ["A) 0", "B) πi", "C) 2πi", "D) -2πi"],
                "answer": "A) 0",
                "explanation": "Residues at z=±i cancel."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "PDE",
                "question": "The PDE \\(u_{xx} - 4u_{yy} = 0\\) is:",
                "options": ["A) Parabolic", "B) Elliptic", "C) Hyperbolic", "D) None"],
                "answer": "C) Hyperbolic",
                "explanation": "B²-4AC=16>0 → Hyperbolic."
            },
        ]
    },
    "csir": {
        "easy": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Real Analysis",
                "question": "In \\(\\mathbb{R}\\), Cauchy \\(\\Leftrightarrow\\):",
                "options": ["A) Bounded", "B) Monotone", "C) Convergent", "D) Bounded+monotone"],
                "answer": "C) Convergent",
                "explanation": "ℝ is complete."
            },
            {
                "year": 2022, "source": "CSIR NET Mathematical Sciences", "topic": "Group Theory",
                "question": "Every subgroup of index 2 is:",
                "options": ["A) Cyclic", "B) Normal", "C) Abelian", "D) Simple"],
                "answer": "B) Normal",
                "explanation": "Index 2 → left cosets = right cosets → normal."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Complex Analysis",
                "question": "\\(f(z) = \\bar{z}\\) is:",
                "options": ["A) Analytic everywhere", "B) Analytic nowhere", "C) Analytic on ℝ", "D) Entire"],
                "answer": "B) Analytic nowhere",
                "explanation": "C-R fails everywhere: ∂u/∂x=1≠∂v/∂y=-1."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Measure Theory",
                "question": "Lebesgue measure of Cantor set:",
                "options": ["A) 1", "B) 1/2", "C) 0", "D) Uncountable"],
                "answer": "C) 0",
                "explanation": "Total measure removed = 1. So Cantor set has measure 0."
            },
        ]
    }
}

# ════════════════════════════════════════════════════════════════
# PYQ ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/pyq/load", methods=["POST"])
@limiter.limit("15 per minute")
def pyq_load():
    try:
        data       = request.get_json(force=True, silent=True) or {}
        exam       = sanitize(data.get("exam", "jam"), 20).lower().strip()
        difficulty = sanitize(data.get("difficulty", "moderate"), 20).lower().strip()
        topic      = sanitize(data.get("topic", ""), 100)
        count      = parse_int_field(data.get("count", 5), 5, 1, 30, "count")
        try:
            year = int(data.get("year")) if data.get("year") else None
        except (ValueError, TypeError):
            year = None

        if exam not in ("jam", "gate", "csir"):
            return jsonify({"success": False, "error": "Invalid exam. Use: jam, gate, csir"}), 400
        if difficulty not in ("easy", "moderate", "difficult"):
            return jsonify({"success": False, "error": "Invalid difficulty. Use: easy, moderate, difficult"}), 400

        # Try DB first
        questions = db_get_questions(exam=exam, difficulty=difficulty,
                                     topic=topic or None, year=year, limit=count)
        source = "database"

        # Fallback to hardcoded
        if not questions:
            pool = list(REAL_PYQS.get(exam, {}).get(difficulty, []))
            random.shuffle(pool)
            questions = pool[:count]
            source = "built-in"

        return jsonify({
            "success": True, "questions": questions,
            "exam": exam, "difficulty": difficulty,
            "count": len(questions), "source": source,
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"PYQ error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/pyq/search", methods=["POST"])
@limiter.limit("20 per minute")
def pyq_search():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        query = sanitize(data.get("query", ""), 200)
        exam  = sanitize(data.get("exam", ""), 20)
        if not query:
            return jsonify({"success": False, "error": "query required"}), 400
        results = db_search(query, exam=exam or None, limit=10)
        return jsonify({"success": True, "results": results, "count": len(results)}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    try:
        return jsonify({"success": True, "stats": db_get_stats()}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/import/text", methods=["POST"])
@limiter.limit("5 per minute")
def admin_import_text():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        text  = sanitize(data.get("text", ""), 20000)
        exam  = sanitize(data.get("exam", "jam"), 20).lower()
        year  = data.get("year")
        paper = sanitize(data.get("paper", ""), 100)

        if not text or len(text) < 50:
            return jsonify({"success": False, "error": "Text too short"}), 400

        with db_connect() as conn:
            conn.execute(
                "INSERT INTO papers (exam, year, session, title, raw_text) VALUES (?,?,?,?,?)",
                (exam, year, paper, f"{exam.upper()} {year or ''} {paper}".strip(), text)
            )

        questions = ai_extract_questions(text)
        for q in questions:
            q['exam']   = exam
            q['year']   = year or q.get('year')
            q['paper']  = paper
            q['source'] = f"Official {exam.upper()} {year or ''} {paper}".strip()

        inserted = db_bulk_insert(questions)

        with db_connect() as conn:
            conn.execute(
                "UPDATE papers SET q_count=? WHERE exam=? AND year=? AND session=?",
                (inserted, exam, year, paper)
            )

        return jsonify({
            "success": True, "extracted": len(questions), "inserted": inserted,
            "message": f"Imported {inserted} questions from {exam.upper()} {year or ''} {paper}".strip()
        }), 200

    except Exception as e:
        logger.exception(f"Import text error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/import/json", methods=["POST"])
@limiter.limit("5 per minute")
def admin_import_json():
    try:
        data      = request.get_json(force=True, silent=True) or {}
        questions = data.get("questions", [])
        if not isinstance(questions, list) or not questions:
            return jsonify({"success": False, "error": "questions array required"}), 400
        if len(questions) > 500:
            return jsonify({"success": False, "error": "Max 500 per import"}), 400
        inserted = db_bulk_insert(questions)
        return jsonify({"success": True, "received": len(questions), "inserted": inserted}), 200
    except Exception as e:
        logger.exception(f"Import JSON error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/questions", methods=["GET"])
def admin_list_questions():
    try:
        exam  = request.args.get("exam", "jam")
        diff  = request.args.get("difficulty", "")
        page  = max(0, int(request.args.get("page", 0)))
        limit = min(50, int(request.args.get("limit", 20)))
        questions = db_get_questions(exam=exam or "jam", difficulty=diff or None,
                                     limit=limit, offset=page * limit)
        return jsonify({"success": True, "questions": questions, "page": page}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/questions/<int:q_id>", methods=["DELETE"])
def admin_delete_question(q_id):
    try:
        db_delete_question(q_id)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# MOCK TEST
# ════════════════════════════════════════════════════════════════

@app.route("/api/mock/generate", methods=["POST"])
@limiter.limit("5 per minute")
def mock_generate():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        exam     = sanitize(data.get("exam", "jam"), 20).lower().strip()
        count    = parse_int_field(data.get("count", 10), 10, 5, 30, "count")
        timed    = bool(data.get("timed", True))
        duration = parse_int_field(data.get("duration", 20), 20, 5, 180, "duration")

        if exam not in ("jam", "gate", "csir"):
            return jsonify({"success": False, "error": "Invalid exam"}), 400

        all_questions = []
        for diff in ["easy", "moderate", "difficult"]:
            qs = db_get_questions(exam=exam, difficulty=diff, limit=20)
            if not qs:
                qs = list(REAL_PYQS.get(exam, {}).get(diff, []))
                for q in qs: q['difficulty'] = diff
            all_questions.extend(qs)

        random.shuffle(all_questions)
        easy_qs = [q for q in all_questions if q.get("difficulty") == "easy"]
        mod_qs  = [q for q in all_questions if q.get("difficulty") == "moderate"]
        hard_qs = [q for q in all_questions if q.get("difficulty") == "difficult"]

        n_easy = max(1, int(count * 0.3))
        n_hard = max(1, int(count * 0.2))
        n_mod  = count - n_easy - n_hard
        selected = (easy_qs[:n_easy] + mod_qs[:n_mod] + hard_qs[:n_hard])
        random.shuffle(selected)

        answer_key = []
        test_questions = []
        for i, q in enumerate(selected):
            answer_key.append({"index": i, "answer": q.get("answer",""), "explanation": q.get("explanation","")})
            test_q = {k: v for k, v in q.items() if k not in ("answer","explanation")}
            test_q["index"] = i
            test_questions.append(test_q)

        return jsonify({
            "success": True,
            "test": {
                "questions": test_questions, "exam": exam,
                "count": len(test_questions), "timed": timed,
                "duration_seconds": duration * 60,
                "instructions": f"{'Timed: ' + str(duration) + ' min. ' if timed else ''}Attempt all questions."
            },
            "answer_key": answer_key
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Mock generate error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/mock/evaluate", methods=["POST"])
@limiter.limit("10 per minute")
def mock_evaluate():
    try:
        data         = request.get_json(force=True, silent=True) or {}
        user_answers = data.get("answers", [])
        answer_key   = data.get("answer_key", [])
        questions    = data.get("questions", [])
        if not user_answers or not answer_key:
            return jsonify({"success": False, "error": "answers and answer_key required"}), 400

        key_map = {item["index"]: item for item in answer_key}
        correct = 0
        results = []
        for ua in user_answers:
            idx         = ua.get("index")
            selected    = ua.get("selected_option", "")
            correct_ans = key_map.get(idx, {}).get("answer", "")
            explanation = key_map.get(idx, {}).get("explanation", "")
            is_correct  = (selected.strip().upper()[0:1] == correct_ans.strip().upper()[0:1]) if selected else False
            if is_correct: correct += 1
            results.append({"index": idx, "selected": selected, "correct_answer": correct_ans,
                             "is_correct": is_correct, "explanation": explanation})

        total     = len(user_answers)
        score_pct = round((correct / total) * 100, 1) if total else 0
        grade     = ("Excellent" if score_pct >= 80 else "Good" if score_pct >= 60
                     else "Average" if score_pct >= 40 else "Needs Improvement")

        wrong_topics = list(set(filter(None, [
            q.get("topic","") for i, q in enumerate(questions)
            if i < len(results) and not results[i]["is_correct"]
        ])))

        feedback = ask_simple(
            f"Student scored {correct}/{total} ({score_pct}%). Weak: {', '.join(wrong_topics) or 'none'}.\n"
            f"Give: 1)performance analysis 2)weak areas 3)study plan 4)tips. Concise.",
            temperature=0.3, max_tokens=600
        )

        return jsonify({
            "success": True, "score": correct, "total": total,
            "percentage": score_pct, "grade": grade,
            "results": results, "weak_topics": wrong_topics, "ai_feedback": feedback
        }), 200

    except Exception as e:
        logger.exception(f"Evaluate error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# OTHER ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
def formula():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic","Calculus"), 100)
        exam  = sanitize(data.get("exam","General"), 50)
        answer = ask_simple(
            f"Comprehensive formula sheet for **{topic}** at {exam} level.\n"
            f"For each formula: LaTeX, meaning, when to use, quick example. Use LaTeX.",
            temperature=0.05, max_tokens=3000
        )
        return jsonify({"answer": answer or "Could not generate"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("10 per minute")
def competition_problems():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        category = sanitize(data.get("category","IMO"), 50)
        count    = parse_int_field(data.get("count",5), 5, 1, 20, "count")
        problems = ask_simple(
            f"Generate {count} {category}-style problems with full solutions. Use LaTeX.",
            temperature=0.35, max_tokens=3500
        )
        return jsonify({"problems": problems}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    try:
        data       = request.get_json(force=True, silent=True) or {}
        topic      = sanitize(data.get("topic","Calculus"), 100)
        count      = parse_int_field(data.get("count",5), 5, 1, 20, "count")
        difficulty = sanitize(data.get("difficulty","moderate"), 20)
        questions  = ask_simple(
            f"Generate {count} {difficulty}-level MCQs on **{topic}**.\n"
            f"Format: Q[n]. [Question]\n(A)...(B)...(C)...(D)...\n✅ Answer: [letter]\n📝 Solution: [steps]",
            temperature=0.25, max_tokens=3500
        )
        return jsonify({"questions": questions}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/research", methods=["POST"])
@limiter.limit("15 per minute")
def research_hub():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        rtype = sanitize(data.get("type","topic"), 50)
        query = sanitize(data.get("query",""), 500)
        if not query:
            return jsonify({"error": "Query required"}), 400
        prompts = {
            "literature": f"Literature review on '{query}': key results, open problems, papers.",
            "topic":      f"Deep exploration of '{query}': definitions, theorems, examples.",
            "methods":    f"Problem-solving methods for '{query}' with examples.",
            "career":     f"Career guidance for '{query}' in India: jobs, exams, salaries.",
            "resources":  f"Best study resources for '{query}' at UG/PG level.",
        }
        response = ask_simple(prompts.get(rtype, prompts["topic"]), temperature=0.2, max_tokens=2500)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/exam/info", methods=["POST"])
@limiter.limit("20 per minute")
def exam_info():
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam","jam"), 20).lower().strip()
        EXAM_DATA = {
            "jam":  {"title":"IIT JAM Mathematics (MA)","when":"February (reg: Sep–Oct)","duration":"3 hours","questions":"60 (MCQ+MSQ+NAT)","subjects":"Real Analysis, Linear Algebra, Calculus, ODE/PDE, Abstract Algebra, Complex Analysis","eligibility":"Bachelor's with Maths (55%)","fee":"₹1,800 / ₹900 SC/ST","admission":"M.Sc. at IITs","cutoff":"40–65/100","marking":"+1/+2, −1/3 or −2/3"},
            "gate": {"title":"GATE Mathematics (MA)","when":"February (reg: Aug–Sep)","duration":"3 hours","questions":"65 (MCQ+MSQ+NAT)","subjects":"Calculus, Linear Algebra, Real/Complex Analysis, Algebra, Topology, PDE, Probability","eligibility":"Any Bachelor's","fee":"₹1,800 / ₹900 SC/ST/Women","admission":"M.Tech IITs/NITs, PSU, Ph.D.","cutoff":"25–50/100","marking":"+1/+2, −1/3 or −2/3"},
            "csir": {"title":"CSIR NET Mathematical Sciences","when":"June & December","duration":"3 hours","questions":"Part A:20, B:40, C:60","subjects":"Analysis, Algebra, Topology, Complex Analysis, ODE/PDE, Probability, Statistics","eligibility":"M.Sc. Mathematics","fee":"₹1,000 / ₹500 OBC / ₹250 SC/ST","admission":"JRF (Ph.D.) + Lectureship","cutoff":"JRF: top ~200; Lectureship: top ~6%","marking":"A:+2/−0.5, B:+3.5/−1, C:+5/0"},
        }
        details = EXAM_DATA.get(exam, EXAM_DATA["jam"])
        return jsonify({"exam": exam, "details": details}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = sanitize(data.get("name","Ramanujan"), 100)
        raw  = ask_simple(
            f"Detailed info about **{name}**: dates, nationality, contributions (LaTeX), theorems, influence.",
            temperature=0.3, max_tokens=2000
        )
        return jsonify({"name": name, "biography": raw or "Unavailable"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        theorem = sanitize(data.get("theorem","Pythagorean Theorem"), 300)
        proof   = ask_simple(
            f"Prove **{theorem}**.\n## Statement\n## Prerequisites\n## Proof\n## Alternative\n## Applications\nUse LaTeX.",
            temperature=0.1, max_tokens=3000
        )
        return jsonify({"proof": proof}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic","Calculus"), 100)
        raw   = ask_simple(
            f"5 maths project ideas for **{topic}**: title, objective, tools, difficulty, outcome, sample code.",
            temperature=0.35, max_tokens=3000
        )
        return jsonify({"projects": raw}), 200
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
    print("\n" + "═"*65)
    print("  🧮  MathSphere v11.1 — Complete Backend")
    print("═"*65)
    print(f"  {'✅' if GROQ_AVAILABLE   else '❌'} Groq API")
    print(f"  {'✅' if GEMINI_AVAILABLE else '❌'} Gemini API")
    print(f"  {'✅' if SYMPY_AVAILABLE  else '❌'} SymPy  |  {'✅' if NUMPY_AVAILABLE else '❌'} NumPy")
    print(f"  📦 DB: {stats['total_questions']} questions, {stats['total_papers']} papers")
    print(f"  🌐 App:   http://localhost:{os.getenv('PORT','5000')}/")
    print(f"  🔧 Admin: http://localhost:{os.getenv('PORT','5000')}/admin")
    print("═"*65 + "\n")


if __name__ == "__main__":
    print_startup()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE, use_reloader=False)