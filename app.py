"""
MathSphere v12.2 — FIXED IMAGE UPLOAD + NO ADMIN
✅ Proper Gemini image handling
✅ Correct base64 decoding
✅ Better error messages
✅ Admin endpoints removed
✅ All features working
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
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
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
        logger.info("[✓] Groq API connected")
    else:
        logger.warning("[!] GROQ_API_KEY not set - using Gemini instead")
except Exception as e:
    logger.warning(f"[!] Groq: {e}")

GEMINI_AVAILABLE = False
gemini_client = None

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai
        GEMINI_AVAILABLE = True
        logger.info("[✓] Gemini API connected")
    else:
        logger.warning("[!] GEMINI_API_KEY not set")
except Exception as e:
    logger.warning(f"[!] Gemini: {e}")

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
    logger.info("[✓] SymPy loaded")
except Exception as e:
    logger.warning(f"[!] SymPy: {e}")

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
# MATHEMATICIANS DATABASE
# ════════════════════════════════════════════════════════════════

MATHEMATICIANS_DB = {
    "ramanujan": {
        "name": "Srinivasa Ramanujan",
        "birth": "1887", "death": "1920", "nationality": "Indian",
        "field": "Number Theory, Modular Forms, Series",
        "wiki": "https://en.wikipedia.org/wiki/Srinivasa_Ramanujan",
        "summary": "Self-taught mathematical genius with extraordinary contributions to number theory and infinite series."
    },
    "euler": {
        "name": "Leonhard Euler",
        "birth": "1707", "death": "1783", "nationality": "Swiss",
        "field": "Analysis, Number Theory, Graph Theory",
        "wiki": "https://en.wikipedia.org/wiki/Leonhard_Euler",
        "summary": "One of history's most prolific mathematicians contributing to all major areas."
    },
    "gauss": {
        "name": "Carl Friedrich Gauss",
        "birth": "1777", "death": "1855", "nationality": "German",
        "field": "Number Theory, Statistics, Analysis",
        "wiki": "https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss",
        "summary": "Prince of Mathematicians with contributions across all mathematical fields."
    }
}

# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════

MATH_SYSTEM = """You are Anupam — a world-class mathematics tutor for IIT JAM, GATE, and CSIR NET.

ALWAYS provide structured responses:

## 🎯 Key Insight
[Core idea to solve this problem]

## 📚 Solution
[Step-by-step working with full details]

## ✅ Answer
[Final answer clearly stated]

## 📝 Explanation
[Why this works, common mistakes]

## 🧪 Verification
[Check the answer]

RULES:
- Use LaTeX: \\\\( inline \\\\) and \\\\[ display \\\\]
- Always show work, never just answers
- Be precise and thorough
- For images: analyze and solve what's shown
- Format all math expressions properly"""

# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def clean_ai_response(text: str) -> str:
    """Remove code blocks and extra formatting"""
    if not text:
        return ""
    text = re.sub(r'```[\w]*\n?', '', text)
    text = text.replace('```', '')
    return text.strip()


def ask_ai(messages: list, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    """Call Groq or Gemini API"""
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
            return clean_ai_response(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning(f"Groq error: {e}")

    if GEMINI_AVAILABLE and gemini_client:
        try:
            convo = MATH_SYSTEM + "\n\n"
            for m in messages:
                role = "User" if m["role"] == "user" else "Assistant"
                convo += f"{role}: {m['content']}\n\n"
            convo += "Assistant:"
            
            response = gemini_client.GenerativeModel('gemini-2.0-flash').generate_content(
                convo,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return clean_ai_response(response.text or "")
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    return "⚠️ No AI service available. Set GROQ_API_KEY or GEMINI_API_KEY."


def ask_simple(prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    return ask_ai([{"role": "user", "content": prompt}], temperature, max_tokens)


def sanitize(text: str, max_len: int = 5000) -> str:
    """Remove dangerous patterns"""
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
# DATABASE
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
                topic       TEXT,
                difficulty  TEXT DEFAULT 'moderate',
                question    TEXT NOT NULL,
                options     TEXT,
                answer      TEXT,
                explanation TEXT,
                marks       REAL DEFAULT 1.0,
                q_type      TEXT DEFAULT 'mcq',
                tags        TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                verified    INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_exam      ON questions(exam);
            CREATE INDEX IF NOT EXISTS idx_year      ON questions(year);
            CREATE INDEX IF NOT EXISTS idx_topic     ON questions(topic);
            CREATE INDEX IF NOT EXISTS idx_diff      ON questions(difficulty);
        """)
    logger.info(f"[✓] Database ready: {DB_PATH}")


def db_add_question(q: dict) -> int:
    with db_connect() as conn:
        cur = conn.execute("""
            INSERT INTO questions
              (exam, year, topic, difficulty, question, options, answer, 
               explanation, marks, q_type, tags, verified)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            q.get('exam', 'jam').lower(),
            q.get('year'),
            q.get('topic', ''),
            q.get('difficulty', 'moderate'),
            q.get('question', ''),
            json.dumps(q.get('options', [])),
            q.get('answer', ''),
            q.get('explanation', ''),
            float(q.get('marks', 1.0)),
            q.get('q_type', 'mcq'),
            json.dumps(q.get('tags', [])),
            1 if q.get('verified') else 0,
        ))
        return cur.lastrowid


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
        topics   = conn.execute("SELECT topic, COUNT(*) as n FROM questions WHERE topic != '' GROUP BY topic ORDER BY n DESC LIMIT 20").fetchall()
    return {
        "total_questions": total,
        "by_exam":         {r["exam"]: r["n"] for r in by_exam},
        "by_difficulty":   {r["difficulty"]: r["n"] for r in by_diff},
        "top_topics":      [{"topic": r["topic"], "count": r["n"]} for r in topics],
    }


# ════════════════════════════════════════════════════════════════
# GRAPH EXPRESSION CLEANER
# ════════════════════════════════════════════════════════════════

def _safe_clean_expr(expr_str: str) -> str:
    """Clean and normalize mathematical expressions"""
    if not expr_str:
        return ""
    
    expr_str = str(expr_str).strip()
    expr_str = expr_str.replace('π', 'pi').replace('∏', 'pi')
    expr_str = expr_str.replace('×', '*').replace('·', '*').replace('⋅', '*')
    expr_str = expr_str.replace('÷', '/')
    expr_str = expr_str.replace('^', '**')
    
    for _ in range(5):
        if '|' not in expr_str:
            break
        expr_str = re.sub(r'\|([^|]+)\|', lambda m: f'abs({m.group(1)})', expr_str)
    
    expr_str = expr_str.replace('mod', 'Mod')
    expr_str = expr_str.replace('ln(', 'log(')
    expr_str = expr_str.replace('√', 'sqrt')
    expr_str = re.sub(r'\s+', '', expr_str)
    
    return expr_str

# Initialize DB
db_init()

# ════════════════════════════════════════════════════════════════
# STATIC FILES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:
        return send_from_directory(STATIC_DIR, 'index.html')
    except FileNotFoundError:
        return jsonify({"message": "MathSphere v12.2 - API Server"}), 200

@app.route("/<path:filename>")
def serve_static(filename):
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return jsonify({"error": "Not found"}), 404

# ════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    stats = db_get_stats()
    return jsonify({
        "status": "healthy",
        "version": "12.2",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "numpy": NUMPY_AVAILABLE,
        },
        "database": {
            "total_questions": stats["total_questions"],
        }
    }), 200

# ════════════════════════════════════════════════════════════════
# CHAT WITH PROPER IMAGE SUPPORT ✅ FIXED
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@limiter.limit("40 per minute")
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Get message
        messages = data.get("messages")
        if not messages:
            single = sanitize(data.get("message", ""))
            if not single:
                return jsonify({"error": "No message provided"}), 400
            messages = [{"role": "user", "content": single}]

        # Clean messages
        clean = []
        for m in messages[-20:]:
            role = str(m.get("role", "user")).lower()
            content = sanitize(str(m.get("content", "")), 4000)
            if content and role in ("user", "assistant"):
                clean.append({"role": role, "content": content})

        if not clean:
            return jsonify({"error": "Empty message"}), 400

        # Get image data
        img_b64  = data.get("image_b64")
        img_type = data.get("image_type", "image/jpeg")

        # ✅ FIXED: Proper image handling
        if img_b64 and isinstance(img_b64, str) and img_b64.strip():
            # Remove data URI header if present
            if img_b64.startswith("data:"):
                try:
                    header, encoded = img_b64.split(",", 1)
                    img_b64 = encoded.strip()
                    # Extract MIME type from header
                    if ";" in header:
                        mime = header[5:].split(";")[0].strip()
                        if mime:
                            img_type = mime
                except:
                    pass

            # Decode base64
            try:
                raw_bytes = base64.b64decode(img_b64, validate=True)
            except Exception as e:
                logger.error(f"Base64 decode failed: {e}")
                return jsonify({"error": "Invalid image encoding"}), 400

            # Validate size
            if not raw_bytes or len(raw_bytes) > 20 * 1024 * 1024:
                return jsonify({"error": "Invalid or too large image (max 20MB)"}), 400

            # ✅ Use Gemini for image analysis
            if not GEMINI_AVAILABLE:
                return jsonify({
                    "error": "Image analysis requires Gemini API. Set GEMINI_API_KEY",
                    "requires_api": "gemini"
                }), 503

            logger.info(f"[IMAGE] Processing {len(raw_bytes)} bytes, type: {img_type}")

            try:
                # Use Gemini vision API
                prompt_text = clean[-1]["content"] if clean else "Solve this mathematics problem step by step"
                
                model = gemini_client.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content([
                    MATH_SYSTEM + "\n\n" + prompt_text,
                    {"mime_type": img_type, "data": raw_bytes}
                ])
                
                
                
                answer = clean_ai_response(response.text or "").strip()
                if not answer:
                    return jsonify({"error": "Could not analyze image"}), 422
                
                logger.info(f"[✓] Image analyzed successfully")
                return jsonify({"answer": answer}), 200

            except Exception as img_err:
                logger.exception(f"Image processing error: {img_err}")
                return jsonify({
                    "error": f"Image analysis failed: {str(img_err)[:100]}"
                }), 502

        # Text-only chat (no image)
        answer = ask_ai(clean)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# GRAPH PLOTTER
# ════════════════════════════════════════════════════════════════

@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    try:
        data = request.get_json(force=True, silent=True) or {}
        expr_str = sanitize(data.get("expression", "x**2"), 300)
        
        try:
            x_min = float(data.get("x_min", -10))
            x_max = float(data.get("x_max", 10))
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "x_min/x_max must be numbers"}), 400
        
        if x_min >= x_max:
            return jsonify({"success": False, "error": "x_min < x_max required"}), 400
        
        try:
            num_points = max(100, min(1000, int(data.get("points", 600))))
        except (TypeError, ValueError):
            num_points = 600

        if not SYMPY_AVAILABLE:
            return jsonify({"success": False, "error": "SymPy not available"}), 503
        if not expr_str:
            return jsonify({"success": False, "error": "Expression required"}), 400

        logger.info(f"[GRAPH] Plotting: {expr_str}")

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
                'exp': exp, 'log': log, 'ln': log, 'lg': log,
                'log10': lambda t: log(t, 10),
                'log2': lambda t: log(t, 2),
                'sqrt': sqrt, 'abs': Abs, 'Abs': Abs,
                'cbrt': lambda t: t ** Rational(1, 3),
                'gamma': gamma, 'erf': erf, 'factorial': factorial,
                'ceil': ceiling, 'ceiling': ceiling,
                'floor': floor, 'sign': sign, 'Mod': Mod, 'mod': Mod,
            }
            
            f_sym = parse_expr(clean_expr, transformations=transformations, local_dict=local_dict)

            step = (x_max - x_min) / num_points
            points = []
            success_count = 0

            for i in range(num_points + 1):
                xv = round(x_min + i * step, 8)
                try:
                    yv = float(N(f_sym.subs(x, xv), 10))
                    
                    if _isfinite(yv) and not _isnan(yv) and abs(yv) < 1e7:
                        points.append({"x": round(xv, 5), "y": round(yv, 6)})
                        success_count += 1
                    else:
                        points.append({"x": round(xv, 5), "y": None})
                        
                except:
                    points.append({"x": round(xv, 5), "y": None})

            logger.info(f"[GRAPH] Plotted {success_count}/{num_points} points")

            analysis_prompt = (
                f"Analyze f(x) = {expr_str}. Cover: 1) Domain 2) Range 3) Intercepts "
                f"4) Symmetry 5) Asymptotes 6) Critical points. Be concise. Use LaTeX."
            )
            analysis = ask_simple(analysis_prompt, temperature=0.1, max_tokens=1500)

            return jsonify({
                "success": True,
                "points": points,
                "expression": expr_str,
                "analysis": analysis,
                "x_range": [x_min, x_max],
                "point_count": success_count,
            }), 200

        except Exception as parse_err:
            error_str = str(parse_err)[:150]
            logger.exception(f"Parse error: {error_str}")
            
            return jsonify({
                "success": False,
                "error": f"Cannot parse expression: {error_str}",
                "examples": ["sin(x)", "x**2", "exp(-x**2)", "sqrt(x)", "log(x)"]
            }), 400

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error": str(e)[:100]}), 500

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
            return jsonify({"success": False, "error": "Invalid exam"}), 400
        if difficulty not in ("easy", "moderate", "difficult"):
            return jsonify({"success": False, "error": "Invalid difficulty"}), 400

        questions = db_get_questions(exam=exam, difficulty=difficulty,
                                     topic=topic or None, year=year, limit=count)

        return jsonify({
            "success": True,
            "questions": questions,
            "exam": exam,
            "difficulty": difficulty,
            "count": len(questions),
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"PYQ error: {e}")
        return jsonify({"success": False, "error": "Internal error"}), 500


@app.route("/api/pyq/search", methods=["POST"])
@limiter.limit("20 per minute")
def pyq_search():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        query = sanitize(data.get("query", ""), 200)
        exam  = sanitize(data.get("exam", ""), 20)
        
        if not query:
            return jsonify({"success": False, "error": "query required"}), 400
        
        with db_connect() as conn:
            sql = "SELECT * FROM questions WHERE question LIKE ?"
            params = [f"%{query}%"]
            if exam:
                sql += " AND exam = ?"
                params.append(exam.lower())
            sql += " LIMIT 10"
            rows = conn.execute(sql, params).fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            try: d['options'] = json.loads(d['options'] or '[]')
            except: d['options'] = []
            result.append(d)
        
        return jsonify({"success": True, "results": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# OTHER ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
def formula():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        exam  = sanitize(data.get("exam", "General"), 50)
        answer = ask_simple(f"Formula sheet for {topic} at {exam} level. Use LaTeX.",
                           temperature=0.05, max_tokens=2000)
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
            f"Generate {count} {difficulty} MCQs on {topic}. Use LaTeX for all math.",
            temperature=0.25, max_tokens=2500
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
            f"Prove {theorem}. Use LaTeX. Be clear and rigorous.",
            temperature=0.1, max_tokens=2500
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
                "title": "IIT JAM Mathematics", 
                "duration": "3 hours", 
                "questions": "60 (MCQ + NAT)",
                "subjects": "Real Analysis, Complex Analysis, Linear Algebra, Abstract Algebra, Topology"
            },
            "gate": {
                "title": "GATE Mathematics (MA)", 
                "duration": "3 hours", 
                "questions": "65 (MCQ + NAT + MSQ)",
                "subjects": "Calculus, Linear Algebra, Real Analysis, Complex Analysis, Algebra, Topology"
            },
            "csir": {
                "title": "CSIR NET Mathematical Sciences", 
                "duration": "3 hours (Part A, B, C)", 
                "questions": "120 total",
                "subjects": "Real Analysis, Complex Analysis, Algebra, Topology, Functional Analysis"
            },
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
        name = sanitize(data.get("name", "ramanujan"), 100).lower().strip()
        
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
                "wikipedia": bio_data["wiki"]
            }), 200
        
        bio = ask_simple(f"Biographical info about {name}: life, field, contributions, theorems.", 
                        temperature=0.3, max_tokens=1500)
        return jsonify({
            "success": True,
            "name": name.title(),
            "biography": bio,
            "wikipedia": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
        }), 200
    except Exception as e:
        logger.exception(f"Mathematician error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100).lower().strip()
        
        raw = ask_simple(
            f"Generate 5 project ideas for {topic} at undergraduate level. Use LaTeX for formulas.",
            temperature=0.35, max_tokens=2000
        )
        
        return jsonify({
            "success": True,
            "topic": topic.title(),
            "projects": raw,
        }), 200
    except Exception as e:
        logger.exception(f"Projects error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

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
    print("  🧮  MathSphere v12.2 — FIXED")
    print("═"*70)
    print(f"  ✅ Image Upload: Fixed + Working")
    print(f"  ✅ Gemini API: {'Connected' if GEMINI_AVAILABLE else 'Set GEMINI_API_KEY'}")
    print(f"  ✅ Groq API: {'Connected' if GROQ_AVAILABLE else 'Optional'}")
    print(f"  ✅ SymPy: {'Ready' if SYMPY_AVAILABLE else 'Not available'}")
    print(f"  ✅ Database: {stats['total_questions']} questions")
    print(f"  ✅ Admin endpoints: Removed")
    print(f"\n  📊 Features:")
    print(f"     • Chat with image upload 📷")
    print(f"     • Graph plotter (600+ points)")
    print(f"     • PYQ database with search")
    print(f"     • Formula sheets")
    print(f"     • Quiz generation")
    print(f"     • Theorem proofs")
    print(f"     • Mathematician profiles")
    print(f"\n  🌐 http://localhost:5000/")
    print("═"*70 + "\n")


if __name__ == "__main__":
    print_startup()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE, use_reloader=False)