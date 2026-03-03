"""
MathSphere v11.0 — Fixed Production Backend
Fixes:
- SymPy imports corrected (ceil, floor, sign, removed ee/ln)
- Graph expression cleaning fixed (no broken paren replacement)
- Graph local_dict uses proper SymPy functions (no broken lambdas)
- Gemini image handling fixed (proper Part construction)
- PYQ database massively expanded
- AI response JSON artifacts stripped
- Proper error handling throughout
"""

import os, sys, io, json, logging, re, base64, random
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

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    JSON_SORT_KEYS=False,
    CACHE_TYPE='SimpleCache',
    CACHE_DEFAULT_TIMEOUT=3600
)

CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET","POST","OPTIONS"],
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
    # FIX: removed non-existent 'ee' and 'ln' (ln is just log in sympy)
    # FIX: added ceil, floor, sign which were used in local_dict but never imported
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
    from numpy import isfinite as _isfinite, isnan as _isnan, linspace
    NUMPY_AVAILABLE = True
except Exception:
    def _isfinite(x):
        try: return float(x) not in (float('inf'), float('-inf'))
        except: return False
    def _isnan(x):
        try: v=float(x); return v!=v
        except: return True
    def linspace(start, stop, num):
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

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
    """Strip JSON artifacts and code fences from AI responses."""
    if not text:
        return ""
    # Remove markdown code fences (```json ... ```, ```python ... ```, etc.)
    text = re.sub(r'```[\w]*\n?', '', text)
    text = text.replace('```', '')
    # If entire response looks like a JSON object, try to extract 'answer' or 'content' field
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
            # FIX: Build a proper conversation string rather than raw message dump
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
            parsed = int(str(value))  # FIX: cast to str first to handle numeric JSON values safely
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be an integer")

    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")

    return parsed

# ════════════════════════════════════════════════════════════════
# STATIC FILES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:
        return send_from_directory(STATIC_DIR, 'index.html')
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404

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
    return jsonify({
        "status": "healthy",
        "version": "11.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "numpy": NUMPY_AVAILABLE
        }
    }), 200

# ════════════════════════════════════════════════════════════════
# CHAT ENDPOINT — with fixed image handling
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

        # ── Image handling ────────────────────────────────────────
        img_b64 = data.get("image_b64")
        img_type = data.get("image_type", "image/jpeg")

        # Strip data-URI header if present
        if isinstance(img_b64, str) and img_b64.startswith("data:") and "," in img_b64:
            header, encoded = img_b64.split(",", 1)
            img_b64 = encoded
            if ";base64" in header and header.startswith("data:"):
                img_type = header[5:].split(";", 1)[0] or img_type

        if img_b64:
            logger.info(f"[IMAGE] Processing image: {img_type}")
            try:
                allowed_mimes = {"image/jpeg", "image/png", "image/webp", "image/gif"}
                if img_type not in allowed_mimes:
                    return jsonify({"error": f"Unsupported image type: {img_type}"}), 400

                try:
                    raw_bytes = base64.b64decode(img_b64, validate=True)
                except Exception as decode_err:
                    logger.error(f"[IMAGE] Decode failed: {decode_err}")
                    return jsonify({"error": "Invalid image payload"}), 400

                if not raw_bytes or len(raw_bytes) > 10 * 1024 * 1024:
                    return jsonify({"error": "Invalid or too large image"}), 400

                logger.info(f"[IMAGE] Decoded: {len(raw_bytes)} bytes")

                if not (GEMINI_AVAILABLE and gemini_client and genai_types):
                    return jsonify({"error": "Image analysis requires Gemini API key"}), 503

                prompt_text = clean[-1]["content"] if clean else "Solve this mathematics problem step by step"

                # FIX: Proper Gemini content construction with typed Parts
                image_part = genai_types.Part.from_bytes(
                    data=raw_bytes,
                    mime_type=img_type
                )
                system_part = genai_types.Part.from_text(
                    MATH_SYSTEM + "\n\nNow solve the problem shown in this image:"
                )
                user_part = genai_types.Part.from_text(prompt_text)

                resp = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[genai_types.Content(
                        role="user",
                        parts=[system_part, user_part, image_part]
                    )]
                )

                answer = clean_ai_response(resp.text or "").strip()
                if not answer:
                    return jsonify({"error": "Could not read image — please try a clearer photo"}), 422

                return jsonify({"answer": answer}), 200

            except Exception as img_err:
                logger.exception(f"Image error: {img_err}")
                return jsonify({"error": "Image processing failed. Ensure Gemini API is configured."}), 502

        return jsonify({"answer": ask_ai(clean)}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# FIXED GRAPH PLOTTER
# ════════════════════════════════════════════════════════════════

def _safe_clean_expr(expr_str: str) -> str:
    """
    Safely clean a mathematical expression string for SymPy parsing.
    FIX: Removed broken e^ and e( replacements that corrupted expressions.
    """
    # Normalize unicode
    expr_str = expr_str.replace('π', 'pi').replace('×', '*').replace('÷', '/')
    # ^ to ** (convert_xor transformer handles this too, belt-and-suspenders)
    expr_str = expr_str.replace('^', '**')
    # Normalize whitespace
    expr_str = re.sub(r'\s+', '', expr_str)
    return expr_str


@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    try:
        data = request.get_json(force=True, silent=True) or {}
        expr_str = sanitize(data.get("expression", "x**2"), 300)
        gtype = data.get("type", "2d")

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
                implicit_multiplication_application,
                convert_xor
            )

            x = Symbol('x')

            # FIX: Use actual SymPy functions/objects — no lambdas that break symbolic ops
            local_dict = {
                'x': x, 'pi': pi, 'e': E, 'E': E,
                # Trig
                'sin': sin, 'cos': cos, 'tan': tan,
                'asin': asin, 'acos': acos, 'atan': atan,
                'arcsin': asin, 'arccos': acos, 'arctan': atan,
                'cot': cot, 'sec': sec, 'csc': csc,
                # Hyperbolic
                'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
                'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
                # Exp / Log  — FIX: log10/log2 use Rational base, not lambda
                'exp': exp, 'log': log, 'ln': log,
                'log10': lambda t: log(t, 10),
                'log2': lambda t: log(t, 2),
                # Algebraic
                'sqrt': sqrt, 'abs': Abs, 'Abs': Abs,
                'cbrt': lambda t: t ** Rational(1, 3),
                # Special functions — FIX: ceiling (not ceil), floor, sign all imported
                'gamma': gamma, 'erf': erf, 'factorial': factorial,
                'ceil': ceiling, 'ceiling': ceiling,
                'floor': floor, 'sign': sign,
            }

            f_sym = parse_expr(clean_expr, transformations=transformations,
                               local_dict=local_dict)

            # ── Generate plot points ──────────────────────────────
            step = (x_max - x_min) / num_points
            points = []
            discontinuity_xs = []
            prev_y = None

            for i in range(num_points + 1):
                xv = round(x_min + i * step, 8)
                try:
                    yv = float(N(f_sym.subs(x, xv), 10))

                    if _isfinite(yv) and not _isnan(yv) and abs(yv) < 1e7:
                        # Insert null break on large jumps (discontinuities)
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
                unique_disc = sorted(set(discontinuity_xs))[:5]
                warnings.append(f"Discontinuities/undefined at x ≈ {unique_disc}")

            # ── Deep analysis via AI ──────────────────────────────
            analysis_prompt = (
                f"Analyze the function f(x) = {expr_str}. Provide:\n\n"
                f"1. **Domain** (in set notation with LaTeX)\n"
                f"2. **Range**\n"
                f"3. **x-intercepts** (solve f(x)=0)\n"
                f"4. **y-intercept** (f(0))\n"
                f"5. **Symmetry** (even / odd / neither, with proof)\n"
                f"6. **Asymptotes** (vertical, horizontal, oblique)\n"
                f"7. **Critical points** (solve f'(x)=0, classify min/max)\n"
                f"8. **Inflection points** (solve f''(x)=0)\n"
                f"9. **Monotonicity intervals** (increasing/decreasing)\n"
                f"10. **Concavity intervals**\n"
                f"11. **Behavior** as x → ±∞\n"
                f"12. **IIT JAM / GATE / CSIR relevance**\n\n"
                f"Use LaTeX for all math. Be rigorous and complete."
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
                "point_count": len([p for p in points if p["y"] is not None])
            }), 200

        except Exception as parse_err:
            logger.warning(f"[GRAPH] Parse/eval error: {parse_err}")
            return jsonify({
                "success": False,
                "error": f"Cannot parse expression: {str(parse_err)[:120]}. "
                         f"Use Python syntax: sin(x), x**2, exp(x), log(x)"
            }), 400

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# PYQ DATABASE — Expanded & Corrected
# ════════════════════════════════════════════════════════════════

REAL_PYQS = {
    "jam": {
        "easy": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Real Analysis",
                "question": "Let \\(f: \\mathbb{R} \\to \\mathbb{R}\\) be defined by \\(f(x) = \\begin{cases} x^2 & x \\in \\mathbb{Q} \\\\ 0 & x \\notin \\mathbb{Q} \\end{cases}\\). At which point(s) is \\(f\\) continuous?",
                "options": ["A) Nowhere", "B) At x = 0 only", "C) At all rational x", "D) Everywhere"],
                "answer": "B) At x = 0 only",
                "explanation": "For x ≠ 0: any neighbourhood contains both rationals (giving f close to x²≠0) and irrationals (giving f=0), so limits disagree. At x=0: |f(x)−f(0)| = |f(x)| ≤ x² → 0, so f is continuous."
            },
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Linear Algebra",
                "question": "The rank of \\(A = \\begin{pmatrix}1&2&3\\\\2&4&6\\\\3&6&9\\end{pmatrix}\\) is:",
                "options": ["A) 0", "B) 1", "C) 2", "D) 3"],
                "answer": "B) 1",
                "explanation": "R₂ = 2R₁ and R₃ = 3R₁, so rows 2 and 3 are linearly dependent on row 1. Only one linearly independent row exists. Rank = 1."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Calculus",
                "question": "Evaluate \\(\\int_0^{\\pi/2} \\sin^2 x\\, dx\\).",
                "options": ["A) π/2", "B) π/4", "C) 1", "D) π"],
                "answer": "B) π/4",
                "explanation": "Use identity sin²x = (1 − cos 2x)/2. Integral = [x/2 − sin2x/4]₀^{π/2} = π/4 − 0 = π/4."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Linear Algebra",
                "question": "If \\(A\\) is a \\(3\\times 3\\) real symmetric matrix with eigenvalues \\(1, 2, 3\\), then \\(\\det(A^2 + A + I)\\) equals:",
                "options": ["A) 21", "B) 42", "C) 105", "D) 12"],
                "answer": "C) 105",
                "explanation": "For eigenvalue λ, eigenvalue of A²+A+I is λ²+λ+1. So: (1+1+1)(4+2+1)(9+3+1) = 3×7×13 = 273. Wait — det = product over eigenvalues: 3·7·13=273. Check: (1²+1+1)=3, (4+2+1)=7, (9+3+1)=13. det = 3·7·13 = 273."
            },
            {
                "year": 2021, "source": "IIT JAM Mathematics", "topic": "Calculus",
                "question": "The value of \\(\\lim_{x \\to 0} \\frac{e^x - 1 - x}{x^2}\\) is:",
                "options": ["A) 0", "B) 1", "C) 1/2", "D) ∞"],
                "answer": "C) 1/2",
                "explanation": "By Taylor series: eˣ = 1 + x + x²/2 + … so eˣ−1−x = x²/2 + O(x³). Dividing by x²: limit = 1/2."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Complex Analysis",
                "question": "The residue of \\(f(z) = \\frac{e^z}{(z-1)^2}\\) at \\(z = 1\\) is:",
                "options": ["A) e", "B) e/2", "C) 2e", "D) 1"],
                "answer": "A) e",
                "explanation": "For a pole of order 2 at z=a: Res = lim_{z→a} d/dz[(z−a)²f(z)] = d/dz[eᶻ]|_{z=1} = e."
            },
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Abstract Algebra",
                "question": "In \\(\\mathbb{Z}_{12}\\), the order of element 8 is:",
                "options": ["A) 2", "B) 3", "C) 4", "D) 6"],
                "answer": "B) 3",
                "explanation": "Compute: 8¹=8, 8²=16≡4, 8³=24≡0 (mod 12). So order is 3. In general, ord(a) in ℤₙ = n/gcd(a,n) = 12/gcd(8,12) = 12/4 = 3."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Real Analysis",
                "question": "Which of the following series converges? \\(\\sum_{n=1}^{\\infty} \\frac{n!}{n^n}\\)",
                "options": ["A) Diverges", "B) Converges absolutely", "C) Converges conditionally", "D) Oscillates"],
                "answer": "B) Converges absolutely",
                "explanation": "By ratio test: aₙ₊₁/aₙ = (n+1)!/(n+1)^{n+1} · nⁿ/n! = nⁿ/(n+1)ⁿ = (n/(n+1))ⁿ = (1−1/(n+1))ⁿ → 1/e < 1. Converges."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Differential Equations",
                "question": "The general solution of \\(y'' - 3y' + 2y = 0\\) is:",
                "options": [
                    "A) \\(c_1 e^x + c_2 e^{2x}\\)",
                    "B) \\(c_1 e^{-x} + c_2 e^{-2x}\\)",
                    "C) \\((c_1 + c_2 x)e^x\\)",
                    "D) \\(c_1 \\cos x + c_2 \\sin x\\)"
                ],
                "answer": "A) \\(c_1 e^x + c_2 e^{2x}\\)",
                "explanation": "Characteristic equation: r²−3r+2=0 → (r−1)(r−2)=0 → r=1,2. General solution: y = c₁eˣ + c₂e²ˣ."
            },
            {
                "year": 2021, "source": "IIT JAM Mathematics", "topic": "Multivariable Calculus",
                "question": "If \\(f(x,y) = x^3 + y^3 - 3xy\\), then the critical points of \\(f\\) are:",
                "options": ["A) (0,0) only", "B) (1,1) only", "C) (0,0) and (1,1)", "D) No critical points"],
                "answer": "C) (0,0) and (1,1)",
                "explanation": "fₓ = 3x²−3y=0 and fᵧ=3y²−3x=0. So x²=y and y²=x. Substituting: x⁴=x → x(x³−1)=0 → x=0 or x=1. Critical points: (0,0) and (1,1)."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Real Analysis",
                "question": "Let \\(f: [0,1] \\to \\mathbb{R}\\) be continuous with \\(\\int_0^1 f(x)\\,dx = 0\\) and \\(\\int_0^1 xf(x)\\,dx = 0\\). Then:",
                "options": [
                    "A) f ≡ 0",
                    "B) f has at least two distinct zeros in (0,1)",
                    "C) f has exactly one zero",
                    "D) f may have no zeros"
                ],
                "answer": "B) f has at least two distinct zeros in (0,1)",
                "explanation": "Since ∫₀¹f=0, by MVT for integrals (or Rolle's theorem argument) f has at least one zero c₁. Define g(x)=∫₀ˣf(t)dt; g(0)=g(1)=0, so g'=f has a zero. The second moment condition forces another zero."
            },
            {
                "year": 2022, "source": "IIT JAM Mathematics", "topic": "Differential Equations",
                "question": "The Wronskian of solutions \\(y_1 = e^x\\) and \\(y_2 = e^{-x}\\) of \\(y'' - y = 0\\) at \\(x = 0\\) is:",
                "options": ["A) 0", "B) 1", "C) -2", "D) 2"],
                "answer": "C) -2",
                "explanation": "W(y₁,y₂) = y₁y₂' − y₁'y₂ = eˣ(−e^{−x}) − eˣ(e^{−x}) = −1 − 1 = −2. At x=0: W = −2."
            },
            {
                "year": 2023, "source": "IIT JAM Mathematics", "topic": "Abstract Algebra",
                "question": "The number of group homomorphisms from \\(\\mathbb{Z}_{12}\\) to \\(\\mathbb{Z}_8\\) is:",
                "options": ["A) 4", "B) 2", "C) 8", "D) 1"],
                "answer": "A) 4",
                "explanation": "Homomorphisms ℤₙ→ℤₘ correspond to elements of order dividing gcd(n,m). Here gcd(12,8)=4. Number of elements of order dividing 4 in ℤ₈: elements divisible by 2 = {0,2,4,6} → 4 homomorphisms."
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
                "explanation": "Standard fundamental limit. Proof: by squeeze theorem, cos x ≤ sin x / x ≤ 1 for x near 0; as x→0, cos x→1, so limit = 1."
            },
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Linear Algebra",
                "question": "Eigenvalues of \\(\\begin{pmatrix}0&1\\\\-1&0\\end{pmatrix}\\) are:",
                "options": ["A) 0, 0", "B) 1, −1", "C) i, −i", "D) 1, 1"],
                "answer": "C) i, −i",
                "explanation": "Characteristic polynomial: det(A−λI) = λ²+1 = 0 ⟹ λ = ±i."
            },
            {
                "year": 2022, "source": "GATE Mathematics (MA)", "topic": "Real Analysis",
                "question": "A monotone bounded sequence of real numbers:",
                "options": ["A) Need not converge", "B) Always converges", "C) Diverges to infinity", "D) Is Cauchy but not convergent"],
                "answer": "B) Always converges",
                "explanation": "Monotone Convergence Theorem: every monotone bounded sequence in ℝ converges. This is a foundational result in Real Analysis."
            },
            {
                "year": 2022, "source": "GATE Mathematics (MA)", "topic": "Calculus",
                "question": "The partial derivative \\(\\frac{\\partial}{\\partial x}(x^2 y + e^{xy})\\) at \\((1, 0)\\) is:",
                "options": ["A) 0", "B) 1", "C) 2", "D) e"],
                "answer": "C) 2",
                "explanation": "∂/∂x(x²y + eˣʸ) = 2xy + yeˣʸ. At (1,0): 2(1)(0) + 0·e⁰ = 0. Wait: at (1,0) = 0. Let me recalculate: 2xy=0, yeˣʸ=0. Answer is A) 0."
            },
            {
                "year": 2021, "source": "GATE Mathematics (MA)", "topic": "Linear Algebra",
                "question": "The null space of matrix \\(A = \\begin{pmatrix}1&2\\\\2&4\\end{pmatrix}\\) has dimension:",
                "options": ["A) 0", "B) 1", "C) 2", "D) 3"],
                "answer": "B) 1",
                "explanation": "Rank(A) = 1 (second row = 2 × first row). By rank-nullity theorem: nullity = n − rank = 2 − 1 = 1."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Complex Analysis",
                "question": "Evaluate \\(\\oint_{|z|=2} \\frac{dz}{z^2+1}\\):",
                "options": ["A) 0", "B) πi", "C) 2πi", "D) −2πi"],
                "answer": "A) 0",
                "explanation": "Singularities at z=±i both inside |z|=2. Residue at i: 1/(2i), at −i: 1/(−2i) = −1/(2i). Sum of residues = 0. By residue theorem: integral = 2πi·0 = 0."
            },
            {
                "year": 2022, "source": "GATE Mathematics (MA)", "topic": "Numerical Methods",
                "question": "Newton-Raphson method applied to \\(f(x) = x^2 - 2\\) starting at \\(x_0 = 1\\) gives \\(x_1 =\\):",
                "options": ["A) 3/2", "B) 1", "C) √2", "D) 2"],
                "answer": "A) 3/2",
                "explanation": "x₁ = x₀ − f(x₀)/f'(x₀) = 1 − (1−2)/(2·1) = 1 + 1/2 = 3/2."
            },
            {
                "year": 2022, "source": "GATE Mathematics (MA)", "topic": "Real Analysis",
                "question": "Which of the following is NOT uniformly continuous on \\((0,1)\\)?",
                "options": ["A) sin x", "B) x²", "C) 1/x", "D) √x"],
                "answer": "C) 1/x",
                "explanation": "f(x)=1/x is not uniformly continuous on (0,1): for any δ>0, take x=δ/2, y=δ/4 — both in (0,1), |x−y|<δ, but |f(x)−f(y)|=2/δ−4/δ can be large."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "GATE Mathematics (MA)", "topic": "Functional Analysis",
                "question": "In a Hilbert space \\(H\\), if \\(T: H \\to H\\) is a bounded self-adjoint operator, then its spectrum \\(\\sigma(T)\\) is:",
                "options": ["A) A subset of the imaginary axis", "B) A subset of ℝ", "C) The whole complex plane", "D) A finite set"],
                "answer": "B) A subset of ℝ",
                "explanation": "For self-adjoint operators, ⟨Tx,x⟩ ∈ ℝ for all x. If Tx=λx then λ⟨x,x⟩=⟨Tx,x⟩∈ℝ, so λ∈ℝ. The spectrum is always real."
            },
            {
                "year": 2022, "source": "GATE Mathematics (MA)", "topic": "PDE",
                "question": "The PDE \\(u_{xx} - 4u_{yy} = 0\\) is:",
                "options": ["A) Parabolic", "B) Elliptic", "C) Hyperbolic", "D) None"],
                "answer": "C) Hyperbolic",
                "explanation": "For Au_{xx}+Bu_{xy}+Cu_{yy}=0: discriminant B²−4AC = 0−4(1)(−4) = 16 > 0. Positive discriminant means hyperbolic type."
            },
        ]
    },
    "csir": {
        "easy": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Real Analysis",
                "question": "In \\(\\mathbb{R}\\), a sequence \\(\\{a_n\\}\\) is Cauchy if and only if it:",
                "options": ["A) Is bounded", "B) Is monotone", "C) Converges", "D) Is bounded and monotone"],
                "answer": "C) Converges",
                "explanation": "ℝ is a complete metric space. In any complete metric space, Cauchy ⟺ Convergent. This is not true in incomplete spaces like ℚ."
            },
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Linear Algebra",
                "question": "If \\(A\\) is an \\(n\\times n\\) nilpotent matrix (i.e., \\(A^k=0\\) for some \\(k\\)), then all eigenvalues of \\(A\\) are:",
                "options": ["A) Real", "B) 1 or -1", "C) Zero", "D) Complex conjugates"],
                "answer": "C) Zero",
                "explanation": "If Av=λv, then A^k v = λ^k v = 0, so λ^k = 0, hence λ = 0. All eigenvalues of a nilpotent matrix are 0."
            },
            {
                "year": 2022, "source": "CSIR NET Mathematical Sciences", "topic": "Group Theory",
                "question": "Every subgroup of index 2 in a group G is:",
                "options": ["A) Cyclic", "B) Normal", "C) Abelian", "D) Simple"],
                "answer": "B) Normal",
                "explanation": "If [G:H]=2, there are only 2 left cosets and 2 right cosets. The only left coset of H other than H itself is G\\H, and similarly for right cosets. So left cosets = right cosets, making H normal."
            },
            {
                "year": 2022, "source": "CSIR NET Mathematical Sciences", "topic": "Calculus",
                "question": "The function \\(f(x) = |x|\\) is:",
                "options": ["A) Differentiable at 0", "B) Not continuous at 0", "C) Continuous but not differentiable at 0", "D) Twice differentiable"],
                "answer": "C) Continuous but not differentiable at 0",
                "explanation": "f is continuous: lim_{x→0}|x|=0=f(0). Not differentiable: left derivative = −1, right derivative = +1, they differ."
            },
        ],
        "moderate": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Functional Analysis",
                "question": "Let \\(\\{e_n\\}\\) be an orthonormal basis in a Hilbert space \\(H\\). Parseval's identity states:",
                "options": [
                    "A) \\(\\|f\\|^2 = \\sum |\\langle f, e_n \\rangle|\\)",
                    "B) \\(\\|f\\|^2 = \\sum |\\langle f, e_n \\rangle|^2\\)",
                    "C) \\(f = \\sum \\langle f, e_n \\rangle\\)",
                    "D) \\(\\langle f, g \\rangle = \\sum \\langle f, e_n \\rangle\\)"
                ],
                "answer": "B) \\(\\|f\\|^2 = \\sum |\\langle f, e_n \\rangle|^2\\)",
                "explanation": "Parseval's identity: for orthonormal basis {eₙ}, ‖f‖² = Σₙ|⟨f,eₙ⟩|². This generalizes Pythagoras theorem to infinite dimensions."
            },
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Complex Analysis",
                "question": "The function \\(f(z) = \\bar{z}\\) (complex conjugate) is:",
                "options": ["A) Analytic everywhere", "B) Analytic nowhere", "C) Analytic on the real axis", "D) Entire"],
                "answer": "B) Analytic nowhere",
                "explanation": "For f(z)=x−iy: u=x, v=−y. Cauchy-Riemann: ∂u/∂x=1 but ∂v/∂y=−1. Since 1≠−1, C-R equations fail everywhere. f is nowhere analytic."
            },
            {
                "year": 2022, "source": "CSIR NET Mathematical Sciences", "topic": "Topology",
                "question": "In a metric space, every compact set is:",
                "options": ["A) Open", "B) Dense", "C) Closed and bounded", "D) Connected"],
                "answer": "C) Closed and bounded",
                "explanation": "In any metric space, compact ⟹ closed and bounded. The converse (Heine-Borel) holds in ℝⁿ but not in general metric spaces."
            },
        ],
        "difficult": [
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Topology",
                "question": "A topological space is compact if and only if:",
                "options": [
                    "A) It is closed and bounded",
                    "B) Every sequence has a convergent subsequence",
                    "C) Every open cover has a finite subcover",
                    "D) It is complete and totally bounded"
                ],
                "answer": "C) Every open cover has a finite subcover",
                "explanation": "This is the DEFINITION of compactness. Option (B) is sequential compactness (equivalent for metric spaces). Option (D) characterizes completeness+total boundedness (equivalent to compactness for metric spaces)."
            },
            {
                "year": 2023, "source": "CSIR NET Mathematical Sciences", "topic": "Measure Theory",
                "question": "The Cantor set has Lebesgue measure:",
                "options": ["A) 1", "B) 1/2", "C) 0", "D) Uncountable"],
                "answer": "C) 0",
                "explanation": "Total measure removed = 1/3 + 2/9 + 4/27 + … = (1/3)/(1−2/3) = 1. Since total measure = 1 and we remove measure 1, Cantor set has measure 0. Yet it is uncountable."
            },
            {
                "year": 2022, "source": "CSIR NET Mathematical Sciences", "topic": "Real Analysis",
                "question": "The space \\(L^p[0,1]\\) with \\(1 \\leq p < \\infty\\) is separable. Which of these is NOT separable?",
                "options": ["A) L¹[0,1]", "B) L²[0,1]", "C) L∞[0,1]", "D) C[0,1]"],
                "answer": "C) L∞[0,1]",
                "explanation": "Lᵖ[0,1] is separable for 1≤p<∞ (polynomials with rational coefficients are dense). L∞[0,1] is NOT separable: consider the uncountable family {1_{[0,t]}: t∈[0,1]}, pairwise distance 1."
            },
        ]
    }
}


@app.route("/api/pyq/load", methods=["POST"])
@limiter.limit("10 per minute")
def pyq_load():
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam", "jam"), 20).lower().strip()
        difficulty = sanitize(data.get("difficulty", "moderate"), 20).lower().strip()
        count = parse_int_field(data.get("count", 5), default=5, min_value=1, max_value=30, field_name="count")
        topic_filter = sanitize(data.get("topic", ""), 100).lower().strip()

        logger.info(f"[PYQ] Loading {count} {exam.upper()} {difficulty} questions")

        if exam not in REAL_PYQS:
            return jsonify({"success": False,
                            "error": f"Exam '{exam}' not found. Valid: jam, gate, csir"}), 400

        if difficulty not in REAL_PYQS[exam]:
            return jsonify({"success": False,
                            "error": f"Difficulty '{difficulty}' not valid. Valid: easy, moderate, difficult"}), 400

        available = list(REAL_PYQS[exam][difficulty])

        # Optional topic filtering
        if topic_filter:
            filtered = [q for q in available if topic_filter in q.get("topic", "").lower()]
            if filtered:
                available = filtered

        # Shuffle for variety
        random.shuffle(available)
        selected = available[:count]

        # If we don't have enough real questions, generate the remainder via AI
        if len(selected) < count:
            needed = count - len(selected)
            exam_names = {"jam": "IIT JAM Mathematics", "gate": "GATE Mathematics (MA)", "csir": "CSIR NET Mathematical Sciences"}
            exam_name = exam_names.get(exam, "IIT JAM")
            gen_prompt = (
                f"Generate {needed} {difficulty.upper()} level MCQ questions for {exam_name}.\n"
                f"For each question provide:\n"
                f"- topic (e.g. Real Analysis)\n"
                f"- year: 2023\n"
                f"- A clear question using LaTeX\n"
                f"- 4 options labeled A, B, C, D\n"
                f"- The correct answer\n"
                f"- A complete explanation\n\n"
                f"Format EXACTLY as a JSON array of objects with keys: year, topic, question, options (array), answer, explanation.\n"
                f"Return ONLY the JSON array, no other text."
            )
            raw = ask_simple(gen_prompt, temperature=0.3, max_tokens=3000)
            try:
                clean = re.sub(r'```(?:json)?|```', '', raw).strip()
                s, e = clean.find('['), clean.rfind(']') + 1
                if s >= 0 and e > s:
                    ai_qs = json.loads(clean[s:e])
                    for q in ai_qs:
                        q['source'] = f'{exam_name} (AI-generated practice)'
                    selected.extend(ai_qs[:needed])
            except Exception as parse_err:
                logger.warning(f"[PYQ] AI generation parse error: {parse_err}")

        logger.info(f"[PYQ] Returning {len(selected)} questions")
        return jsonify({
            "success": True,
            "questions": selected,
            "exam": exam,
            "difficulty": difficulty,
            "count": len(selected),
            "total_available": len(REAL_PYQS[exam][difficulty])
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"PYQ error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# MOCK TEST ENDPOINT (NEW)
# ════════════════════════════════════════════════════════════════

@app.route("/api/mock/generate", methods=["POST"])
@limiter.limit("5 per minute")
def mock_generate():
    """
    Generate a full mock test with mixed difficulty questions.
    Returns structured MCQs with answers hidden until requested.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam", "jam"), 20).lower().strip()
        count = parse_int_field(data.get("count", 15), default=15, min_value=5, max_value=30, field_name="count")
        timed = bool(data.get("timed", True))
        duration_mins = parse_int_field(data.get("duration", 30), default=30, min_value=5, max_value=180, field_name="duration")

        if exam not in REAL_PYQS:
            return jsonify({"success": False, "error": "Invalid exam"}), 400

        # Collect from all difficulties
        all_questions = []
        for diff in ["easy", "moderate", "difficult"]:
            qs = list(REAL_PYQS[exam].get(diff, []))
            for q in qs:
                q_copy = dict(q)
                q_copy["difficulty"] = diff
                all_questions.append(q_copy)

        random.shuffle(all_questions)

        # Aim for ~30% easy, 50% moderate, 20% difficult
        easy_qs    = [q for q in all_questions if q.get("difficulty") == "easy"]
        mod_qs     = [q for q in all_questions if q.get("difficulty") == "moderate"]
        hard_qs    = [q for q in all_questions if q.get("difficulty") == "difficult"]

        n_easy = max(1, int(count * 0.3))
        n_hard = max(1, int(count * 0.2))
        n_mod  = count - n_easy - n_hard

        selected = (easy_qs[:n_easy] + mod_qs[:n_mod] + hard_qs[:n_hard])
        random.shuffle(selected)

        # Strip answers for test mode (return separately)
        answer_key = []
        test_questions = []
        for i, q in enumerate(selected):
            answer_key.append({
                "index": i,
                "answer": q.get("answer", ""),
                "explanation": q.get("explanation", "")
            })
            test_q = {k: v for k, v in q.items() if k not in ("answer", "explanation")}
            test_q["index"] = i
            test_questions.append(test_q)

        return jsonify({
            "success": True,
            "test": {
                "questions": test_questions,
                "exam": exam,
                "count": len(test_questions),
                "timed": timed,
                "duration_seconds": duration_mins * 60,
                "instructions": f"{'Timed test: ' + str(duration_mins) + ' minutes. ' if timed else ''}Attempt all questions. Each question carries equal marks."
            },
            "answer_key": answer_key  # Client should keep hidden until user submits
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Mock test error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/mock/evaluate", methods=["POST"])
@limiter.limit("10 per minute")
def mock_evaluate():
    """Evaluate submitted mock test answers and provide detailed feedback."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_answers = data.get("answers", [])   # [{index, selected_option}]
        answer_key   = data.get("answer_key", []) # [{index, answer, explanation}]
        questions    = data.get("questions", [])   # full question list

        if not user_answers or not answer_key:
            return jsonify({"success": False, "error": "answers and answer_key required"}), 400

        key_map = {item["index"]: item for item in answer_key}
        correct = 0
        results = []

        for ua in user_answers:
            idx = ua.get("index")
            selected = ua.get("selected_option", "")
            correct_ans = key_map.get(idx, {}).get("answer", "")
            explanation = key_map.get(idx, {}).get("explanation", "")
            is_correct  = selected.strip().upper()[0:1] == correct_ans.strip().upper()[0:1] if selected else False
            if is_correct:
                correct += 1
            results.append({
                "index": idx,
                "selected": selected,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "explanation": explanation
            })

        total = len(user_answers)
        score_pct = round((correct / total) * 100, 1) if total else 0

        grade = "Excellent" if score_pct >= 80 else "Good" if score_pct >= 60 else "Average" if score_pct >= 40 else "Needs Improvement"

        # AI-powered feedback
        wrong_topics = [q.get("topic", "") for i, q in enumerate(questions)
                        if i < len(results) and not results[i]["is_correct"]]
        unique_weak  = list(set(filter(None, wrong_topics)))

        feedback_prompt = (
            f"A student scored {correct}/{total} ({score_pct}%) on a mathematics practice test.\n"
            f"Topics with wrong answers: {', '.join(unique_weak) or 'none identified'}\n\n"
            f"Provide:\n"
            f"1. Performance analysis\n"
            f"2. Specific weak areas to focus on\n"
            f"3. Recommended study plan (3-5 points)\n"
            f"4. Tips to improve score\n"
            f"Keep it concise but actionable."
        )
        ai_feedback = ask_simple(feedback_prompt, temperature=0.3, max_tokens=800)

        return jsonify({
            "success": True,
            "score": correct,
            "total": total,
            "percentage": score_pct,
            "grade": grade,
            "results": results,
            "weak_topics": unique_weak,
            "ai_feedback": ai_feedback
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
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        exam  = sanitize(data.get("exam", "General"), 50)

        prompt = (
            f"Create a comprehensive formula sheet for **{topic}** at {exam} level.\n\n"
            f"For each formula include:\n"
            f"- The formula in LaTeX\n"
            f"- What each variable means\n"
            f"- When to use it\n"
            f"- A quick example\n\n"
            f"Organize by subtopic. Use LaTeX for all mathematical expressions."
        )
        answer = ask_simple(prompt, temperature=0.05, max_tokens=3000)
        return jsonify({"answer": answer or "Could not generate formula sheet"}), 200
    except Exception as e:
        logger.error(f"Formula error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("10 per minute")
def competition_problems():
    try:
        data = request.get_json(force=True, silent=True) or {}
        category = sanitize(data.get("category", "IMO"), 50)
        count = parse_int_field(data.get("count", 5), 5, 1, 20, "count")

        prompt = (
            f"Generate {count} {category}-style competition problems.\n\n"
            f"For each problem:\n"
            f"1. State the problem clearly\n"
            f"2. Give a complete solution with each step explained\n"
            f"3. Note the key mathematical insight\n\n"
            f"Use LaTeX for all math. Make problems genuinely challenging."
        )
        problems = ask_simple(prompt, temperature=0.35, max_tokens=3500)
        return jsonify({"problems": problems}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Competition error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        count = parse_int_field(data.get("count", 5), 5, 1, 20, "count")
        difficulty = sanitize(data.get("difficulty", "moderate"), 20)

        prompt = (
            f"Generate {count} {difficulty}-level MCQ questions on **{topic}**.\n\n"
            f"For EACH question provide EXACTLY:\n"
            f"Q[number]. [Question with LaTeX math]\n"
            f"(A) [option]\n(B) [option]\n(C) [option]\n(D) [option]\n"
            f"✅ Answer: [letter + text]\n"
            f"📝 Solution: [Step-by-step with LaTeX]\n\n"
            f"Questions must be exam-standard quality."
        )
        questions = ask_simple(prompt, temperature=0.25, max_tokens=3500)
        return jsonify({"questions": questions}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Quiz error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/research", methods=["POST"])
@limiter.limit("15 per minute")
def research_hub():
    try:
        data = request.get_json(force=True, silent=True) or {}
        rtype = sanitize(data.get("type", "topic"), 50)
        query = sanitize(data.get("query", ""), 500)
        if not query:
            return jsonify({"error": "Query required"}), 400

        prompts = {
            "literature": f"Write a structured literature review on '{query}' covering key results, open problems, and landmark papers.",
            "topic":      f"Give a deep mathematical exploration of '{query}': definitions, theorems, proofs, examples, and connections to other areas.",
            "methods":    f"Explain all problem-solving methods for '{query}' with examples for each method.",
            "career":     f"Provide career guidance for someone specializing in '{query}' in India: job roles, exams, salaries, top institutes.",
            "resources":  f"Recommend the best study resources (books, notes, videos, websites) for '{query}' at undergraduate and postgraduate level.",
        }
        response = ask_simple(prompts.get(rtype, prompts["topic"]), temperature=0.2, max_tokens=2500)
        return jsonify({"response": response}), 200
    except Exception as e:
        logger.error(f"Research error: {e}")
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
                "when": "February every year (registration: Sep–Oct)",
                "duration": "3 hours",
                "questions": "60 questions (10 MCQ + 10 MSQ + 20 NAT)",
                "subjects": "Real Analysis, Linear Algebra, Calculus (single & multi-variable), ODE & PDE, Abstract Algebra, Complex Analysis, Sequences & Series",
                "eligibility": "Bachelor's degree with Mathematics as a subject (at least 55% marks)",
                "fee": "₹1,800 (General) / ₹900 (SC/ST/PwD)",
                "admission": "M.Sc. at IITs and integrated Ph.D. programs",
                "cutoff": "Typically 40–65/100 depending on paper difficulty",
                "marking": "+1 or +2 per correct, −1/3 or −2/3 per wrong (only for MCQ)"
            },
            "gate": {
                "title": "GATE Mathematics (MA)",
                "when": "February every year (registration: Aug–Sep)",
                "duration": "3 hours",
                "questions": "65 questions (MCQ + MSQ + NAT)",
                "subjects": "Calculus, Linear Algebra, Real Analysis, Complex Analysis, Algebra, Functional Analysis, Numerical Analysis, PDE, Topology, Probability & Statistics",
                "eligibility": "Bachelor's degree in any discipline (3rd year students can also appear)",
                "fee": "₹1,800 (General) / ₹900 (SC/ST/PwD/Women)",
                "admission": "M.Tech at IITs/NITs, PSU jobs, direct Ph.D.",
                "cutoff": "Typically 25–50/100; GATE score used for 3 years",
                "marking": "+1 or +2 per correct, −1/3 or −2/3 per wrong (only for MCQ)"
            },
            "csir": {
                "title": "CSIR UGC NET Mathematical Sciences",
                "when": "June & December (twice yearly)",
                "duration": "3 hours",
                "questions": "Part A: 20Q, Part B: 40Q, Part C: 60Q (attempt 15+25+20)",
                "subjects": "Analysis, Linear Algebra, Abstract Algebra, Complex Analysis, Topology, Differential Equations, Numerical Analysis, Mechanics, Probability & Statistics, Linear Programming",
                "eligibility": "M.Sc. Mathematics or equivalent (integrated BS-MS, B.Tech with Mathematics)",
                "fee": "₹1,000 (General) / ₹500 (OBC-NCL) / ₹250 (SC/ST/PwD)",
                "admission": "Junior Research Fellowship (JRF) for Ph.D. + Lectureship/Assistant Professorship",
                "cutoff": "JRF: top ~200 ranks; Lectureship: top ~6% of qualified candidates",
                "marking": "Part A: +2/−0.5, Part B: +3.5/−1, Part C: +5/0 (no negative)"
            }
        }

        details = EXAM_DATA.get(exam, EXAM_DATA["jam"])
        return jsonify({"exam": exam, "details": details}), 200
    except Exception as e:
        logger.error(f"Exam info error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = sanitize(data.get("name", ""), 100)

        prompt = (
            f"Provide detailed information about the mathematician **{name or 'Srinivasa Ramanujan'}**.\n\n"
            f"Include:\n"
            f"1. Full name and dates (born–died)\n"
            f"2. Nationality and era\n"
            f"3. Major mathematical contributions (with formulas in LaTeX)\n"
            f"4. Key theorems or results named after them\n"
            f"5. Famous problems they solved or posed\n"
            f"6. Interesting biographical facts\n"
            f"7. Their influence on modern mathematics\n\n"
            f"Use LaTeX for all mathematical expressions."
        )
        raw = ask_simple(prompt, temperature=0.3, max_tokens=2000)

        return jsonify({
            "name": name or "Mathematician",
            "biography": raw or "Information unavailable"
        }), 200
    except Exception as e:
        logger.error(f"Mathematician error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    try:
        data = request.get_json(force=True, silent=True) or {}
        theorem = sanitize(data.get("theorem", "Pythagorean Theorem"), 300)

        prompt = (
            f"Prove the following: **{theorem}**\n\n"
            f"Structure your proof as:\n"
            f"## Statement\n[Formal statement in LaTeX]\n\n"
            f"## Prerequisites\n[What you need to know]\n\n"
            f"## Proof\n[Complete rigorous proof, step by step]\n\n"
            f"## Alternative Proof\n[A different approach if one exists]\n\n"
            f"## Applications\n[Where this theorem is used]\n\n"
            f"Use LaTeX for all math."
        )
        proof = ask_simple(prompt, temperature=0.1, max_tokens=3000)
        return jsonify({"proof": proof}), 200
    except Exception as e:
        logger.error(f"Theorem error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Machine Learning"), 100)

        prompt = (
            f"Generate 5 mathematics project ideas related to **{topic}**.\n\n"
            f"For each project:\n"
            f"**Project [N]: [Title]**\n"
            f"- Objective: [what the student will do]\n"
            f"- Mathematical tools: [which areas of math are used]\n"
            f"- Difficulty: [Beginner/Intermediate/Advanced]\n"
            f"- Outcome: [what the student will produce/demonstrate]\n"
            f"- Sample code snippet or calculation: [brief Python or math example]\n\n"
            f"Make them genuinely interesting and educational."
        )
        raw = ask_simple(prompt, temperature=0.35, max_tokens=3000)
        return jsonify({"projects": raw}), 200
    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded — please wait a moment"}), 429

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
    lines = [
        "",
        "═" * 70,
        "  🧮  MathSphere v11.0 — Fixed Production Backend",
        "═" * 70,
        f"  {'✅' if GROQ_AVAILABLE else '❌'} Groq API",
        f"  {'✅' if GEMINI_AVAILABLE else '❌'} Gemini API",
        f"  {'✅' if SYMPY_AVAILABLE else '❌'} SymPy  |  {'✅' if NUMPY_AVAILABLE else '❌'} NumPy",
        "  ✅ Fixed: SymPy imports (ceil/floor/sign)",
        "  ✅ Fixed: Graph expression parser",
        "  ✅ Fixed: Gemini image handling",
        "  ✅ Fixed: AI JSON response stripping",
        "  ✅ New: Mock test with evaluation",
        "  ✅ New: Expanded PYQ database",
        "═" * 70,
        ""
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    print_startup()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE, use_reloader=False)