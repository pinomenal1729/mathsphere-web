"""
MathSphere v10.0 — Production Backend
Compatible with: flask==3.0.3, groq==0.9.0, google-genai==0.8.0,
                 sympy==1.13.1, numpy==1.26.4, flask-limiter==3.5.0
"""

import os, sys, io, json, logging, re
from datetime import datetime
import base64
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

# ── Windows UTF-8 ──────────────────────────────────────────────
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Logging ────────────────────────────────────────────────────
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

# ── Flask App ──────────────────────────────────────────────────
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

CORS(app, resources={r"/api/*": {
    "origins": "*", "methods": ["GET","POST","OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

limiter = Limiter(app=app, key_func=get_remote_address,
                  default_limits=["300 per day","60 per hour"],
                  storage_uri="memory://")
cache = Cache(app)

# ── Groq (groq==0.9.0) ─────────────────────────────────────────
GROQ_AVAILABLE = False
groq_client    = None
try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client    = Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
        logger.info("[OK] Groq connected")
    else:
        logger.warning("[WARN] GROQ_API_KEY not set")
except Exception as e:
    logger.warning(f"[WARN] Groq: {e}")

# ── Gemini (google-genai==0.8.0) ───────────────────────────────
# IMPORTANT: google-genai==0.8.0 uses Client() not configure()
GEMINI_AVAILABLE = False
gemini_client    = None
genai_types      = None
try:
    from google import genai as _genai
    from google.genai import types as _types
    if GEMINI_API_KEY:
        gemini_client    = _genai.Client(api_key=GEMINI_API_KEY)
        genai_types      = _types
        GEMINI_AVAILABLE = True
        logger.info("[OK] Gemini (google-genai==0.8.0) connected")
    else:
        logger.warning("[WARN] GEMINI_API_KEY not set")
except Exception as e:
    logger.warning(f"[WARN] Gemini: {e}")

# ── SymPy (sympy==1.13.1) ──────────────────────────────────────
SYMPY_AVAILABLE = False
try:
    from sympy import Symbol, N, diff, solve, sin, cos, tan, exp, log, sqrt, pi, E
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)
    SYMPY_AVAILABLE = True
    logger.info("[OK] SymPy loaded")
except Exception as e:
    logger.warning(f"[WARN] SymPy: {e}")

# ── NumPy (numpy==1.26.4) ──────────────────────────────────────
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
# AI SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════

MATH_SYSTEM = """You are Anupam — a world-class mathematics tutor and researcher.

For EVERY mathematical problem or concept, provide:
1. 🧠 HOW TO THINK — mental approach and key insight
2. 📚 PREREQUISITES — topics the student must know first  
3. 🎯 WHY IT MATTERS — importance and real-world relevance
4. 📋 EXAM RELEVANCE — IIT JAM / GATE / CSIR NET section coverage
5. 📝 STEP-BY-STEP SOLUTION — complete rigorous working
6. ✅ VERIFICATION — how to check the answer

FORMATTING:
- ALL maths in LaTeX: inline → \\( ... \\)  display → \\[ ... \\]
- Use **bold** for key terms, ## for headings
- Be thorough, encouraging, pedagogically excellent"""


def ask_ai(messages: list, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    """Call Groq first, fallback to Gemini (google-genai==0.8.0)."""
    if not messages:
        return ""

    # ── Groq ──
    if GROQ_AVAILABLE and groq_client:
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"system","content":MATH_SYSTEM}] + messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"Groq error (falling back): {e}")

    # ── Gemini (google-genai==0.8.0 Client API) ──
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
            return resp.text or ""
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    return "⚠️ No AI service available. Check your API keys in .env"


def ask_simple(prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    return ask_ai([{"role":"user","content":prompt}], temperature, max_tokens)


def sanitize(text: str, max_len: int = 5000) -> str:
    if not text: return ""
    text = str(text).strip()[:max_len]
    for pat in [r'__import__', r'\beval\b', r'\bexec\b', r'subprocess', r'\bos\.']:
        if re.search(pat, text, re.IGNORECASE):
            logger.warning(f"Blocked input: {text[:60]}")
            return ""
    return text
def parse_int_field(value, default: int, min_value: int, max_value: int, field_name: str = "value") -> int:
    """Parse and clamp integer request fields with clear 400-style messages."""
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = int(value)
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
    try:    return send_from_directory(STATIC_DIR, 'index.html')
    except: return jsonify({"error":"index.html not found in static/"}), 404

@app.route("/<path:filename>")
def serve_static(filename):
    try:    return send_from_directory(STATIC_DIR, filename)
    except: return jsonify({"error":"File not found"}), 404

# ════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy", "version": "10.0",
        "services": {"groq": GROQ_AVAILABLE, "gemini": GEMINI_AVAILABLE,
                     "sympy": SYMPY_AVAILABLE, "numpy": NUMPY_AVAILABLE},
        "timestamp": datetime.now().isoformat()
    })

# ════════════════════════════════════════════════════════════════
# CHAT  /api/chat
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@limiter.limit("40 per minute")
def chat():
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Accept both { messages:[...] } and { message:"..." }
        messages = data.get("messages")
        if not messages:
            single = sanitize(data.get("message",""))
            if not single:
                return jsonify({"error":"No message provided"}), 400
            messages = [{"role":"user","content":single}]

        clean = []
        for m in messages[-20:]:
            role    = str(m.get("role","user"))
            content = sanitize(str(m.get("content","")), 4000)
            if content:
                clean.append({"role":role, "content":content})

        if not clean:
            return jsonify({"error":"Empty message"}), 400

        # Image handling (google-genai==0.8.0)
        img_b64  = data.get("image_b64")
        img_type = data.get("image_type", "image/jpeg")

        if img_b64:
            try:
                allowed_mimes = {"image/jpeg", "image/png", "image/webp"}
                if img_type not in allowed_mimes:
                    return jsonify({"error": "Unsupported image type. Use JPEG, PNG, or WEBP."}), 400

                raw_bytes = base64.b64decode(img_b64, validate=True)
                if not raw_bytes:
                    return jsonify({"error": "Invalid image payload."}), 400
                if len(raw_bytes) > 10 * 1024 * 1024:
                    return jsonify({"error": "Image too large. Max size is 10MB."}), 400

                if not (GEMINI_AVAILABLE and gemini_client and genai_types):
                    return jsonify({"error": "Image solving currently unavailable. Configure GEMINI_API_KEY."}), 503
                prompt_text = clean[-1]["content"] if clean else "Solve this mathematics problem."
                resp = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        MATH_SYSTEM,
                        prompt_text,
                        genai_types.Part.from_bytes(data=raw_bytes, mime_type=img_type)
                    ]
                )
                return jsonify({"answer": resp.text or ""}), 200
            except (ValueError, base64.binascii.Error):
                return jsonify({"error": "Invalid image payload."}), 400
            except Exception as e:
                logger.warning(f"Image processing failed: {e} — using text only")

        return jsonify({"answer": ask_ai(clean)}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error."}), 500

# ════════════════════════════════════════════════════════════════
# GRAPH  /api/graph
# ════════════════════════════════════════════════════════════════

@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        expr_str = sanitize(data.get("expression","x**2"), 300)
        gtype    = data.get("type","2d")
        points   = []

        if SYMPY_AVAILABLE and expr_str:
            try:
                clean_expr = expr_str.replace('^','**').replace('π','pi')
                tf = standard_transformations + (
                    implicit_multiplication_application, convert_xor)
                x = Symbol('x')
                local = {'x':x,'pi':pi,'E':E,'sin':sin,'cos':cos,
                         'tan':tan,'exp':exp,'log':log,'sqrt':sqrt}
                f = parse_expr(clean_expr, transformations=tf, local_dict=local)

                x_min, x_max, n = -10, 10, 400
                step = (x_max - x_min) / n
                for i in range(n + 1):
                    xv = x_min + i * step
                    try:
                        yv = float(N(f.subs(x, xv), 8))
                        if _isfinite(yv) and not _isnan(yv) and abs(yv) < 1e6:
                            points.append({"x": round(xv,4), "y": round(yv,4)})
                        else:
                            points.append({"x": round(xv,4), "y": None})
                    except Exception:
                        points.append({"x": round(xv,4), "y": None})
            except Exception as e:
                logger.warning(f"SymPy graph error: {e}")

        analysis = ask_simple(
            f"Analyse f(x) = {expr_str}: domain, range, intercepts, symmetry, asymptotes, "
            f"critical points \\( f'(x)=0 \\), inflection points, behaviour as x→±∞, "
            f"and which JAM/GATE/CSIR topics it illustrates. Use LaTeX throughout.",
            max_tokens=1500
        )

        return jsonify({"sympy":SYMPY_AVAILABLE, "points":points,
                        "expression":expr_str, "type":gtype,
                        "analysis":analysis, "success":True}), 200

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error":"Internal server error.", "success":False}), 500

# ════════════════════════════════════════════════════════════════
# FORMULA SHEET  /api/formula   → { answer }
# ════════════════════════════════════════════════════════════════

@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
def formula():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic","Calculus"), 100)
        exam  = sanitize(data.get("exam","General"),   50)

        prompt = f"""Generate a COMPLETE formula sheet for **{topic}** for **{exam}** exam.

## Section 1: [Category]
**1.** \\[ formula \\]  *Use case & {exam} weight*
**2.** \\[ formula \\]  *Use case*
...

## Section 2: [Next Category]
...

## ⚡ Quick Tricks for {exam}
1. Trick one
2. Trick two
...5 tricks total

RULES: Min 50 numbered formulas. Every formula in LaTeX. Cover ALL subtopics of {topic}."""

        answer = ask_simple(prompt, temperature=0.05, max_tokens=4000)
        return jsonify({"answer": answer or f"Could not generate sheet for {topic}."}), 200
    except Exception as e:
        logger.error(f"Formula error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# COMPETITION  /api/competition/problems   → { problems }
# ════════════════════════════════════════════════════════════════

@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("10 per minute")
def competition_problems():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        category = sanitize(data.get("category","IMO"), 50)
        count    = parse_int_field(data.get("count", 10), default=10, min_value=1, max_value=30, field_name="count")

        prompt = f"""Generate {count} {category}-style competition problems with full pedagogical solutions.

For EACH use this format:
---
## Problem N — [Title]
**Difficulty:** Easy/Medium/Hard | **Topic:** [area]

**Statement:** \\[ ... \\]

**🧠 How to Think:** [key insight]
**📚 Prerequisites:** [topics needed]
**🎯 Exam Relevance:** [JAM/GATE/CSIR section]

**Solution:**
Step 1: \\[ ... \\]
Step 2: \\[ ... \\]

**Answer:** \\[ \\boxed{{answer}} \\]
---
Generate all {count} for {category}."""

        problems = ask_simple(prompt, temperature=0.3, max_tokens=4000)
        return jsonify({"problems": problems or "Could not generate."}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Competition error: {e}")
        return jsonify({"error":"Internal server error."}), 500

# ════════════════════════════════════════════════════════════════
# QUIZ  /api/quiz/generate   → { questions }
# ════════════════════════════════════════════════════════════════

@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic","Calculus"), 100)
        count = parse_int_field(data.get("count", 10), default=10, min_value=1, max_value=30, field_name="count")

        prompt = f"""Generate {count} exam-style MCQs on **{topic}**.

Format:
---
## Question N
\\[ Question in LaTeX \\]

**(A)** \\( opt_A \\)  **(B)** \\( opt_B \\)  **(C)** \\( opt_C \\)  **(D)** \\( opt_D \\)

**✅ Answer:** (X)
**🧠 Insight:** [key idea]
**📝 Solution:**
Step 1: \\[ ... \\]
Final: \\[ \\boxed{{ans}} \\]
**❌ Why others wrong:** brief reason each
**📋 Exam:** JAM/GATE/CSIR topic section
---
Generate all {count} for {topic}."""

        questions = ask_simple(prompt, temperature=0.2, max_tokens=4000)
        return jsonify({"questions": questions or "Could not generate quiz."}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Quiz error: {e}")
        return jsonify({"error":"Internal server error."}), 500

# ════════════════════════════════════════════════════════════════
# RESEARCH  /api/research   → { response }
# ════════════════════════════════════════════════════════════════

@app.route("/api/research", methods=["POST"])
@limiter.limit("15 per minute")
def research_hub():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        rtype = sanitize(data.get("type","topic"),  50)
        query = sanitize(data.get("query",""),      500)
        if not query:
            return jsonify({"error":"Query required"}), 400

        prompts = {
            "literature": f"Write a literature review for '{query}': key papers, authors, major results in LaTeX, open problems, citations.",
            "topic":      f"Deep-dive on '{query}': definition in LaTeX, history, key theorems with proof sketches, examples, connections, study roadmap.",
            "methods":    f"All problem-solving methods for '{query}': for each method — name, when to use, worked example in LaTeX, pros/cons.",
            "career":     f"Career guidance for '{query}' in India: academic paths (JAM/GATE/CSIR), industry roles, skills, salaries, top institutions.",
            "resources":  f"Best study resources for '{query}': textbooks with authors, NPTEL/MIT OCW courses, YouTube channels, notes, past papers.",
        }
        response = ask_simple(prompts.get(rtype, prompts["topic"]), temperature=0.2, max_tokens=2500)
        return jsonify({"response": response or "No results."}), 200
    except Exception as e:
        logger.error(f"Research error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# EXAM INFO  /api/exam/info   → { exam, details }
# ════════════════════════════════════════════════════════════════

@app.route("/api/exam/info", methods=["POST"])
@limiter.limit("20 per minute")
def exam_info():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        exam     = sanitize(data.get("exam","jam"), 20).lower().strip()
        req_type = sanitize(data.get("type","info"), 20)

        EXAM_DATA = {
            "jam": {
                "title":       "IIT JAM — Joint Admission Test for M.Sc.",
                "when":        "February every year (CBT online)",
                "duration":    "3 hours",
                "questions":   "60 — MCQ (10×1 + 30×2) + MSQ + NAT",
                "subjects":    "Real Analysis, Linear Algebra, Calculus, ODE/PDE, Abstract Algebra, Complex Analysis, Numerical Methods, Statistics",
                "eligibility": "Bachelor's with Mathematics (55% General / 50% SC/ST)",
                "fee":         "₹1,800 (General/OBC) | ₹900 (SC/ST/PwD/Female)",
                "admission":   "M.Sc. & Joint M.Sc.-Ph.D. at IITs and IISc",
                "cutoff":      "50–70 / 100 (varies by IIT and category)"
            },
            "gate": {
                "title":       "GATE Mathematics (MA)",
                "when":        "January–February every year (CBT online)",
                "duration":    "3 hours | 100 marks",
                "questions":   "65 — MCQ + MSQ + NAT",
                "subjects":    "Calculus, Linear Algebra, Real Analysis, Complex Analysis, ODE, PDE, Abstract Algebra, Functional Analysis, Numerical Analysis, Probability & Statistics",
                "eligibility": "B.E./B.Tech/B.Sc./M.Sc. or equivalent (final year eligible)",
                "fee":         "₹1,800 (General/OBC) | ₹900 (SC/ST/PwD/Female)",
                "admission":   "M.Tech/PhD at IITs/NITs; PSU recruitment; valid 3 years",
                "cutoff":      "40–60 / 100"
            },
            "csir": {
                "title":       "CSIR NET Mathematical Sciences",
                "when":        "June and December every year",
                "duration":    "3 hours | 200 marks",
                "questions":   "Part A: 20 (Aptitude) | Part B: 40 | Part C: 60",
                "subjects":    "Full UG+PG Mathematics: Analysis, Algebra, Topology, Complex Analysis, Functional Analysis, ODE/PDE, Differential Geometry, Mechanics, Statistics",
                "eligibility": "M.Sc. Mathematics or appearing (Age ≤ 28 for JRF)",
                "fee":         "₹1,000 (General) | ₹500 (OBC-NCL) | ₹250 (SC/ST/PwD)",
                "admission":   "JRF: ₹37,000/month | Lectureship eligibility",
                "cutoff":      "Top 6% qualify | JRF ≈ top 200 candidates"
            }
        }

        details = EXAM_DATA.get(exam, EXAM_DATA["jam"])

        if req_type == "syllabus":
            syl = ask_simple(
                f"Give complete detailed syllabus for {details['title']}: every topic, subtopic, "
                f"key theorems in LaTeX, approximate exam weightage. Group by subject area.",
                max_tokens=2500
            )
            details = dict(details)
            details["subjects"] = syl or details["subjects"]

        return jsonify({"exam": exam, "details": details}), 200
    except Exception as e:
        logger.error(f"Exam info error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# PYQ LOAD  /api/pyq/load   → { success, questions, exam, count }
# THIS ENDPOINT WAS MISSING — NOW ADDED
# ════════════════════════════════════════════════════════════════

@app.route("/api/pyq/load", methods=["POST"])
@limiter.limit("10 per minute")
def pyq_load():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        exam  = sanitize(data.get("exam","jam"), 20).lower().strip()
        count = parse_int_field(data.get("count", 10), default=10, min_value=1, max_value=50, field_name="count")

        exam_names = {
            "jam":  "IIT JAM Mathematics",
            "gate": "GATE Mathematics (MA)",
            "csir": "CSIR NET Mathematical Sciences"
        }
        exam_name = exam_names.get(exam, "IIT JAM Mathematics")

        prompt = f"""Generate {count} realistic PYQ-style questions for **{exam_name}**.

For EACH use EXACTLY this format:

**Question N** (Year: 20XX | Marks: X | Type: MCQ/NAT)
\\[ Question statement in LaTeX \\]

**(A)** \\( opt_A \\)
**(B)** \\( opt_B \\)
**(C)** \\( opt_C \\)
**(D)** \\( opt_D \\)

**Topic:** [Subtopic]
**Difficulty:** Easy/Medium/Hard

---

Generate all {count} questions for {exam_name}."""

        raw = ask_simple(prompt, temperature=0.2, max_tokens=4000)

        if not raw or len(raw) < 100:
            return jsonify({"success":False, "error":"Could not generate PYQs."}), 500

        return jsonify({"success":True, "questions":raw, "exam":exam, "count":count}), 200
    except ValueError as ve:
        return jsonify({"success":False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"PYQ error: {e}")
        return jsonify({"success":False, "error":"Internal server error."}), 500

# ════════════════════════════════════════════════════════════════
# MATHEMATICIAN  /api/mathematician   → JSON object
# ════════════════════════════════════════════════════════════════

@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = sanitize(data.get("name",""), 100)

        subject = f"the mathematician {name}" if name else "a randomly chosen lesser-known influential mathematician"

        prompt = f"""Return information about {subject} as JSON.
RETURN ONLY VALID JSON — no markdown, no extra text.

{{
  "name": "Full Name",
  "period": "YYYY–YYYY",
  "country": "Country",
  "fields": ["Field1","Field2"],
  "biography": "2-3 sentence biography.",
  "famous_quote": "Quote or empty string.",
  "major_contributions": ["contrib with LaTeX e.g. \\\\( formula \\\\)","contrib2","contrib3"],
  "impact_today": "Modern impact.",
  "learning_resources": ["Book: Title by Author","Online: resource"]
}}"""

        raw = ask_simple(prompt, temperature=0.3, max_tokens=1500)

        try:
            clean = re.sub(r'```(?:json)?|```','', raw or '').strip()
            s, e  = clean.find('{'), clean.rfind('}')+1
            if s >= 0 and e > s:
                return jsonify(json.loads(clean[s:e])), 200
        except Exception as je:
            logger.warning(f"Mathematician JSON parse failed: {je}")

        return jsonify({
            "name": name or "Mathematician", "period":"","country":"","fields":[],
            "biography": raw or "Information unavailable.",
            "famous_quote":"","major_contributions":[],"impact_today":"","learning_resources":[]
        }), 200
    except Exception as e:
        logger.error(f"Mathematician error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# THEOREM  /api/theorem/prove   → { proof }
# ════════════════════════════════════════════════════════════════

@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        theorem = sanitize(data.get("theorem","Pythagorean Theorem"), 300)

        prompt = f"""Prove: **{theorem}**

## Theorem Statement
\\[ formal statement \\]

## 🧠 How to Think
[key insight and proof strategy]

## 📚 Prerequisites
[topics student must know]

## 🎯 Exam Relevance
[JAM / GATE / CSIR NET sections that test this]

## 📝 Formal Proof
**Step 1 — [description]:** \\[ ... \\]
**Step 2 — [description]:** \\[ ... \\]
[continue] \\[ \\blacksquare \\]

## 💡 Alternative Proof Methods
[1-2 other approaches briefly]

## 🔗 Related Theorems
[2-3 connected results in LaTeX]

## 📐 Applications
[3-4 applications with mini-examples]

Use LaTeX for ALL maths."""

        proof = ask_simple(prompt, temperature=0.1, max_tokens=3500)
        return jsonify({"proof": proof or f"Could not prove {theorem}."}), 200
    except Exception as e:
        logger.error(f"Theorem error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# PROJECTS  /api/projects/generate   → { projects }
# ════════════════════════════════════════════════════════════════

@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic","Machine Learning"), 100)

        prompt = f"""Generate 5 maths projects for: **{topic}**
Return ONLY a valid JSON array — no markdown.

[
  {{
    "title":"Project Title",
    "difficulty":"Beginner",
    "description":"2-3 sentences emphasising mathematics.",
    "math_concepts":["Concept1","Concept2","Concept3"],
    "step_by_step":["Step 1: ...","Step 2: ...","Step 3: ...","Step 4: ...","Step 5: ..."],
    "code_snippet":"# Python\\nimport numpy as np\\n# code here",
    "resources":["Reference 1","Reference 2"]
  }},
  {{"...second project (Intermediate)..."}},
  {{"...third project (Advanced)..."}},
  {{"...fourth project..."}},
  {{"...fifth project..."}}
]"""

        raw = ask_simple(prompt, temperature=0.3, max_tokens=3500)

        try:
            clean = re.sub(r'```(?:json)?|```','', raw or '').strip()
            s, e  = clean.find('['), clean.rfind(']')+1
            if s >= 0 and e > s:
                return jsonify({"projects": json.loads(clean[s:e])}), 200
        except Exception as je:
            logger.warning(f"Projects JSON failed: {je}")

        return jsonify({"projects": raw or "Could not generate projects."}), 200
    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({"error":str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error":"Rate limit exceeded. Please wait."}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error":"Not found."}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500: {e}")
    return jsonify({"error":"Internal server error."}), 500

# ════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════

def print_startup():
    print("\n" + "═"*62)
    print("  🧮  MathSphere v10.0 — Backend Ready")
    print("═"*62)
    print(f"  Groq  (groq==0.9.0):            {'✅ Connected' if GROQ_AVAILABLE   else '❌ Set GROQ_API_KEY'}")
    print(f"  Gemini (google-genai==0.8.0):   {'✅ Connected' if GEMINI_AVAILABLE else '❌ Set GEMINI_API_KEY'}")
    print(f"  SymPy (sympy==1.13.1):          {'✅ Loaded'    if SYMPY_AVAILABLE  else '❌ Failed'}")
    print(f"  NumPy (numpy==1.26.4):          {'✅ Loaded'    if NUMPY_AVAILABLE  else '❌ Failed'}")
    print(f"  Static: {STATIC_DIR}")
    print("═"*62 + "\n")


if __name__ == "__main__":
    print_startup()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port,
            debug=(FLASK_ENV=="development"), use_reloader=False)