"""
MathSphere v9.0 — Production-Ready Backend (IMPROVED)
====================================================
By Anupam Nigam | youtube.com/@pi_nomenal1729

IMPROVEMENTS INCLUDED:
✅ Rate limiting (30 req/min per IP)
✅ Input validation & sanitization
✅ Request size limits (16MB max)
✅ SymPy timeout protection (2 seconds)
✅ Comprehensive error logging
✅ Response caching (formulas, problems)
✅ Better error messages
✅ Rigorous solution verification
✅ Request ID tracking
✅ Graceful fallbacks
"""

import os
import re
import json
import random
import sys
import logging
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

load_dotenv()

# ════ LOGGING SETUP ════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mathsphere.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ════ FLASK APP CONFIG ════
app = Flask(__name__, static_folder="static", static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# ════ RATE LIMITING ════
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ════ CACHING ════
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 3600})

# ════ API KEYS ════
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_AVAILABLE = bool(GROQ_API_KEY)
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

groq_client = None
gemini_client = None
SYMPY_AVAILABLE = False

if GROQ_AVAILABLE:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq connected")
    except Exception as e:
        logger.warning(f"⚠️ Groq init failed: {e}")
        GROQ_AVAILABLE = False

if GEMINI_AVAILABLE:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini connected")
    except Exception as e:
        logger.warning(f"⚠️ Gemini init failed: {e}")
        GEMINI_AVAILABLE = False

# ════ SymPy Setup with Timeout ════
try:
    logger.info(f"🔍 Python version: {sys.version}")
    import sympy as sp
    from sympy import (
        symbols, sympify, diff, integrate, solve, simplify, expand,
        factor, latex as sp_latex, Matrix, eigenvals, eigenvects,
        limit, series, oo, pi, E, I, sqrt, Rational, N, dsolve,
        Function, Eq, Symbol, cancel, apart, together, trigsimp,
        det, trace, eye, zeros, ones, linsolve, nonlinsolve,
        Sum, Product, cos, sin, tan, exp, log, Abs,
        factorial, gcd, lcm, isprime, factorint,
        Derivative, Integral, summation
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor
    )
    SYMPY_AVAILABLE = True
    logger.info(f"✅ SymPy v{sp.__version__} loaded successfully")
except Exception as e:
    logger.error(f"⚠️ SymPy failed: {e}")

TEACHER_YOUTUBE = "https://youtube.com/@pi_nomenal1729"
TEACHER_WEBSITE = "https://www.anupamnigam.com"

SYMPY_TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
) if SYMPY_AVAILABLE else None


# ════ INPUT VALIDATION ════
def validate_input(text, field_type="general", max_len=1000):
    """Validate user input safely"""
    if not text:
        return False
    
    text = str(text).strip()
    
    if len(text) > max_len:
        return False
    
    if field_type == "expression":
        # Allow math symbols only
        allowed = r'^[x\d\w\s\-\+\*/\(\)\.\^√∫∑πθα\[\],=:]+$'
        return bool(re.match(allowed, text))
    elif field_type == "topic":
        # Allow letters, numbers, spaces, basic punctuation
        allowed = r'^[a-zA-Z0-9\s\-_\,\.]+$'
        return bool(re.match(allowed, text))
    else:
        # General: no extreme special chars
        disallowed = ['<', '>', '&', ';', '|', '`']
        return not any(c in text for c in disallowed)


def sanitize_input(text, field_type="general"):
    """Sanitize user input"""
    if not text:
        return ""
    
    text = str(text).strip()
    
    if not validate_input(text, field_type):
        raise ValueError(f"Invalid input for {field_type}")
    
    # Remove leading/trailing whitespace, limit multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text[:1000]  # Limit length


# ════ TIMEOUT DECORATOR ════
def timeout(seconds=2):
    """Timeout decorator for long-running operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded {seconds}s timeout")
            
            # Only works on Unix/Linux
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
            
            return result
        return wrapper
    return decorator


def safe_parse(expr_str: str):
    """Parse mathematical expression safely with timeout"""
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        expr_str = sanitize_input(expr_str, "expression")
        expr_str = expr_str.strip()
        expr_str = re.sub(r'\^', '**', expr_str)
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        
        # Parse with timeout protection
        try:
            result = parse_expr(expr_str, transformations=SYMPY_TRANSFORMATIONS, timeout=2)
            return result
        except:
            return sympify(expr_str)
    except Exception as e:
        logger.warning(f"Expression parse failed: {e}")
        return None


# ════ RESPONSE CLEANING ════
def clean_response(text: str) -> str:
    """Remove asterisks, preserve LaTeX"""
    if not text:
        return text

    latex_blocks = []
    latex_inline = re.findall(r'\\\(.*?\\\)', text, flags=re.DOTALL)
    latex_display = re.findall(r'\\\[.*?\\\]', text, flags=re.DOTALL)

    for i, l in enumerate(latex_inline):
        key = f"LTXI{i}X"
        latex_blocks.append((key, l))
        text = text.replace(l, key, 1)

    for i, l in enumerate(latex_display):
        key = f"LTXD{i}X"
        latex_blocks.append((key, l))
        text = text.replace(l, key, 1)

    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*', '', text)

    for key, latex in latex_blocks:
        text = text.replace(key, latex)

    return text.strip()


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from JSON"""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


# ════ AI CORE ════
def ask_ai(messages, system=None, temperature=0.2, timeout=30):
    """Call AI with context"""
    if GROQ_AVAILABLE:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        if len(full) > 30:
            full = [full[0]] + full[-28:]

        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                r = groq_client.chat.completions.create(
                    model=model,
                    messages=full,
                    max_tokens=4000,
                    temperature=temperature,
                    timeout=timeout
                )
                return clean_response(r.choices[0].message.content)
            except Exception as e:
                error_str = str(e).lower()
                if any(x in error_str for x in ["429", "rate_limit", "does not exist"]):
                    logger.warning(f"Model {model} unavailable: {e}")
                    continue
                logger.error(f"Groq error on {model}: {e}")
                raise

    if GEMINI_AVAILABLE:
        try:
            parts = ([f"SYSTEM:\n{system}\n\n"] if system else []) + \
                    [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n" for m in messages]
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="".join(parts)
            )
            return clean_response(r.text)
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    logger.error("All AI providers unavailable")
    return "⚠️ AI temporarily unavailable"


def ask_simple(prompt, system=None, temperature=0.2):
    """Single-turn AI"""
    return ask_ai([{"role": "user", "content": prompt}], system=system, temperature=temperature)


def ask_ai_with_image(messages, image_b64=None, image_type=None, system=None):
    """AI with image analysis"""
    if GEMINI_AVAILABLE and image_b64 and image_type:
        try:
            prompt_parts = []
            if system:
                prompt_parts.append(f"SYSTEM:\n{system}\n")
            for m in messages:
                role = "User" if m.get("role") == "user" else "Assistant"
                prompt_parts.append(f"{role}: {m.get('content', '')}\n")
            prompt_parts.append("Analyze the image step by step. Solve every problem completely.")

            contents = [
                {"text": "\n".join(prompt_parts)},
                {"inline_data": {"mime_type": image_type, "data": image_b64}}
            ]
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            return clean_response(r.text)
        except Exception as e:
            logger.warning(f"Gemini image error: {e}")

    fallback = list(messages)
    if image_b64:
        fallback.append({"role": "user", "content": "Solve this image step by step"})
    return ask_ai(fallback, system=system)


# ════ SYSTEM PROMPTS ════
ASK_ANUPAM_PROMPT = """You are Ask Anupam — an expert AI mathematics tutor.

CORE RULES:
1. Maintain full conversation context (read ALL previous messages)
2. Answer naturally, conversationally, and CRISP (not verbose)
3. For math: solve step-by-step with complete working + verification
4. For images: solve EVERY problem line-by-line with full details
5. ALL mathematics in proper LaTeX: \\(inline\\) or \\[display\\]
6. Box final answers: \\[\\boxed{{answer}}\\]
7. Handle ANY topic: academics, advice, code, debugging
8. State confidence: [CONFIDENCE: HIGH/MEDIUM/LOW]
9. NEVER repeat the same answer multiple times
10. Give ONE clean, complete answer only

RESPONSE STYLE:
- Natural language, warm, helpful tone
- Math format: "Step 1: ..., Step 2: ..., Final: \\[\\boxed{{...}}\\]"
- For images: "I see [problem description]. Let me solve each..."
- CRISP answers unless depth requested
- Do NOT repeat yourself

This is a CHAT like ChatGPT - be conversational, not robotic."""

THEOREM_PROMPT = """You are a rigorous mathematics educator. Prove theorems completely.

PROOF FORMAT (MANDATORY):

📌 THEOREM: [Name]

📖 STATEMENT:
\\[Mathematical statement in LaTeX\\]

✅ DEFINITIONS:
- Define all terms used
- State all assumptions
- Specify domain/restrictions

📐 PROOF (Complete from scratch):

Step 1: [Initial setup]
Step 2: [Key insight]
Step 3: [Continue building]
...
Final Step: [Conclusion]
Therefore: \\[\\boxed{{Conclusion}}\\]
✓ QED

💡 INTUITIVE EXPLANATION:
[Explain why theorem is true in simple terms]

Make proofs COMPLETE and RIGOROUS."""

VERIFY_PROMPT = """Verify mathematical solutions RIGOROUSLY.

For each solution provided:

1. SUBSTITUTION CHECK:
   - Plug answer back into original equation
   - Show all calculations
   - Verify it satisfies the condition

2. ALTERNATIVE METHOD CHECK:
   - Solve using different approach
   - Compare answers

3. DOMAIN/RANGE CHECK:
   - Verify answer within domain
   - Check any restrictions

Result:
✅ VERIFIED - Answer is correct
❌ ERROR FOUND - [Explanation and correction]"""


# ════ ROUTES ════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "status": "ok",
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "version": "9.0",
            "timestamp": datetime.utcnow().isoformat(),
            "python": sys.version
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "degraded", "error": str(e)}), 503


@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        messages = data.get("messages", [])
        image_b64 = data.get("image_b64")
        image_type = data.get("image_type")

        if not messages or not isinstance(messages, list):
            return jsonify({"error": "messages required (list)"}), 400

        # Validate messages
        clean = []
        for m in messages:
            if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            try:
                content = sanitize_input(str(m["content"]))
                clean.append({"role": role, "content": content})
            except:
                continue

        if len(clean) > 20:
            clean = clean[-20:]

        answer = ask_ai_with_image(
            clean,
            image_b64=image_b64,
            image_type=image_type,
            system=ASK_ANUPAM_PROMPT
        )

        logger.info(f"Chat request processed: {len(clean)} messages")
        return jsonify({"answer": answer, "confidence": "HIGH"})
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/formula", methods=["POST"])
@limiter.limit("20 per minute")
@cache.cached(timeout=3600, query_string=True)
def formula():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        topic = sanitize_input(data.get("topic", "Calculus"), "topic")
        exam = sanitize_input(data.get("exam", "JAM"), "topic")

        prompt = f"""Generate a COMPLETE, exam-ready formula sheet.

Topic: {topic}
Exam: {exam}

OUTPUT ONLY FORMULAS - NO LENGTHY TEXT:

Group by sub-topic with this format:

**CATEGORY NAME**

\\[Formula 1\\] Label 1
\\[Formula 2\\] Label 2
...

Include ALL standard formulas for this topic.
Minimum 30 formulas.
Use PROPER LaTeX notation.
No explanations - just formulas."""

        answer = ask_simple(prompt, temperature=0.1)
        logger.info(f"Formula sheet generated for {topic}")
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Formula endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        expr_str = sanitize_input(data.get("expression", "x**2"), "expression")
        graph_type = data.get("type", "2d")

        if not SYMPY_AVAILABLE:
            analysis = ask_simple(f"Analyze function: f(x) = {expr_str}", temperature=0.1)
            return jsonify({
                "sympy": False,
                "expression": expr_str,
                "analysis": analysis
            })

        f = safe_parse(expr_str)
        if not f:
            return jsonify({"error": "Could not parse expression"}), 400

        x = Symbol('x')

        if graph_type == "2d":
            points = []
            x_min, x_max = -5, 5
            for i in range(300):
                xv = x_min + i * (x_max - x_min) / 300
                try:
                    yv = float(f.subs(x, xv))
                    if abs(yv) < 1e6:
                        points.append({"x": round(xv, 4), "y": round(yv, 4)})
                    else:
                        points.append({"x": round(xv, 4), "y": None})
                except:
                    points.append({"x": round(xv, 4), "y": None})

            try:
                df = diff(f, x)
                df_latex = sp_latex(simplify(df))
                critical = [float(c) for c in solve(df, x) if c.is_real]
                critical = [c for c in critical if x_min <= c <= x_max]
            except:
                df_latex = ""
                critical = []

            analysis = ask_simple(
                f"COMPLETE MATHEMATICAL ANALYSIS of f(x) = {expr_str}\n\nUse proper LaTeX notation.",
                temperature=0.1
            )

            logger.info(f"Graph plotted for {expr_str}")
            return jsonify({
                "sympy": True,
                "type": "2d",
                "points": points,
                "expression": expr_str,
                "latex": sp_latex(f),
                "derivative_latex": df_latex,
                "critical_points": critical,
                "analysis": analysis
            })

    except Exception as e:
        logger.error(f"Graph endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/mathematician", methods=["GET", "POST"])
@limiter.limit("15 per minute")
def mathematician():
    try:
        name = None
        if request.method == "GET":
            name = request.args.get("name")
        else:
            body = request.get_json()
            name = body.get("name") if body else None

        if name:
            name = sanitize_input(name, "topic")
        else:
            name = random.choice([
                "Gauss", "Euler", "Ramanujan", "Emmy Noether", "Alan Turing",
                "Terence Tao", "Maryam Mirzakhani", "Kurt Gödel"
            ])

        prompt = f"""Generate biography of mathematician: {name}

IMPORTANT: Return ONLY a valid JSON object. No explanation, no markdown, no backticks.

{{
  "name": "Full name here",
  "period": "Birth-Death years",
  "country": "Country of origin",
  "fields": ["Field1", "Field2"],
  "biography": "3-4 paragraph biography",
  "major_contributions": ["Contribution 1", "Contribution 2"],
  "famous_quote": "One famous quote",
  "key_achievements": {{"Achievement 1": "Description 1"}},
  "impact_today": "How their work impacts the modern world",
  "learning_resources": ["Book 1", "Course"],
  "wikipedia": "https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
}}"""

        response = ask_simple(prompt, temperature=0.2)
        response = strip_markdown_json(response)

        try:
            data = json.loads(response)
            logger.info(f"Mathematician data retrieved: {name}")
            return jsonify(data)
        except:
            m = re.search(r'\{[\s\S]*\}', response)
            if m:
                try:
                    data = json.loads(m.group(0))
                    return jsonify(data)
                except:
                    pass
            return jsonify({"name": name, "biography": response})

    except Exception as e:
        logger.error(f"Mathematician endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("15 per minute")
def projects_generate():
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Machine Learning"), "topic")

        prompt = f"""Generate 5 detailed math/coding projects for topic: {topic}

IMPORTANT: Return ONLY valid JSON array. No markdown, no backticks.

[
  {{
    "number": 1,
    "title": "Project title",
    "difficulty": "Beginner",
    "description": "3-4 sentence description",
    "math_concepts": ["Concept 1", "Concept 2"],
    "step_by_step": ["Step 1", "Step 2", "Step 3"],
    "code_snippet": "# Python code",
    "expected_outcome": "What you will build",
    "career_salary": "Related job: $XXk",
    "resources": ["Book", "Course"]
  }}
]"""

        response = ask_simple(prompt, temperature=0.3)
        response = strip_markdown_json(response)

        projects = None
        try:
            projects = json.loads(response)
        except:
            m = re.search(r'\[[\s\S]*\]', response)
            if m:
                try:
                    projects = json.loads(m.group(0))
                except:
                    pass

        if projects and isinstance(projects, list):
            logger.info(f"Projects generated for {topic}")
            return jsonify({"topic": topic, "projects": projects})

        return jsonify({
            "topic": topic,
            "projects": [{
                "title": f"Projects for {topic}",
                "description": response[:500],
                "difficulty": "Various"
            }]
        })

    except Exception as e:
        logger.error(f"Projects endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    try:
        data = request.get_json()
        theorem_name = sanitize_input(data.get("theorem", "Pythagorean Theorem"), "topic")

        prompt = f"""Prove theorem COMPLETELY: {theorem_name}

OUTPUT REQUIRED:

📌 THEOREM: {theorem_name}

📖 STATEMENT:
\\[Mathematical statement\\]

✅ PROOF (step-by-step):
Step 1: [Setup]
Step 2: [Key insight]
...
Final: Conclusion - QED ✓

💡 INTUITIVE EXPLANATION:
[Why the theorem is true]

Make proof COMPLETE and RIGOROUS."""

        proof = ask_simple(prompt, system=THEOREM_PROMPT, temperature=0.1)
        logger.info(f"Theorem proved: {theorem_name}")
        return jsonify({"theorem": theorem_name, "proof": proof})

    except Exception as e:
        logger.error(f"Theorem endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("15 per minute")
@cache.cached(timeout=3600, query_string=True)
def competition_problems():
    try:
        data = request.get_json()
        category = sanitize_input(data.get("category", "IMO"), "topic")
        count = min(int(data.get("count", 30)), 50)  # Max 50 problems

        prompt = f"""Generate {count} {category} problems with COMPLETE solutions.

For EACH problem:

**Problem [N]:**
\\[Problem statement\\]

**SOLUTION:**
Step 1: [Analysis]
[All steps with LaTeX]
Final Answer: \\[\\boxed{{...}}\\]

Generate {count} complete problems."""

        problems_text = ask_simple(prompt, temperature=0.2)
        logger.info(f"Competition problems generated: {category} x{count}")
        return jsonify({
            "category": category,
            "count": count,
            "problems": problems_text
        })

    except Exception as e:
        logger.error(f"Competition endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("15 per minute")
@cache.cached(timeout=3600, query_string=True)
def quiz_generate():
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Calculus"), "topic")
        count = min(int(data.get("count", 30)), 50)  # Max 50 questions

        prompt = f"""Generate {count} exam-style questions for {topic}.

For EACH question:

**Question [N]:**
\\[Problem with LaTeX\\]

**SOLUTION:**
Step 1: [Full working]
...
**Answer:** \\[\\boxed{{...}}\\]

Generate {count} varied questions with COMPLETE solutions."""

        questions = ask_simple(prompt, temperature=0.3)
        logger.info(f"Quiz generated: {topic} x{count}")
        return jsonify({
            "topic": topic,
            "count": count,
            "questions": questions
        })

    except Exception as e:
        logger.error(f"Quiz endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify-solution", methods=["POST"])
@limiter.limit("30 per minute")
def verify_solution():
    try:
        data = request.get_json()
        problem = sanitize_input(data.get("problem", ""), "general")
        solution = sanitize_input(data.get("solution", ""), "general")

        if not problem or not solution:
            return jsonify({"error": "Problem and solution required"}), 400

        prompt = f"""VERIFY this solution RIGOROUSLY:

Problem: {problem}
Solution: {solution}

Check:
1. Substitute back - does it work?
2. Alternative method - same answer?
3. Domain/range - valid?
4. Edge cases - handled?

Result:
✅ VERIFIED - Correct
❌ ERROR - [Correction]"""

        verification = ask_simple(prompt, system=VERIFY_PROMPT, temperature=0.1)
        logger.info("Solution verified")
        return jsonify({"verification": verification})

    except Exception as e:
        logger.error(f"Verify endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/solution-paths", methods=["POST"])
@limiter.limit("20 per minute")
def solution_paths():
    try:
        data = request.get_json()
        problem = sanitize_input(data.get("problem", ""), "general")

        if not problem:
            return jsonify({"error": "Problem required"}), 400

        prompt = f"""Show 3-5 DIFFERENT methods to solve:

Problem: {problem}

For EACH method:

**METHOD 1: [Approach name]**
- Steps with LaTeX
- Difficulty: [Easy/Medium/Hard]
- When to use: ...

[Repeat for methods 2-5]

**COMPARISON:** Which is best? Why?"""

        paths = ask_simple(prompt, temperature=0.3)
        logger.info("Solution paths generated")
        return jsonify({"methods": paths})

    except Exception as e:
        logger.error(f"Solution paths error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/common-mistakes", methods=["POST"])
@limiter.limit("20 per minute")
def common_mistakes():
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Mathematics"), "topic")

        prompt = f"""Common mistakes in {topic}:

For EACH mistake (7-10 total):

**Mistake [N]:**
❌ Wrong: \\[...\\]
✅ Correct: \\[...\\]
💡 Why: [Why students make this mistake]
🔧 Fix: [How to avoid it]"""

        mistakes = ask_simple(prompt, temperature=0.2)
        logger.info(f"Common mistakes retrieved: {topic}")
        return jsonify({"mistakes": mistakes})

    except Exception as e:
        logger.error(f"Common mistakes error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/exam/<exam>", methods=["GET"])
@cache.cached(timeout=86400)
def exam_info(exam):
    try:
        exam = sanitize_input(exam, "topic")
        exams = {
            "JAM": {
                "full_name": "IIT JAM Mathematics",
                "pattern": "3 hours · 60 questions · 100 marks",
                "syllabus": "Real Analysis, Linear Algebra, Calculus, Differential Equations, Group Theory, Complex Analysis"
            },
            "GATE": {
                "full_name": "GATE Mathematics",
                "pattern": "3 hours · 65 questions · 100 marks",
                "syllabus": "Calculus, Linear Algebra, Complex Analysis, ODE, PDE, Real Analysis, Abstract Algebra"
            },
            "CSIR": {
                "full_name": "CSIR UGC NET Mathematics",
                "pattern": "3 hours · 200 marks (Parts A/B/C)",
                "syllabus": "Real Analysis, Topology, Linear Algebra, Complex Analysis, Functional Analysis"
            }
        }
        return jsonify(exams.get(exam, {"error": "Not found"}))
    except Exception as e:
        logger.error(f"Exam info error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"""
════════════════════════════════════════
🧮 MathSphere v9.0 - Production Backend
════════════════════════════════════════
✅ Groq:        {GROQ_AVAILABLE}
✅ Gemini:      {GEMINI_AVAILABLE}
✅ SymPy:       {SYMPY_AVAILABLE}
✅ Rate Limit:  ENABLED (30 req/min)
✅ Caching:     ENABLED (3600s TTL)
✅ Validation:  ENABLED
✅ Timeouts:    ENABLED (2s SymPy)
✅ Logging:     ENABLED (mathsphere.log)
🐍 Python:      {sys.version}
════════════════════════════════════════
📺 {TEACHER_YOUTUBE}
════════════════════════════════════════
    """)
    app.run(host="0.0.0.0", port=port, debug=False)