"""
╔════════════════════════════════════════════════════════════════╗
║           MathSphere v10.0 - Production Backend                ║
║        Complete Flask Application with All Improvements         ║
╚════════════════════════════════════════════════════════════════╝

FIXED: Now serves HTML frontend at root route
"""

import os
import sys
import io
import json
import logging
import time
import re
import base64
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Tuple

# ════════════════════════════════════════════════════════════════
# 1. IMPORT FLASK & EXTENSIONS
# ════════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════════
# 2. FIX WINDOWS ENCODING (UTF-8)
# ════════════════════════════════════════════════════════════════

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ════════════════════════════════════════════════════════════════
# 3. SETUP LOGGING
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mathsphere.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 4. LOAD ENVIRONMENT VARIABLES
# ════════════════════════════════════════════════════════════════

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
FLASK_ENV = os.getenv('FLASK_ENV', 'production')

# ════════════════════════════════════════════════════════════════
# 5. IMPORT AI LIBRARIES
# ════════════════════════════════════════════════════════════════

try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE = True
    logger.info("[OK] Groq connected")
except Exception as e:
    logger.warning(f"[WARN] Groq failed: {e}")
    GROQ_AVAILABLE = False
    groq_client = None

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
    logger.info("[OK] Gemini connected")
except Exception as e:
    logger.warning(f"[WARN] Gemini failed: {e}")
    GEMINI_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
# 6. IMPORT SYMPY (SAFE)
# ════════════════════════════════════════════════════════════════

try:
    logger.info("[INFO] SymPy import started...")
    
    from sympy import (
        symbols, sympify, diff, integrate, solve, simplify, expand,
        factor, latex as sp_latex, Matrix,
        limit, series, oo, pi, E, I, sqrt, Rational, N, dsolve,
        Function, Eq, Symbol, cancel, apart, together, trigsimp,
        det, trace, eye, zeros, ones, linsolve, nonlinsolve,
        Sum, Product, cos, sin, tan, exp, log, Abs, asin, acos, atan,
        sinh, cosh, tanh, factorial, gcd, lcm, isprime, factorint,
        Derivative, Integral, summation
    )
    
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor
    )
    
    SYMPY_AVAILABLE = True
    logger.info("[OK] SymPy loaded successfully")
    print("[OK] SymPy loaded successfully")
    
except ImportError as e:
    logger.warning(f"[WARN] SymPy ImportError: {e}")
    print(f"[WARN] SymPy ImportError: {e}")
    SYMPY_AVAILABLE = False
except Exception as e:
    logger.error(f"[ERROR] SymPy failed: {type(e).__name__}: {e}")
    print(f"[ERROR] SymPy failed: {type(e).__name__}: {e}")
    SYMPY_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
# 7. NUMPY IMPORT
# ════════════════════════════════════════════════════════════════

try:
    from numpy import isfinite, isnan
    NUMPY_AVAILABLE = True
    logger.info("[OK] NumPy available")
except Exception as e:
    logger.warning(f"[WARN] NumPy not available: {e}")
    NUMPY_AVAILABLE = False
    # Fallback functions
    def isfinite(x):
        try:
            return float(x) != float('inf') and float(x) != float('-inf')
        except:
            return False
    def isnan(x):
        try:
            return float(x) != float(x)  # NaN != NaN
        except:
            return True

# ════════════════════════════════════════════════════════════════
# 8. INITIALIZE FLASK APP
# ════════════════════════════════════════════════════════════════

# Get the absolute path to the static folder
base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, static_folder=static_dir, static_url_path='')
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# ════════════════════════════════════════════════════════════════
# 9. SETUP CORS
# ════════════════════════════════════════════════════════════════

CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ════════════════════════════════════════════════════════════════
# 10. SETUP RATE LIMITER
# ════════════════════════════════════════════════════════════════

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ════════════════════════════════════════════════════════════════
# 11. SETUP CACHING
# ════════════════════════════════════════════════════════════════

cache_config = {
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600  # 1 hour
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# ════════════════════════════════════════════════════════════════
# 12. UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def sanitize_input(user_input: str, input_type: str = "text") -> str:
    """Sanitize and validate user input"""
    if not user_input:
        return ""
    
    # Strip whitespace
    user_input = user_input.strip()
    
    # Limit length
    if input_type == "expression":
        max_len = 200
        allowed_chars = r'^[a-zA-Z0-9x\(\)\+\-\*\/\^\.\,\s]*$'
    elif input_type == "topic":
        max_len = 100
        allowed_chars = r'^[a-zA-Z0-9\s\-]*$'
    else:
        max_len = 5000
        allowed_chars = None
    
    if len(user_input) > max_len:
        user_input = user_input[:max_len]
    
    # Basic injection prevention
    dangerous_patterns = [
        r'__import__',
        r'eval',
        r'exec',
        r'system',
        r'shell',
        r'os\.',
        r'subprocess'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            logger.warning(f"Blocked potentially dangerous input: {user_input[:50]}")
            return ""
    
    return user_input


def safe_parse(expr_str: str):
    """Safely parse a mathematical expression"""
    if not SYMPY_AVAILABLE:
        return None
    
    try:
        # Clean up expression
        expr_str = expr_str.replace('^', '**')  # Convert ^ to **
        expr_str = expr_str.replace('π', 'pi')
        expr_str = expr_str.replace('e', str(E))
        
        # Parse with transformations
        transformations = (standard_transformations + 
                          (implicit_multiplication_application, convert_xor))
        
        expr = parse_expr(expr_str, transformations=transformations)
        return expr
    except Exception as e:
        logger.warning(f"Parse error for '{expr_str}': {e}")
        return None


def timeout_decorator(timeout_seconds=2):
    """Decorator to timeout long-running operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation exceeded {timeout_seconds} seconds")
            
            # Set timeout (Windows doesn't support signals well, so try/except)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except Exception as e:
                logger.warning(f"Timeout in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator


def ask_simple(prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
    """Get response from LLM (try Groq first, fallback to Gemini)"""
    
    if not prompt:
        return ""
    
    try:
        # Try Groq first (faster)
        if GROQ_AVAILABLE and groq_client:
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert mathematics tutor. Provide clear, concise explanations with proper mathematical notation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq error, trying Gemini: {e}")
        
        # Fallback to Gemini
        if GEMINI_AVAILABLE:
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text if response.text else ""
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return "Error: Unable to generate response. Please try again."
        
        return "Error: No AI service available"
        
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return ""


# ════════════════════════════════════════════════════════════════
# 13. FRONTEND ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the frontend HTML"""
    try:
        return send_from_directory(static_dir, 'index.html')
    except FileNotFoundError:
        logger.error("index.html not found in static folder")
        return jsonify({
            "error": "Frontend not found",
            "message": "Place index.html in the 'static' folder",
            "status": "backend-only"
        }), 404


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static assets (CSS, JS, images)"""
    try:
        return send_from_directory(static_dir, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


# ════════════════════════════════════════════════════════════════
# 14. HEALTH CHECK ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "message": "MathSphere v10.0 running",
        "services": {
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "numpy": NUMPY_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }), 200


# ════ CHAT ENDPOINT ════
@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    """Main chat endpoint - Ask Anupam"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = sanitize_input(data.get("message", ""), "text")
        mode = sanitize_input(data.get("mode", "ask"), "topic")
        image_data = data.get("image", None)  # Base64 image
        
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # System prompts by mode
        system_prompts = {
            "ask": "You are Anupam, an expert mathematics tutor. Answer clearly with proper mathematical notation. Use LaTeX for formulas: \\[formula\\]",
            "explain": "Explain this concept in detail with examples. Use LaTeX for mathematics.",
            "solve": "Solve this step-by-step. Show all working. Use LaTeX for each step.",
            "verify": "Check if this solution is correct. Provide feedback.",
            "suggest": "Suggest the best approach to solve this problem."
        }
        
        system_prompt = system_prompts.get(mode, system_prompts["ask"])
        
        # Handle image (if provided)
        if image_data:
            prompt = f"[IMAGE PROVIDED]\n\n{message}\n\nAnalyze the image and provide a detailed solution."
        else:
            prompt = message
        
        # Get response from AI
        response = ask_simple(prompt, temperature=0.2, max_tokens=2000)
        
        if not response:
            response = "Unable to generate response. Please try again."
        
        logger.info(f"Chat: {message[:50]}...")
        
        return jsonify({
            "answer": response,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ GRAPH ENDPOINT ════
@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    """Plot graphs - 2D and 3D"""
    try:
        if not SYMPY_AVAILABLE:
            return jsonify({"error": "Math engine not available"}), 503
        
        data = request.get_json()
        expr_str = sanitize_input(data.get("expression", "x**2"), "expression")
        graph_type = sanitize_input(data.get("type", "2d"), "topic")
        
        if not expr_str:
            return jsonify({"error": "Invalid expression"}), 400
        
        # Parse expression
        f = safe_parse(expr_str)
        if not f:
            return jsonify({"error": "Invalid mathematical expression"}), 400
        
        # Generate points
        x = Symbol('x')
        x_min, x_max = -5, 5
        step = (x_max - x_min) / 300
        points = []
        
        x_vals = [x_min + i * step for i in range(301)]
        
        for xv in x_vals:
            try:
                yv = float(N(f.subs(x, xv), 5))
                
                # Validate point
                if yv is not None and isfinite(yv) and not isnan(yv):
                    if abs(yv) < 1e4:  # Reasonable range
                        points.append({"x": round(xv, 4), "y": round(yv, 4)})
                    else:
                        points.append({"x": round(xv, 4), "y": None})
                else:
                    points.append({"x": round(xv, 4), "y": None})
            except:
                points.append({"x": round(xv, 4), "y": None})
        
        # Get analysis
        analysis = f"Function: f(x) = {expr_str}\n"
        analysis += f"Points generated: {len([p for p in points if p['y'] is not None])}\n"
        
        try:
            df = diff(f, x)
            critical_pts = solve(df, x)
            critical_pts = [float(c) for c in critical_pts if c.is_real and x_min <= float(c) <= x_max]
            
            if critical_pts:
                analysis += f"Critical points: {', '.join([f'{c:.4f}' for c in critical_pts[:3]])}"
        except:
            pass
        
        logger.info(f"Graph generated for {expr_str}")
        
        return jsonify({
            "sympy": True,
            "points": points,
            "expression": expr_str,
            "type": graph_type,
            "analysis": analysis,
            "success": True
        }), 200
        
    except Exception as e:
        logger.error(f"Graph error: {e}")
        return jsonify({"error": str(e), "success": False}), 500


# ════ FORMULA SHEET ENDPOINT ════
@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
@cache.cached(timeout=3600, query_string=True)
def formula():
    """Generate formula sheets"""
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Calculus"), "topic")
        exam = sanitize_input(data.get("exam", "General"), "topic")
        
        prompt = f"""Generate a COMPLETE formula sheet for {topic} (Exam: {exam}).

FORMAT:
**CATEGORY 1: [Name]**
1. \\[formula1\\] - Brief label
2. \\[formula2\\] - Brief label
3. \\[formula3\\] - Brief label

**CATEGORY 2: [Name]**
[Continue with numbered formulas]

REQUIREMENTS:
✓ MINIMUM 40 formulas
✓ MAXIMUM 1 sentence per label
✓ Every formula in \\[...\\] or \\(...\\)
✓ Organized by logical categories
✓ NO duplicates
✓ Include: theorems, identities, techniques
✓ Cover all major subtopics for {topic}

Topic: {topic}
Exam: {exam}"""

        answer = ask_simple(prompt, temperature=0.05, max_tokens=3000)
        
        if not answer or len(answer) < 200:
            answer = f"Unable to generate complete formula sheet for {topic}. Please try again."
        
        formula_count = answer.count('\\[') + answer.count('\\(')
        
        logger.info(f"Formula sheet: {topic} ({formula_count} formulas)")
        
        return jsonify({
            "answer": answer,
            "topic": topic,
            "exam": exam,
            "formula_count": formula_count
        }), 200
        
    except Exception as e:
        logger.error(f"Formula error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ COMPETITION PROBLEMS ENDPOINT ════
@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("10 per minute")
def competition_problems():
    """Generate competition problems"""
    try:
        data = request.get_json()
        category = sanitize_input(data.get("category", "IMO"), "topic")
        count = min(int(data.get("count", 10)), 30)
        
        all_problems = []
        chunk_size = 5
        
        for chunk_num in range(0, count, chunk_size):
            remaining = count - chunk_num
            current_chunk_size = min(chunk_size, remaining)
            
            prompt = f"""Generate exactly {current_chunk_size} {category} competition problems (Problems {chunk_num+1} to {chunk_num+current_chunk_size}).

FOR EACH PROBLEM - Use EXACTLY this format:

**Problem {chunk_num+1}: [Problem Title]**
Difficulty: [Easy/Medium/Hard]
\\[Problem statement in LaTeX\\]

**SOLUTION:**
Step 1: [Analysis with LaTeX]
Step 2: [Derivation with LaTeX]
Step 3: [Complete working]

Final Answer: \\[\\boxed{{answer}}\\]

Key Insight: One sentence explanation

---

NOW GENERATE {current_chunk_size} PROBLEMS WITH ALL FORMULAS IN LATEX:"""

            try:
                response = ask_simple(prompt, temperature=0.2, max_tokens=2000)
                
                if response and len(response) > 150:
                    all_problems.append(response)
                    logger.info(f"Competition chunk {chunk_num//chunk_size + 1}/{(count + chunk_size - 1)//chunk_size}")
                else:
                    logger.warning(f"Incomplete chunk {chunk_num//chunk_size + 1}")
                    
            except Exception as e:
                logger.error(f"Chunk error: {e}")
        
        full_response = "\n".join(all_problems)
        problem_count = full_response.count("**Problem")
        
        return jsonify({
            "category": category,
            "count": count,
            "problems": full_response,
            "generated": problem_count,
            "success": problem_count >= (count * 0.8)
        }), 200
        
    except Exception as e:
        logger.error(f"Competition problems error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ QUIZ ENDPOINT ════
@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    """Generate quiz questions"""
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Calculus"), "topic")
        count = min(int(data.get("count", 10)), 30)
        
        all_questions = []
        chunk_size = 5
        
        for chunk_num in range(0, count, chunk_size):
            remaining = count - chunk_num
            current_chunk_size = min(chunk_size, remaining)
            
            prompt = f"""Generate exactly {current_chunk_size} {topic} exam-style questions (Questions {chunk_num+1} to {chunk_num+current_chunk_size}).

FOR EACH QUESTION - Use EXACTLY this format:

**Question {chunk_num+1}:**
\\[Problem in LaTeX\\]

A) Option 1
B) Option 2
C) Option 3
D) Option 4

**Answer:** [B]
**Solution:**
Step 1: [Working with LaTeX]
Step 2: [Continue]
Step 3: [Final answer]

Explanation: Why correct, why others wrong.

---

GENERATE {current_chunk_size} QUESTIONS WITH ALL MATH IN LATEX:"""

            try:
                response = ask_simple(prompt, temperature=0.2, max_tokens=2000)
                
                if response and len(response) > 150:
                    all_questions.append(response)
                    logger.info(f"Quiz chunk {chunk_num//chunk_size + 1}/{(count + chunk_size - 1)//chunk_size}")
                else:
                    logger.warning(f"Quiz chunk {chunk_num//chunk_size + 1} incomplete")
                    
            except Exception as e:
                logger.error(f"Quiz chunk error: {e}")
        
        full_response = "\n".join(all_questions)
        
        return jsonify({
            "topic": topic,
            "count": count,
            "questions": full_response,
            "success": True
        }), 200
        
    except Exception as e:
        logger.error(f"Quiz error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ RESEARCH ENDPOINT ════
@app.route("/api/research", methods=["POST"])
@limiter.limit("15 per minute")
def research_hub():
    """Research Hub - Literature, topics, methods, career"""
    try:
        data = request.get_json()
        research_type = sanitize_input(data.get("type", "topic"), "topic")
        query = sanitize_input(data.get("query", ""), "text")
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        prompt = f"""Provide {research_type} information for "{query}":
        
        For topic: Explain comprehensively with prerequisites, applications, and connections
        For literature: Find papers and articles with summaries and citations
        For methods: Show different approaches with step-by-step solutions
        For career: Discuss career paths, salaries, skills needed
        
        Use LaTeX for mathematical content. Be detailed and helpful."""

        response = ask_simple(prompt, temperature=0.2, max_tokens=2000)
        
        if not response:
            response = f"Unable to complete research for: {query}"
        
        logger.info(f"Research: {research_type} - {query[:30]}")
        
        return jsonify({
            "type": research_type,
            "query": query,
            "response": response
        }), 200
        
    except Exception as e:
        logger.error(f"Research error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ EXAM INFO ENDPOINT ════
@app.route("/api/exam/info", methods=["POST"])
@limiter.limit("20 per minute")
@cache.cached(timeout=3600, query_string=True)
def exam_info():
    """Get exam information"""
    try:
        data = request.get_json()
        exam = sanitize_input(data.get("exam", "jam"), "topic")
        
        exam_details = {
            "jam": {
                "title": "IIT JAM - Joint Admission Test for Masters",
                "when": "January every year",
                "duration": "3 hours",
                "questions": "60 (MCQ + Numeric)",
                "subjects": "Real Analysis, Linear Algebra, Calculus, ODE, Group Theory, Complex Analysis",
                "eligibility": "Bachelor's degree in Science",
                "fee": "1000 INR",
                "admission": "Top performers get admission to IIT Masters programs",
                "cutoff": "50-70 out of 100 (varies by category)"
            },
            "gate": {
                "title": "GATE - Graduate Aptitude Test in Engineering",
                "when": "January/February every year",
                "duration": "3 hours",
                "questions": "65 (MCQ + NAT)",
                "subjects": "Calculus, Linear Algebra, Complex Analysis, ODE, PDE, Real Analysis",
                "eligibility": "Bachelor's degree (any specialization)",
                "fee": "1500 INR",
                "admission": "Used for admission to IIT/NIT postgraduate programs",
                "cutoff": "50-60 out of 100"
            },
            "csir": {
                "title": "CSIR NET - National Eligibility Test",
                "when": "June and December every year",
                "duration": "3 hours",
                "questions": "200 (Objective)",
                "subjects": "Entire undergraduate mathematics curriculum",
                "eligibility": "Master's degree in Science or final year postgraduate",
                "fee": "1000 INR",
                "admission": "JRF (₹31,000/month) and Lecturership positions",
                "cutoff": "Top 6% qualify"
            }
        }
        
        details = exam_details.get(exam, exam_details["jam"])
        
        return jsonify({
            "exam": exam,
            "details": details
        }), 200
        
    except Exception as e:
        logger.error(f"Exam info error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ MATHEMATICIAN ENDPOINT ════
@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    """Get mathematician information"""
    try:
        data = request.get_json()
        name = sanitize_input(data.get("name", ""), "text")
        
        if name:
            prompt = f"""Provide detailed information about the mathematician: {name}

            Include:
            1. Full name and period (birth-death)
            2. Country/region
            3. Main fields of mathematics
            4. Biography (2-3 sentences)
            5. Major contributions (3-4 bullet points)
            6. Famous quote (if available)
            7. Impact on mathematics today
            8. Learning resources (books, papers)
            
            Format as structured data."""
        else:
            prompt = """Suggest a random famous mathematician and provide the same information as above."""
        
        response = ask_simple(prompt, temperature=0.2, max_tokens=2000)
        
        if not response:
            response = "Unable to fetch mathematician information."
        
        logger.info(f"Mathematician: {name or 'random'}")
        
        return jsonify({
            "name": name,
            "response": response
        }), 200
        
    except Exception as e:
        logger.error(f"Mathematician error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ THEOREM ENDPOINT ════
@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    """Prove a theorem"""
    try:
        data = request.get_json()
        theorem = sanitize_input(data.get("theorem", "Pythagorean Theorem"), "text")
        
        prompt = f"""Provide a complete, rigorous proof of: {theorem}

        Include:
        1. Theorem statement (with LaTeX)
        2. Assumptions and prerequisites
        3. Proof (step-by-step with LaTeX)
        4. Key lemmas used
        5. Applications
        6. Historical context
        
        Make it suitable for undergraduate mathematics students."""
        
        response = ask_simple(prompt, temperature=0.1, max_tokens=2500)
        
        if not response:
            response = f"Unable to generate proof for {theorem}"
        
        logger.info(f"Theorem: {theorem}")
        
        return jsonify({
            "theorem": theorem,
            "proof": response
        }), 200
        
    except Exception as e:
        logger.error(f"Theorem error: {e}")
        return jsonify({"error": str(e)}), 500


# ════ PROJECTS ENDPOINT ════
@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    """Generate math projects"""
    try:
        data = request.get_json()
        topic = sanitize_input(data.get("topic", "Machine Learning"), "text")
        
        prompt = f"""Generate 5 detailed mathematics projects for: {topic}

        For EACH project provide:
        1. Title
        2. Difficulty level (Beginner/Intermediate/Advanced)
        3. Description (2-3 sentences)
        4. Mathematical concepts needed
        5. Step-by-step approach (5-6 steps)
        6. Python code snippet (if applicable)
        7. Resources and references
        
        Make projects practical and educational with heavy mathematics focus."""
        
        response = ask_simple(prompt, temperature=0.2, max_tokens=3000)
        
        if not response:
            response = f"Unable to generate projects for {topic}"
        
        logger.info(f"Projects: {topic}")
        
        return jsonify({
            "topic": topic,
            "projects": response
        }), 200
        
    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════
# 15. ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit exceeded"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please wait before trying again."
    }), 429


@app.errorhandler(404)
def not_found(e):
    """Not found"""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist."
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Internal server error"""
    logger.error(f"Internal error: {e}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again."
    }), 500


# ════════════════════════════════════════════════════════════════
# 16. STARTUP INFO
# ════════════════════════════════════════════════════════════════

def print_startup_info():
    """Print startup information"""
    print("\n" + "═" * 60)
    print("🧮 MathSphere v10.0 - Production Backend")
    print("═" * 60)
    print(f"✅ Groq:           {GROQ_AVAILABLE}")
    print(f"✅ Gemini:         {GEMINI_AVAILABLE}")
    print(f"✅ SymPy:          {SYMPY_AVAILABLE}")
    print(f"✅ NumPy:          {NUMPY_AVAILABLE}")
    print(f"✅ Rate Limit:     ENABLED (30 req/min)")
    print(f"✅ Caching:        ENABLED (3600s TTL)")
    print(f"✅ Validation:     ENABLED")
    print(f"✅ Frontend:       SERVING from static/index.html")
    print(f"✅ Logging:        ENABLED (mathsphere.log)")
    print(f"🐍 Python:         {sys.version.split()[0]}")
    print(f"🌐 Environment:    {FLASK_ENV}")
    print("═" * 60)
    print("📺 https://youtube.com/@pi_nomenal1729")
    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════
# 17. MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting MathSphere v10.0")
    print_startup_info()
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 5000))
    
    # Run Flask
    app.run(
        host="0.0.0.0",
        port=port,
        debug=(FLASK_ENV == "development"),
        use_reloader=False
    )