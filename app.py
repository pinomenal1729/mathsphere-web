"""
MathSphere v10.0 — Production Backend — COMPLETE & FIXED
Compatible with: flask==3.0.3, groq==0.9.0, google-genai==0.8.0,
                 sympy==1.13.1, numpy==1.26.4, flask-limiter==3.5.0

Author: Anupam Nigam
Date: March 2026

FEATURES:
✅ AI Chat with image support (Groq + Gemini fallback)
✅ Graph plotter with SymPy
✅ Formula sheet generator
✅ Competition problems
✅ Quiz mode
✅ Research hub
✅ Exam info (JAM, GATE, CSIR)
✅ Previous year questions (PYQs)
✅ Mathematician explorer
✅ Theorem prover
✅ Math projects generator
✅ Rate limiting
✅ Caching
✅ Full error handling
"""

import os
import sys
import io
import json
import logging
import re
import base64
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

# Windows UTF-8 Support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mathsphere.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIG
# ════════════════════════════════════════════════════════════════

load_dotenv()

GROQ_API_KEY   = os.getenv('GROQ_API_KEY', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
SECRET_KEY     = os.getenv('SECRET_KEY', 'mathsphere-secret-2024')
FLASK_ENV      = os.getenv('FLASK_ENV', 'production')
DEBUG_MODE     = FLASK_ENV == 'development'

# ════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ════════════════════════════════════════════════════════════════

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
    JSON_SORT_KEYS=False,
    CACHE_TYPE='SimpleCache',
    CACHE_DEFAULT_TIMEOUT=3600  # 1 hour cache
)

# ════════════════════════════════════════════════════════════════
# CORS SETUP
# ════════════════════════════════════════════════════════════════

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ════════════════════════════════════════════════════════════════
# RATE LIMITER & CACHE
# ════════════════════════════════════════════════════════════════

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["300 per day", "60 per hour"],
    storage_uri="memory://"
)

cache = Cache(app)

# ════════════════════════════════════════════════════════════════
# GROQ CLIENT INITIALIZATION
# ════════════════════════════════════════════════════════════════

GROQ_AVAILABLE = False
groq_client    = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client    = Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
        logger.info("[OK] Groq API connected successfully")
    else:
        logger.warning("[WARN] GROQ_API_KEY environment variable not set")
except ImportError:
    logger.warning("[WARN] Groq library not installed: pip install groq")
except Exception as e:
    logger.warning(f"[WARN] Groq initialization error: {e}")

# ════════════════════════════════════════════════════════════════
# GEMINI CLIENT INITIALIZATION (google-genai==0.8.0)
# ════════════════════════════════════════════════════════════════

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
        logger.info("[OK] Gemini API (google-genai==0.8.0) connected successfully")
    else:
        logger.warning("[WARN] GEMINI_API_KEY environment variable not set")
except ImportError:
    logger.warning("[WARN] google-genai library not installed: pip install google-genai")
except Exception as e:
    logger.warning(f"[WARN] Gemini initialization error: {e}")

# ════════════════════════════════════════════════════════════════
# SYMPY INITIALIZATION
# ════════════════════════════════════════════════════════════════

SYMPY_AVAILABLE = False

try:
    from sympy import (
        Symbol, N, diff, solve,
        sin, cos, tan, exp, log, sqrt, pi, E
    )
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor
    )
    SYMPY_AVAILABLE = True
    logger.info("[OK] SymPy library loaded successfully")
except ImportError:
    logger.warning("[WARN] SymPy library not installed: pip install sympy")
except Exception as e:
    logger.warning(f"[WARN] SymPy initialization error: {e}")

# ════════════════════════════════════════════════════════════════
# NUMPY INITIALIZATION (with fallback)
# ════════════════════════════════════════════════════════════════

NUMPY_AVAILABLE = False

try:
    from numpy import isfinite as _isfinite, isnan as _isnan
    NUMPY_AVAILABLE = True
    logger.info("[OK] NumPy library loaded successfully")
except ImportError:
    logger.warning("[WARN] NumPy library not installed: pip install numpy")
    # Fallback implementations
    def _isfinite(x):
        try:
            val = float(x)
            return val not in (float('inf'), float('-inf'))
        except (TypeError, ValueError):
            return False

    def _isnan(x):
        try:
            val = float(x)
            return val != val  # NaN != NaN
        except (TypeError, ValueError):
            return True

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

FORMATTING RULES:
- ALL mathematics in LaTeX:
  - Inline: \\( ... \\)
  - Display: \\[ ... \\]
- Use **bold** for key terms and definitions
- Use ## for section headings
- Be thorough, encouraging, and pedagogically excellent
- For images: first describe what you see, then solve"""

# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def ask_ai(messages: list, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    """
    Call Groq first, fallback to Gemini if Groq unavailable.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Model temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
    
    Returns:
        Generated text response
    """
    if not messages:
        return ""

    # ── Try Groq First ──
    if GROQ_AVAILABLE and groq_client:
        try:
            logger.info("[GROQ] Sending request to Groq API")
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": MATH_SYSTEM}] + messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                logger.info("[GROQ] Response received successfully")
                return text
        except Exception as e:
            logger.warning(f"[GROQ] Error (falling back to Gemini): {e}")

    # ── Try Gemini Fallback ──
    if GEMINI_AVAILABLE and gemini_client and genai_types:
        try:
            logger.info("[GEMINI] Sending request to Gemini API")
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
            logger.info("[GEMINI] Response received successfully")
            return resp.text or ""
        except Exception as e:
            logger.error(f"[GEMINI] Error: {e}")

    # ── No AI service available ──
    return "⚠️ No AI service available. Ensure GROQ_API_KEY or GEMINI_API_KEY is set in .env"


def ask_simple(prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    """Convenience function for single-prompt requests."""
    return ask_ai([{"role": "user", "content": prompt}], temperature, max_tokens)


def sanitize(text: str, max_len: int = 5000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Raw user input
        max_len: Maximum allowed length
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = str(text).strip()[:max_len]
    
    # Block dangerous patterns
    dangerous_patterns = [
        r'__import__',
        r'\beval\b',
        r'\bexec\b',
        r'subprocess',
        r'\bos\.',
        r'__builtins__'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"[SECURITY] Blocked dangerous input: {text[:60]}")
            return ""
    
    return text


def parse_int_field(value, default: int, min_value: int, max_value: int, 
                   field_name: str = "value") -> int:
    """
    Parse and validate integer request fields.
    
    Args:
        value: Raw value from request
        default: Default value if not provided
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Field name for error messages
    
    Returns:
        Validated integer
    
    Raises:
        ValueError: If value is invalid
    """
    if value is None or value == "":
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be an integer")

    if parsed < min_value or parsed > max_value:
        raise ValueError(
            f"{field_name} must be between {min_value} and {max_value}"
        )
    
    return parsed


def escape_html(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s or '')
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )

# ════════════════════════════════════════════════════════════════
# STATIC FILE ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main HTML index page."""
    try:
        return send_from_directory(STATIC_DIR, 'index.html')
    except FileNotFoundError:
        return jsonify({"error": "index.html not found in static/"}), 404


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files (CSS, JS, images, etc.)."""
    try:
        return send_from_directory(STATIC_DIR, filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# ════════════════════════════════════════════════════════════════
# HEALTH CHECK ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """Check backend health and service availability."""
    return jsonify({
        "status": "healthy",
        "version": "10.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": GROQ_AVAILABLE,
            "gemini": GEMINI_AVAILABLE,
            "sympy": SYMPY_AVAILABLE,
            "numpy": NUMPY_AVAILABLE
        },
        "environment": FLASK_ENV
    }), 200

# ════════════════════════════════════════════════════════════════
# CHAT ENDPOINT — MAIN FEATURE (WITH IMAGE SUPPORT)
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@limiter.limit("40 per minute")
def chat():
    """
    Main chat endpoint supporting text and image-based queries.
    
    Request format:
    {
        "message": "string" OR "messages": [{"role": "user|assistant", "content": "..."}],
        "image_b64": "optional base64 encoded image",
        "image_type": "image/jpeg|png|webp|gif"
    }
    
    Response:
    {
        "answer": "response text with LaTeX"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        # ═══════════════════════════════════════════════════════════
        # PARSE MESSAGES
        # ═══════════════════════════════════════════════════════════
        messages = data.get("messages")
        if not messages:
            # Support single message format
            single = sanitize(data.get("message", ""))
            if not single:
                return jsonify({"error": "No message provided"}), 400
            messages = [{"role": "user", "content": single}]

        # Validate and clean messages (keep last 20)
        clean = []
        for m in messages[-20:]:
            role = str(m.get("role", "user")).lower()
            content = sanitize(str(m.get("content", "")), 4000)
            if content and role in ("user", "assistant"):
                clean.append({"role": role, "content": content})

        if not clean:
            return jsonify({"error": "Empty message"}), 400

        # ═══════════════════════════════════════════════════════════
        # IMAGE HANDLING (FIXED VERSION)
        # ═══════════════════════════════════════════════════════════
        img_b64  = data.get("image_b64")
        img_type = data.get("image_type", "image/jpeg")

        # Parse data URI format if provided
        if isinstance(img_b64, str) and img_b64.startswith("data:") and "," in img_b64:
            header, encoded = img_b64.split(",", 1)
            img_b64 = encoded
            # Extract MIME type from data URI
            if ";base64" in header and header.startswith("data:"):
                img_type = header[5:].split(";", 1)[0] or img_type

        if img_b64:
            logger.info(f"[IMAGE] Processing image: type={img_type}, size={len(img_b64)} chars")

            try:
                # ───────────────────────────────────────────────────
                # Validate MIME type
                # ───────────────────────────────────────────────────
                allowed_mimes = {
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                    "image/gif"
                }
                if img_type not in allowed_mimes:
                    error_msg = (
                        f"Unsupported image type: {img_type}. "
                        f"Supported: {', '.join(allowed_mimes)}"
                    )
                    logger.warning(f"[IMAGE] {error_msg}")
                    return jsonify({"error": error_msg}), 400

                # ───────────────────────────────────────────────────
                # Decode base64
                # ───────────────────────────────────────────────────
                try:
                    raw_bytes = base64.b64decode(img_b64, validate=True)
                except Exception as decode_err:
                    logger.error(f"[IMAGE] Base64 decode failed: {decode_err}")
                    return jsonify({
                        "error": "Invalid image payload: base64 decode failed"
                    }), 400

                if not raw_bytes:
                    logger.error("[IMAGE] Decoded bytes are empty")
                    return jsonify({
                        "error": "Invalid image payload: empty after decode"
                    }), 400

                # ───────────────────────────────────────────────────
                # Validate size
                # ───────────────────────────────────────────────────
                image_size_mb = len(raw_bytes) / (1024 * 1024)
                if len(raw_bytes) > 10 * 1024 * 1024:
                    logger.warning(
                        f"[IMAGE] Image too large: {image_size_mb:.2f}MB (max 10MB)"
                    )
                    return jsonify({
                        "error": f"Image too large ({image_size_mb:.2f}MB). Max 10MB."
                    }), 400

                logger.info(f"[IMAGE] Image decoded: {len(raw_bytes)} bytes")

                # ───────────────────────────────────────────────────
                # Check Gemini availability
                # ───────────────────────────────────────────────────
                if not (GEMINI_AVAILABLE and gemini_client and genai_types):
                    logger.error("[IMAGE] Gemini not available!")
                    return jsonify({
                        "error": (
                            "Image solving unavailable. "
                            "Configure GEMINI_API_KEY in .env"
                        )
                    }), 503

                # ───────────────────────────────────────────────────
                # Get user question
                # ───────────────────────────────────────────────────
                prompt_text = (
                    clean[-1]["content"]
                    if clean
                    else "Please solve this mathematics problem step by step."
                )

                logger.info(
                    f"[IMAGE] Sending to Gemini: "
                    f"prompt_len={len(prompt_text)}, "
                    f"image_size={len(raw_bytes)} bytes"
                )

                # ───────────────────────────────────────────────────
                # ✅ CREATE IMAGE PART (THE FIX!)
                # ───────────────────────────────────────────────────
                try:
                    image_part = genai_types.Part.from_bytes(
                        data=raw_bytes,
                        mime_type=img_type
                    )
                except Exception as part_err:
                    logger.error(f"[IMAGE] Failed to create image part: {part_err}")
                    return jsonify({
                        "error": f"Failed to process image: {str(part_err)[:100]}"
                    }), 502

                # ───────────────────────────────────────────────────
                # ✅ SEND TO GEMINI WITH IMAGE
                # ───────────────────────────────────────────────────
                try:
                    resp = gemini_client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[
                            MATH_SYSTEM,
                            prompt_text,
                            image_part  # ✅ IMAGE INCLUDED!
                        ]
                    )

                    answer = (resp.text or "").strip()
                    logger.info(
                        f"[IMAGE] Gemini response received: {len(answer)} chars"
                    )

                    if not answer:
                        return jsonify({
                            "error": (
                                "Could not read the image. "
                                "Please ensure:\n"
                                "1. Image is clear and well-lit\n"
                                "2. Text/equations are sharp and readable\n"
                                "3. For handwritten work, contrast is good\n"
                                "Try uploading a clearer image."
                            )
                        }), 422

                    return jsonify({"answer": answer}), 200

                except Exception as gemini_err:
                    logger.exception(f"[IMAGE] Gemini API error: {gemini_err}")
                    return jsonify({
                        "error": (
                            f"Gemini API error: {str(gemini_err)[:100]}"
                        )
                    }), 502

            except ValueError as ve:
                logger.error(f"[IMAGE] ValueError: {ve}")
                return jsonify({"error": f"Invalid image: {str(ve)}"}), 400
            except Exception as img_err:
                logger.exception(f"[IMAGE] Unexpected error: {img_err}")
                return jsonify({
                    "error": f"Image processing failed: {str(img_err)[:100]}"
                }), 502

        # ═══════════════════════════════════════════════════════════
        # TEXT-ONLY CHAT (no image)
        # ═══════════════════════════════════════════════════════════
        logger.info("[TEXT] Processing text-only chat")
        answer = ask_ai(clean)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# GRAPH PLOTTER ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/graph", methods=["POST"])
@limiter.limit("20 per minute")
def graph_plotter():
    """
    Plot mathematical functions and analyze them.
    
    Request:
    {
        "expression": "x**2",
        "type": "2d"
    }
    
    Response:
    {
        "points": [...],
        "expression": "x**2",
        "analysis": "Deep analysis with LaTeX",
        "success": true
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        expr_str = sanitize(data.get("expression", "x**2"), 300)
        gtype = data.get("type", "2d")
        points = []

        # ──────────────────────────────────────────────────────────
        # Generate points using SymPy
        # ──────────────────────────────────────────────────────────
        if SYMPY_AVAILABLE and expr_str:
            try:
                logger.info(f"[GRAPH] Plotting: {expr_str}")

                # Clean expression
                clean_expr = expr_str.replace('^', '**').replace('π', 'pi')

                # Parse with SymPy
                transformations = standard_transformations + (
                    implicit_multiplication_application,
                    convert_xor
                )

                x = Symbol('x')
                local_dict = {
                    'x': x,
                    'pi': pi,
                    'E': E,
                    'sin': sin,
                    'cos': cos,
                    'tan': tan,
                    'exp': exp,
                    'log': log,
                    'sqrt': sqrt
                }

                f = parse_expr(clean_expr, transformations=transformations, 
                             local_dict=local_dict)

                # Generate plot points
                x_min, x_max, num_points = -10, 10, 400
                step = (x_max - x_min) / num_points

                for i in range(num_points + 1):
                    xv = x_min + i * step
                    try:
                        yv = float(N(f.subs(x, xv), 8))
                        # Check if finite
                        if _isfinite(yv) and not _isnan(yv) and abs(yv) < 1e6:
                            points.append({"x": round(xv, 4), "y": round(yv, 4)})
                        else:
                            points.append({"x": round(xv, 4), "y": None})
                    except Exception:
                        points.append({"x": round(xv, 4), "y": None})

                logger.info(f"[GRAPH] Generated {len(points)} points")

            except Exception as e:
                logger.warning(f"[GRAPH] SymPy error: {e}")

        # ──────────────────────────────────────────────────────────
        # Generate analysis
        # ──────────────────────────────────────────────────────────
        analysis_prompt = (
            f"Analyze the function f(x) = {expr_str}:\n\n"
            f"Provide:\n"
            f"1. Domain and range in LaTeX\n"
            f"2. x-intercepts and y-intercepts\n"
            f"3. Symmetry properties (even/odd)\n"
            f"4. Asymptotes (if any)\n"
            f"5. Critical points where f'(x) = 0\n"
            f"6. Inflection points where f''(x) = 0\n"
            f"7. Behavior as x → ±∞\n"
            f"8. Which IIT JAM / GATE / CSIR NET topics cover this\n\n"
            f"Use LaTeX for all mathematics: \\(inline\\) and \\[display\\]"
        )

        analysis = ask_simple(analysis_prompt, temperature=0.2, max_tokens=1500)

        return jsonify({
            "success": True,
            "sympy": SYMPY_AVAILABLE,
            "points": points,
            "expression": expr_str,
            "type": gtype,
            "analysis": analysis
        }), 200

    except Exception as e:
        logger.exception(f"Graph error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# FORMULA SHEET ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/formula", methods=["POST"])
@limiter.limit("15 per minute")
def formula():
    """
    Generate comprehensive formula sheets for any topic.
    
    Request:
    {
        "topic": "Calculus",
        "exam": "IIT JAM"
    }
    
    Response:
    {
        "answer": "Complete formula sheet with LaTeX"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        exam = sanitize(data.get("exam", "General"), 50)

        logger.info(f"[FORMULA] Generating for {topic} ({exam})")

        prompt = f"""Generate a COMPLETE, detailed formula sheet for **{topic}** for **{exam}** exam.

Structure your response as:

## Section 1: [First Major Category]
**1.** \\[ important_formula_1 \\]  
Formula usage and exam relevance.

**2.** \\[ important_formula_2 \\]  
When and how to use this formula.

[Continue with 8-10 formulas per section]

## Section 2: [Second Major Category]
[Same format]

## Quick Tricks for {exam}
- Trick 1: [Shortcut and when to use]
- Trick 2: [Shortcut and when to use]
- ... (5-7 tricks total)

## Common Mistakes
- Mistake 1: [What students get wrong]
- Mistake 2: [Another common error]

REQUIREMENTS:
- Minimum 50 numbered formulas
- Every formula in LaTeX
- Cover ALL subtopics of {topic}
- Include derivations where helpful
- Mark which are most important for {exam}
- Use descriptive headings"""

        answer = ask_simple(prompt, temperature=0.05, max_tokens=4000)
        logger.info(f"[FORMULA] Generated {len(answer)} characters")

        return jsonify({
            "answer": answer or f"Could not generate sheet for {topic}."
        }), 200

    except Exception as e:
        logger.error(f"Formula error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# COMPETITION PROBLEMS ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/competition/problems", methods=["POST"])
@limiter.limit("10 per minute")
def competition_problems():
    """
    Generate Olympiad-style competition problems.
    
    Request:
    {
        "category": "IMO",
        "count": 10
    }
    
    Response:
    {
        "problems": "Problem statements with solutions"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        category = sanitize(data.get("category", "IMO"), 50)
        count = parse_int_field(
            data.get("count", 10),
            default=10,
            min_value=1,
            max_value=30,
            field_name="count"
        )

        logger.info(f"[COMPETITION] Generating {count} {category} problems")

        prompt = f"""Generate {count} {category}-style competition problems with full pedagogical solutions.

For EACH problem, use this EXACT format:

---
## Problem {{N}} — [Creative Title]
**Difficulty:** Easy/Medium/Hard | **Topic:** [Area of Math]
**Year:** [Approximate year] | **Country:** [If applicable]

**Problem Statement:**
\\[ problem_in_latex \\]

**Hint (for self-study):**
[Short hint without spoiling]

**Solution:**

**Step 1 — [Description]:**
\\[ work_here \\]

**Step 2 — [Description]:**
\\[ work_here \\]

[Continue with detailed steps]

**Final Answer:**
\\[ \\boxed{{answer}} \\]

**Why this matters:**
[Connection to broader concepts]

**Follow-up Question:**
[A related question to deepen understanding]

---

Ensure:
- Problems are authentic {category}-level
- Solutions are rigorous and complete
- Each problem teaches important technique
- Difficulty increases gradually"""

        problems = ask_simple(prompt, temperature=0.3, max_tokens=4000)
        logger.info(f"[COMPETITION] Generated {len(problems)} characters")

        return jsonify({
            "problems": problems or "Could not generate problems."
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Competition error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# QUIZ ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/quiz/generate", methods=["POST"])
@limiter.limit("10 per minute")
def quiz_generate():
    """
    Generate exam-style MCQ quiz.
    
    Request:
    {
        "topic": "Calculus",
        "count": 10
    }
    
    Response:
    {
        "questions": "Quiz with solutions"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Calculus"), 100)
        count = parse_int_field(
            data.get("count", 10),
            default=10,
            min_value=1,
            max_value=30,
            field_name="count"
        )

        logger.info(f"[QUIZ] Generating {count} MCQs on {topic}")

        prompt = f"""Generate {count} exam-style MCQs on **{topic}**.

For EACH question, use this format:

---
## Question {{N}}
**Marks:** {{1|2|3}} | **Type:** MCQ | **Difficulty:** Easy/Medium/Hard

\\[ question_statement_in_latex \\]

**(A)** \\( option_A \\)

**(B)** \\( option_B \\)

**(C)** \\( option_C \\)

**(D)** \\( option_D \\)

**Correct Answer:** (X)

**Solution:**
\\[ working \\]

**Key Insight:**
[Explain the concept being tested]

**Why other options are wrong:**
- (A) is wrong because...
- (B) is wrong because...
- (D) is wrong because...

**Exam Coverage:**
[IIT JAM / GATE / CSIR NET topic]

---

Requirements:
- Mix of difficulty levels
- Each option is plausible
- Solutions are detailed
- Teaches important concepts
- Covers all subtopics of {topic}"""

        questions = ask_simple(prompt, temperature=0.2, max_tokens=4000)
        logger.info(f"[QUIZ] Generated {len(questions)} characters")

        return jsonify({
            "questions": questions or "Could not generate quiz."
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception(f"Quiz error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ════════════════════════════════════════════════════════════════
# RESEARCH HUB ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/research", methods=["POST"])
@limiter.limit("15 per minute")
def research_hub():
    """
    Research hub for deep learning on topics.
    
    Request:
    {
        "type": "literature|topic|methods|career|resources",
        "query": "Topic name"
    }
    
    Response:
    {
        "response": "Detailed research content"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        rtype = sanitize(data.get("type", "topic"), 50)
        query = sanitize(data.get("query", ""), 500)

        if not query:
            return jsonify({"error": "Query required"}), 400

        logger.info(f"[RESEARCH] {rtype}: {query}")

        # Define prompts for each research type
        prompts = {
            "literature": (
                f"Write a comprehensive literature review for '{query}':\n\n"
                f"Include:\n"
                f"1. Historical development\n"
                f"2. Key papers and authors\n"
                f"3. Major theoretical results\n"
                f"4. Important theorems (in LaTeX)\n"
                f"5. Open problems\n"
                f"6. Recent developments\n"
                f"7. Applications\n"
                f"8. Suggested reading list with citations"
            ),
            "topic": (
                f"Provide a comprehensive deep-dive on '{query}':\n\n"
                f"Include:\n"
                f"1. Clear definition (in LaTeX)\n"
                f"2. Historical context\n"
                f"3. Key theorems with proof sketches\n"
                f"4. Important examples\n"
                f"5. Connections to other areas\n"
                f"6. Common misconceptions\n"
                f"7. Study roadmap\n"
                f"8. Practice problems (conceptual)"
            ),
            "methods": (
                f"List all problem-solving methods for '{query}':\n\n"
                f"For each method:\n"
                f"1. Name and definition\n"
                f"2. When to use it\n"
                f"3. Step-by-step procedure\n"
                f"4. Worked example (in LaTeX)\n"
                f"5. Advantages\n"
                f"6. Disadvantages\n"
                f"7. Common mistakes\n"
                f"8. Similar methods"
            ),
            "career": (
                f"Provide career guidance for '{query}' in India:\n\n"
                f"Include:\n"
                f"1. Academic paths (JAM/GATE/CSIR)\n"
                f"2. Industry roles and companies\n"
                f"3. Required skills\n"
                f"4. Salary ranges (approx)\n"
                f"5. Top institutions\n"
                f"6. Research opportunities\n"
                f"7. Job market trends\n"
                f"8. How to prepare"
            ),
            "resources": (
                f"Best study resources for '{query}':\n\n"
                f"Include:\n"
                f"1. Textbooks with authors\n"
                f"2. Online courses (NPTEL, MIT OCW, Coursera)\n"
                f"3. YouTube channels\n"
                f"4. Research papers\n"
                f"5. Problem collections\n"
                f"6. Previous year papers\n"
                f"7. Study notes\n"
                f"8. Tools and software"
            )
        }

        prompt = prompts.get(rtype, prompts["topic"])
        response = ask_simple(prompt, temperature=0.2, max_tokens=2500)
        logger.info(f"[RESEARCH] Generated {len(response)} characters")

        return jsonify({
            "response": response or "No results found."
        }), 200

    except Exception as e:
        logger.error(f"Research error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# EXAM INFO ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/exam/info", methods=["POST"])
@limiter.limit("20 per minute")
def exam_info():
    """
    Get information about entrance exams.
    
    Request:
    {
        "exam": "jam|gate|csir",
        "type": "info|syllabus"
    }
    
    Response:
    {
        "exam": "jam",
        "details": {...}
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam", "jam"), 20).lower().strip()
        req_type = sanitize(data.get("type", "info"), 20).lower()

        logger.info(f"[EXAM] {exam.upper()} - {req_type}")

        # Exam database
        EXAM_DATA = {
            "jam": {
                "title": "IIT JAM — Joint Admission Test for M.Sc.",
                "when": "February every year (CBT online)",
                "duration": "3 hours",
                "questions": (
                    "60 questions total:\n"
                    "- Section A: 10 MCQs (1 mark each)\n"
                    "- Section B: 30 MCQs (2 marks each)\n"
                    "- Section C: 20 NAT (Integer/Decimal, varies)"
                ),
                "subjects": (
                    "Real Analysis, Linear Algebra, Calculus, "
                    "Differential Equations (ODE/PDE), Abstract Algebra, "
                    "Complex Analysis, Numerical Methods, Statistics"
                ),
                "eligibility": (
                    "Bachelor's degree with Mathematics as main subject\n"
                    "Min 55% (General/OBC) or 50% (SC/ST)"
                ),
                "fee": "₹1,800 (General/OBC) | ₹900 (SC/ST/PwD/Female)",
                "admission": (
                    "M.Sc. programs at:\n"
                    "IIT Bombay, Delhi, Guwahati, Kanpur, Kharagpur, "
                    "Madras, Roorkee, BHU, ISI, IISc"
                ),
                "cutoff": "~50-70/100 (varies by IIT and category)"
            },
            "gate": {
                "title": "GATE Mathematics (MA) — Graduate Aptitude Test",
                "when": "January–February every year (CBT online)",
                "duration": "3 hours | 100 marks total",
                "questions": (
                    "65 questions:\n"
                    "- MCQ: 30 questions (1 or 2 marks)\n"
                    "- NAT: 25 questions (1 or 2 marks)\n"
                    "- MSQ: 10 questions (1 or 2 marks)"
                ),
                "subjects": (
                    "Calculus, Linear Algebra, Real Analysis, "
                    "Complex Analysis, ODE, PDE, Abstract Algebra, "
                    "Functional Analysis, Numerical Analysis, "
                    "Probability & Statistics"
                ),
                "eligibility": (
                    "B.E./B.Tech/B.Sc./M.Sc. or equivalent\n"
                    "Final year students also eligible"
                ),
                "fee": "₹1,800 (General/OBC) | ₹900 (SC/ST/PwD/Female)",
                "admission": (
                    "M.Tech/Ph.D. at IITs/NITs/GFTIs\n"
                    "PSU recruitment\n"
                    "Valid for 3 years"
                ),
                "cutoff": "~40-60/100"
            },
            "csir": {
                "title": "CSIR NET Mathematical Sciences",
                "when": "June & December every year",
                "duration": "3 hours | 200 marks",
                "questions": (
                    "Part A: 20 questions (General Aptitude) - 40 marks\n"
                    "Part B: 40 questions (50% correct answers) - 80 marks\n"
                    "Part C: 60 questions (Best 40 attempted) - 80 marks"
                ),
                "subjects": (
                    "Full UG+PG Mathematics:\n"
                    "Real & Complex Analysis, Topology, Algebra, "
                    "Differential Geometry, Functional Analysis, "
                    "Measure Theory, ODE/PDE, Mechanics, "
                    "Numerical Analysis, Statistics"
                ),
                "eligibility": (
                    "M.Sc. (Math/Physics/Chemistry) or equivalent\n"
                    "Final year candidates also eligible\n"
                    "Age ≤ 28 for JRF (with relaxations)"
                ),
                "fee": "₹1,000 (General) | ₹500 (OBC-NCL) | ₹250 (SC/ST/PwD)",
                "admission": (
                    "JRF Fellowship: ₹37,000/month (first 2 years), "
                    "then ₹42,000/month\n"
                    "Lectureship in CSIR Labs\n"
                    "PhD at universities"
                ),
                "cutoff": "Top 6% qualify | JRF ≈ top 200-300"
            }
        }

        details = EXAM_DATA.get(exam, EXAM_DATA["jam"])

        # Generate detailed syllabus if requested
        if req_type == "syllabus":
            syllabus_prompt = (
                f"Generate a COMPLETE, detailed syllabus for {details['title']}:\n\n"
                f"Include:\n"
                f"1. Each topic with subtopics\n"
                f"2. Key theorems (in LaTeX)\n"
                f"3. Approximate weightage per topic\n"
                f"4. Important formulas\n"
                f"5. Recommended sections from standard books\n"
                f"6. Important definitions\n"
                f"7. Practice problem types"
            )
            syl = ask_simple(syllabus_prompt, max_tokens=2500)
            details = dict(details)
            details["subjects"] = syl or details["subjects"]

        logger.info(f"[EXAM] Returning info for {exam.upper()}")

        return jsonify({
            "exam": exam,
            "details": details
        }), 200

    except Exception as e:
        logger.error(f"Exam info error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# PYQ LOAD ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/pyq/load", methods=["POST"])
@limiter.limit("10 per minute")
def pyq_load():
    """
    Load previous year questions.
    
    Request:
    {
        "exam": "jam|gate|csir",
        "count": 10
    }
    
    Response:
    {
        "success": true,
        "questions": "PYQ content",
        "exam": "jam",
        "count": 10
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        exam = sanitize(data.get("exam", "jam"), 20).lower().strip()
        count = parse_int_field(
            data.get("count", 10),
            default=10,
            min_value=1,
            max_value=50,
            field_name="count"
        )

        logger.info(f"[PYQ] Loading {count} {exam.upper()} questions")

        exam_names = {
            "jam": "IIT JAM Mathematics",
            "gate": "GATE Mathematics (MA)",
            "csir": "CSIR NET Mathematical Sciences"
        }
        exam_name = exam_names.get(exam, "IIT JAM Mathematics")

        prompt = f"""Generate {count} realistic PYQ-style questions for **{exam_name}**.

For EACH question, use this EXACT format:

**Question N** (Year: 20XX | Topic: [Area] | Type: MCQ/NAT/MSQ)

\\[ Question statement in LaTeX \\]

**(A)** \\( option_A \\)

**(B)** \\( option_B \\)

**(C)** \\( option_C \\)

**(D)** \\( option_D \\)

**Topic:** [Specific topic]

**Difficulty:** Easy/Medium/Hard

**Answer:** (Filled after solving)

---

Requirements:
- Make questions authentic-looking
- Include variety of topics
- Vary difficulty levels
- Ensure they're solvable with standard techniques
- No trick questions
- Mirror actual exam format"""

        raw = ask_simple(prompt, temperature=0.2, max_tokens=4000)
        logger.info(f"[PYQ] Generated {len(raw)} characters")

        if not raw or len(raw) < 100:
            return jsonify({
                "success": False,
                "error": "Could not generate PYQs"
            }), 500

        return jsonify({
            "success": True,
            "questions": raw,
            "exam": exam,
            "count": count
        }), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        logger.exception(f"PYQ error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

# ════════════════════════════════════════════════════════════════
# MATHEMATICIAN ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/mathematician", methods=["POST"])
@limiter.limit("15 per minute")
def mathematician():
    """
    Explore mathematicians' lives and works.
    
    Request:
    {
        "name": "Ramanujan"  // optional
    }
    
    Response:
    {
        "name": "...",
        "period": "...",
        "biography": "...",
        ...
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = sanitize(data.get("name", ""), 100)

        subject = (
            f"the mathematician {name}"
            if name
            else "a randomly chosen lesser-known but influential mathematician"
        )

        logger.info(f"[MATH] Exploring: {name or 'random'}")

        prompt = f"""Return information about {subject} as VALID JSON ONLY.

Output format (VALID JSON, no markdown):
{{
  "name": "Full Name",
  "period": "Birth–Death (e.g., 1887–1920)",
  "country": "Country of origin",
  "fields": ["Field1", "Field2", "Field3"],
  "biography": "2-3 sentence biography",
  "famous_quote": "A famous quote or empty string",
  "major_contributions": [
    "Contribution 1 (with LaTeX if applicable)",
    "Contribution 2",
    "Contribution 3"
  ],
  "impact_today": "How their work affects modern mathematics",
  "learning_resources": [
    "Book: Title by Author",
    "Online: Website or course",
    "Paper: Title (link if available)"
  ]
}}

MUST be VALID JSON. No markdown. No preamble."""

        raw = ask_simple(prompt, temperature=0.3, max_tokens=1500)

        # Try to parse as JSON
        try:
            # Clean markdown if present
            clean = re.sub(r'```(?:json)?|```', '', raw or '').strip()
            s, e = clean.find('{'), clean.rfind('}') + 1
            if s >= 0 and e > s:
                parsed = json.loads(clean[s:e])
                logger.info(f"[MATH] Parsed JSON for {parsed.get('name', 'Unknown')}")
                return jsonify(parsed), 200
        except Exception as je:
            logger.warning(f"[MATH] JSON parse failed: {je}")

        # Fallback: return as text
        return jsonify({
            "name": name or "Mathematician",
            "period": "",
            "country": "",
            "fields": [],
            "biography": raw or "Information unavailable.",
            "famous_quote": "",
            "major_contributions": [],
            "impact_today": "",
            "learning_resources": []
        }), 200

    except Exception as e:
        logger.error(f"Mathematician error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# THEOREM PROVER ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/theorem/prove", methods=["POST"])
@limiter.limit("15 per minute")
def theorem_prove():
    """
    Prove mathematical theorems with full rigor.
    
    Request:
    {
        "theorem": "Pythagorean Theorem"
    }
    
    Response:
    {
        "proof": "Complete proof with steps"
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        theorem = sanitize(data.get("theorem", "Pythagorean Theorem"), 300)

        logger.info(f"[THEOREM] Proving: {theorem}")

        prompt = f"""Prove: **{theorem}**

Structure your response as:

## Theorem Statement
\\[ formal statement in LaTeX \\]

## Definitions
[Key definitions needed]

## Prerequisites
[What the student must know first]

## 🧠 Proof Strategy
[Main idea and approach]

## 📚 Exam Relevance
[Which exams test this theorem]

## Formal Proof

**Step 1 — [Description]:**
\\[ work_here \\]
Explanation

**Step 2 — [Description]:**
\\[ work_here \\]
Explanation

[Continue with all steps]

**Conclusion:** \\[ \\blacksquare \\]

## Alternative Proofs
[1-2 other proofs briefly]

## Related Theorems
[2-3 connected results]

## Applications
[3-4 practical applications with examples]

## Practice Problems
[3 problems testing understanding of this theorem]

Use LaTeX for ALL mathematics."""

        proof = ask_simple(prompt, temperature=0.1, max_tokens=3500)
        logger.info(f"[THEOREM] Generated {len(proof)} characters")

        return jsonify({
            "proof": proof or f"Could not prove {theorem}."
        }), 200

    except Exception as e:
        logger.error(f"Theorem error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# PROJECTS ENDPOINT
# ════════════════════════════════════════════════════════════════

@app.route("/api/projects/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate_projects():
    """
    Generate math projects with code.
    
    Request:
    {
        "topic": "Machine Learning"
    }
    
    Response:
    {
        "projects": [...]  // JSON array of projects
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = sanitize(data.get("topic", "Machine Learning"), 100)

        logger.info(f"[PROJECTS] Generating for {topic}")

        prompt = f"""Generate 5 mathematical projects for: **{topic}**

Output ONLY valid JSON array, no markdown:

[
  {{
    "title": "Project Title",
    "difficulty": "Beginner",
    "description": "2-3 sentences about the project",
    "math_concepts": ["Concept1", "Concept2", "Concept3"],
    "objectives": ["Objective 1", "Objective 2"],
    "step_by_step": [
      "Step 1: Description",
      "Step 2: Description",
      "Step 3: Description",
      "Step 4: Description",
      "Step 5: Description"
    ],
    "code_snippet": "# Python code\\nimport numpy as np\\n# Example code",
    "expected_output": "What the project should produce",
    "extensions": ["Extension 1", "Extension 2"],
    "resources": ["Book/Paper 1", "Online Resource 2"]
  }},
  {{"...second project (Intermediate)..."}},
  {{"...third project (Advanced)..."}},
  {{"...fourth and fifth projects..."}}
]

Requirements:
- Increasing difficulty
- Real math concepts
- Practical and educational
- Achievable in 2-4 weeks
- Includes code
- Has clear objectives"""

        raw = ask_simple(prompt, temperature=0.3, max_tokens=3500)
        logger.info(f"[PROJECTS] Generated {len(raw)} characters")

        try:
            # Extract JSON
            clean = re.sub(r'```(?:json)?|```', '', raw or '').strip()
            s, e = clean.find('['), clean.rfind(']') + 1
            if s >= 0 and e > s:
                projects = json.loads(clean[s:e])
                logger.info(f"[PROJECTS] Parsed {len(projects)} projects")
                return jsonify({"projects": projects}), 200
        except Exception as je:
            logger.warning(f"[PROJECTS] JSON parse failed: {je}")

        # Fallback: return as text
        return jsonify({
            "projects": raw or "Could not generate projects."
        }), 200

    except Exception as e:
        logger.error(f"Projects error: {e}")
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    logger.warning(f"[RATELIMIT] Rate limit exceeded: {e}")
    return jsonify({
        "error": "Rate limit exceeded. Please wait a moment and try again."
    }), 429


@app.errorhandler(404)
def not_found(e):
    """Handle 404 not found."""
    logger.warning(f"[404] Path not found: {e}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 internal server error."""
    logger.error(f"[500] Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(400)
def bad_request(e):
    """Handle 400 bad request."""
    logger.warning(f"[400] Bad request: {e}")
    return jsonify({"error": "Bad request"}), 400

# ════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════

def print_startup():
    """Print startup banner and service status."""
    print("\n" + "═" * 70)
    print("  🧮  MathSphere v10.0 — Production Backend Ready")
    print("═" * 70)
    print(f"  Environment:                        {FLASK_ENV.upper()}")
    print(f"  Debug Mode:                         {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"")
    print(f"  API Services:")
    print(f"    Groq (groq==0.9.0):              {'✅ Connected' if GROQ_AVAILABLE else '❌ Not configured'}")
    print(f"    Gemini (google-genai==0.8.0):    {'✅ Connected' if GEMINI_AVAILABLE else '❌ Not configured'}")
    print(f"    SymPy (sympy==1.13.1):           {'✅ Loaded' if SYMPY_AVAILABLE else '❌ Not installed'}")
    print(f"    NumPy (numpy==1.26.4):           {'✅ Loaded' if NUMPY_AVAILABLE else '❌ Not installed'}")
    print(f"")
    print(f"  Paths:")
    print(f"    Static Files:                     {STATIC_DIR}")
    print(f"    Log File:                         {os.path.abspath('mathsphere.log')}")
    print(f"")
    print(f"  Endpoints Available:")
    print(f"    GET  /")
    print(f"    GET  /api/health")
    print(f"    POST /api/chat                    (text + image)")
    print(f"    POST /api/graph")
    print(f"    POST /api/formula")
    print(f"    POST /api/competition/problems")
    print(f"    POST /api/quiz/generate")
    print(f"    POST /api/research")
    print(f"    POST /api/exam/info")
    print(f"    POST /api/pyq/load")
    print(f"    POST /api/mathematician")
    print(f"    POST /api/theorem/prove")
    print(f"    POST /api/projects/generate")
    print("═" * 70 + "\n")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_startup()
    
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting MathSphere on {host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=DEBUG_MODE,
        use_reloader=False  # Important for production
    )