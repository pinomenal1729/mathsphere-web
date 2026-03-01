"""
MathSphere Web v6.0 — Professor Edition
========================================
NEW in v6.0:
✅ SymPy integration — symbolic CAS for verified answers
✅ Answer Verifier — student submits answer, system checks it
✅ Step Validator — validates each step of a student's work
✅ Confidence Scoring — every AI answer gets a verification badge
✅ Numerical Verification — substitution-based cross-checking
✅ Graphing data endpoint — Desmos-ready function data
✅ Error Analysis Mode — identifies exactly where student went wrong
✅ Smart prompt engineering — forces step-by-step, boxes answers, states assumptions
✅ All original features preserved and enhanced

By Anupam Nigam | youtube.com/@pi_nomenal1729
"""

import os, re, json, random, traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── API Keys ─────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GROQ_AVAILABLE  = bool(GROQ_API_KEY)
GEMINI_AVAILABLE= bool(GEMINI_API_KEY)

groq_client    = None
gemini_client  = None
SYMPY_AVAILABLE = False

if GROQ_AVAILABLE:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq connected")
    except Exception as e:
        print(f"⚠️ Groq init failed: {e}"); GROQ_AVAILABLE = False

if GEMINI_AVAILABLE:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini connected")
    except Exception as e:
        print(f"⚠️ Gemini init failed: {e}"); GEMINI_AVAILABLE = False

# ── SymPy Setup ──────────────────────────────────
try:
    import sympy as sp
    from sympy import (
        symbols, sympify, diff, integrate, solve, simplify, expand,
        factor, latex as sp_latex, Matrix, eigenvals, eigenvects,
        limit, series, oo, pi, E, I, sqrt, Rational, N, dsolve,
        Function, Eq, Symbol, cancel, apart, together, trigsimp,
        det, trace, eye, zeros, ones, linsolve, nonlinsolve,
        Sum, Product, FiniteSet, Interval, Union, Intersection,
        cos, sin, tan, exp, log, Abs, conjugate, re as sp_re, im as sp_im,
        binomial, factorial, gcd, lcm, isprime, factorint, primerange,
        Derivative, Integral, DiracDelta, Heaviside, fourier_transform
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor
    )
    SYMPY_AVAILABLE = True
    print("✅ SymPy loaded — symbolic verification active")
except ImportError:
    print("⚠️ SymPy not installed. Run: pip install sympy --break-system-packages")

GROQ_MODELS     = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TEACHER_YOUTUBE = "https://youtube.com/@pi_nomenal1729"
TEACHER_INSTAGRAM="https://instagram.com/pi_nomenal1729"
TEACHER_WEBSITE = "https://www.anupamnigam.com"


# ══════════════════════════════════════════════════
# SYMPY ENGINE — Core symbolic computation
# ══════════════════════════════════════════════════

SYMPY_TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
) if SYMPY_AVAILABLE else None


def safe_parse(expr_str: str):
    """Parse a string expression into a SymPy expression safely."""
    if not SYMPY_AVAILABLE:
        return None
    try:
        # Clean common input formats
        expr_str = expr_str.strip()
        expr_str = re.sub(r'\^', '**', expr_str)
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        return parse_expr(expr_str, transformations=SYMPY_TRANSFORMATIONS)
    except Exception:
        try:
            return sympify(expr_str)
        except Exception:
            return None


def sympy_compute(problem_type: str, expression: str, variable: str = "x",
                  extra: dict = None) -> dict:
    """
    Core SymPy computation engine.
    Returns: {success, result_latex, result_str, steps, error}
    """
    if not SYMPY_AVAILABLE:
        return {"success": False, "error": "SymPy not available"}

    extra = extra or {}
    result = {"success": False, "result_latex": "", "result_str": "", "steps": [], "error": ""}

    try:
        x = Symbol(variable)
        expr = safe_parse(expression)
        if expr is None:
            result["error"] = f"Could not parse: {expression}"
            return result

        if problem_type == "differentiate":
            order = extra.get("order", 1)
            der = diff(expr, x, order)
            simplified = simplify(der)
            result.update({
                "success": True,
                "result_latex": sp_latex(simplified),
                "result_str": str(simplified),
                "steps": [
                    f"Expression: {sp_latex(expr)}",
                    f"Apply differentiation rules (order {order})",
                    f"Result: {sp_latex(simplified)}"
                ]
            })

        elif problem_type == "integrate":
            limits = extra.get("limits")
            if limits:
                a, b = safe_parse(str(limits[0])), safe_parse(str(limits[1]))
                integral = integrate(expr, (x, a, b))
                simplified = simplify(integral)
                result.update({
                    "success": True,
                    "result_latex": sp_latex(simplified),
                    "result_str": str(simplified),
                    "steps": [
                        f"Definite integral from {limits[0]} to {limits[1]}",
                        f"Integrand: {sp_latex(expr)}",
                        f"Result: {sp_latex(simplified)}"
                    ]
                })
            else:
                integral = integrate(expr, x)
                result.update({
                    "success": True,
                    "result_latex": sp_latex(integral) + " + C",
                    "result_str": str(integral) + " + C",
                    "steps": [
                        f"Integrand: {sp_latex(expr)}",
                        f"Apply integration rules",
                        f"Result: {sp_latex(integral)} + C"
                    ]
                })

        elif problem_type == "solve":
            rhs_str = extra.get("rhs", "0")
            rhs = safe_parse(rhs_str)
            solutions = solve(Eq(expr, rhs), x)
            sols_latex = [sp_latex(s) for s in solutions]
            result.update({
                "success": True,
                "result_latex": ", ".join(sols_latex) if sols_latex else "\\text{No solution}",
                "result_str": str(solutions),
                "steps": [
                    f"Equation: {sp_latex(expr)} = {sp_latex(rhs)}",
                    f"Solve for {variable}",
                    f"Solutions: {', '.join(sols_latex)}"
                ]
            })

        elif problem_type == "limit":
            point = extra.get("point", "0")
            direction = extra.get("direction", "+")
            pt = safe_parse(str(point)) if point not in ["oo", "inf", "-oo"] else (oo if point in ["oo", "inf"] else -oo)
            lim = limit(expr, x, pt, direction)
            result.update({
                "success": True,
                "result_latex": sp_latex(lim),
                "result_str": str(lim),
                "steps": [
                    f"Expression: {sp_latex(expr)}",
                    f"Limit as {variable} → {point}",
                    f"Result: {sp_latex(lim)}"
                ]
            })

        elif problem_type == "simplify":
            simp = simplify(expr)
            result.update({
                "success": True,
                "result_latex": sp_latex(simp),
                "result_str": str(simp),
                "steps": [f"Original: {sp_latex(expr)}", f"Simplified: {sp_latex(simp)}"]
            })

        elif problem_type == "factor":
            fac = factor(expr)
            result.update({
                "success": True,
                "result_latex": sp_latex(fac),
                "result_str": str(fac),
                "steps": [f"Original: {sp_latex(expr)}", f"Factored: {sp_latex(fac)}"]
            })

        elif problem_type == "expand":
            exp_result = expand(expr)
            result.update({
                "success": True,
                "result_latex": sp_latex(exp_result),
                "result_str": str(exp_result),
                "steps": [f"Original: {sp_latex(expr)}", f"Expanded: {sp_latex(exp_result)}"]
            })

        elif problem_type == "series":
            pt = safe_parse(str(extra.get("point", "0")))
            n = extra.get("n", 6)
            ser = series(expr, x, pt, n)
            result.update({
                "success": True,
                "result_latex": sp_latex(ser),
                "result_str": str(ser),
                "steps": [
                    f"Function: {sp_latex(expr)}",
                    f"Taylor series around x = {pt}, up to order {n}",
                    f"Result: {sp_latex(ser)}"
                ]
            })

        elif problem_type == "eigenvalues":
            # expression here is a matrix like "[[1,2],[3,4]]"
            try:
                mat_data = json.loads(expression)
                M = Matrix(mat_data)
                evals = M.eigenvals()
                evects = M.eigenvects()
                evals_latex = ", ".join([f"{sp_latex(k)} (\\text{{mult }}={v})" for k, v in evals.items()])
                result.update({
                    "success": True,
                    "result_latex": evals_latex,
                    "result_str": str(evals),
                    "steps": [
                        f"Matrix: {sp_latex(M)}",
                        f"Characteristic polynomial: {sp_latex(M.charpoly())}",
                        f"Eigenvalues: {evals_latex}"
                    ]
                })
            except Exception as me:
                result["error"] = f"Matrix parse error: {me}"

        elif problem_type == "verify_answer":
            # Check if student_answer equals the computed result
            student_ans = safe_parse(extra.get("student_answer", "0"))
            computed = safe_parse(extra.get("computed", "0"))
            if student_ans is not None and computed is not None:
                diff_result = simplify(student_ans - computed)
                is_correct = diff_result == 0
                result.update({
                    "success": True,
                    "is_correct": is_correct,
                    "result_latex": "\\text{Correct}" if is_correct else f"\\text{{Incorrect: difference}} = {sp_latex(diff_result)}",
                    "result_str": "correct" if is_correct else f"incorrect, difference = {diff_result}",
                    "steps": [
                        f"Your answer: {sp_latex(student_ans)}",
                        f"Correct answer: {sp_latex(computed)}",
                        f"Difference: {sp_latex(diff_result)}",
                        "✅ VERIFIED CORRECT" if is_correct else "❌ INCORRECT"
                    ]
                })

        elif problem_type == "numerical_check":
            # Numerically verify by substituting test values
            test_vals = extra.get("test_vals", [1, 2, 3])
            expr2 = safe_parse(extra.get("expr2", "0"))
            if expr2 is not None:
                all_match = True
                checks = []
                for val in test_vals:
                    try:
                        v1 = complex(expr.subs(x, val))
                        v2 = complex(expr2.subs(x, val))
                        match = abs(v1 - v2) < 1e-9
                        if not match:
                            all_match = False
                        checks.append(f"x={val}: LHS={round(v1.real,6)}, RHS={round(v2.real,6)}, {'✅' if match else '❌'}")
                    except Exception:
                        checks.append(f"x={val}: evaluation error")
                result.update({
                    "success": True,
                    "is_correct": all_match,
                    "result_latex": "\\text{Numerically verified}" if all_match else "\\text{Numerical mismatch found}",
                    "result_str": "match" if all_match else "mismatch",
                    "steps": checks
                })

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def detect_math_type(text: str) -> tuple:
    """Detect what type of math problem this is. Returns (type, expression, extra)."""
    text_lower = text.lower().strip()

    # Differentiation
    m = re.search(r"(?:diff(?:erentiate)?|derivative)\s+(?:of\s+)?([^,\n]+?)(?:\s+with respect to\s+(\w))?(?:\s*$|,)", text_lower)
    if m:
        return ("differentiate", m.group(1).strip(), {"variable": m.group(2) or "x"})

    # Integration
    m = re.search(r"(?:integrat|antiderivative)\w*\s+(?:of\s+)?([^,\n]+?)(?:\s+(?:from|with respect))?", text_lower)
    if m:
        limits_m = re.search(r"from\s+(-?[\d\w./]+)\s+to\s+(-?[\d\w./]+|inf(?:inity)?|oo)", text_lower)
        extra = {}
        if limits_m:
            extra["limits"] = [limits_m.group(1), limits_m.group(2)]
        return ("integrate", m.group(1).strip(), extra)

    # Solve equation
    m = re.search(r"solve\s+(.+?)(?:\s*=\s*(.+))?(?:\s+for\s+(\w))?$", text_lower)
    if m:
        rhs = m.group(2) or "0"
        return ("solve", m.group(1).strip(), {"rhs": rhs, "variable": m.group(3) or "x"})

    # Limit
    m = re.search(r"limit\s+(?:of\s+)?(.+?)\s+as\s+(\w)\s*(?:->|→|approaches?)\s*(-?[\w./]+)", text_lower)
    if m:
        return ("limit", m.group(1).strip(), {"variable": m.group(2), "point": m.group(3)})

    # Simplify
    m = re.search(r"simplif\w+\s+(.+)", text_lower)
    if m:
        return ("simplify", m.group(1).strip(), {})

    # Factor
    m = re.search(r"factor\s+(.+)", text_lower)
    if m:
        return ("factor", m.group(1).strip(), {})

    return (None, None, {})


def numerical_verify_derivative(f_str: str, df_str: str, variable: str = "x") -> dict:
    """Verify df is the derivative of f by numerical differentiation."""
    if not SYMPY_AVAILABLE:
        return {"verified": None, "message": "SymPy unavailable"}
    try:
        x = Symbol(variable)
        f  = safe_parse(f_str)
        df = safe_parse(df_str)
        if f is None or df is None:
            return {"verified": None, "message": "Could not parse expressions"}
        test_points = [0.3, 1.1, 2.7, -0.5, 5.2]
        h = 1e-7
        mismatches = 0
        details = []
        for val in test_points:
            try:
                f_val  = float(f.subs(x, val))
                f_val2 = float(f.subs(x, val + h))
                numerical_d = (f_val2 - f_val) / h
                symbolic_d  = float(df.subs(x, val))
                match = abs(numerical_d - symbolic_d) < 1e-4
                if not match:
                    mismatches += 1
                details.append({"x": val, "numerical": round(numerical_d, 6), "symbolic": round(symbolic_d, 6), "match": match})
            except Exception:
                pass
        verified = mismatches == 0
        return {"verified": verified, "details": details, "mismatches": mismatches}
    except Exception as e:
        return {"verified": None, "message": str(e)}


def verify_integral(f_str: str, F_str: str, variable: str = "x") -> dict:
    """Verify F is an antiderivative of f by differentiating F."""
    if not SYMPY_AVAILABLE:
        return {"verified": None, "message": "SymPy unavailable"}
    try:
        x = Symbol(variable)
        f = safe_parse(f_str)
        F_no_c = re.sub(r'\+\s*[Cc]$', '', F_str.strip())
        F = safe_parse(F_no_c)
        if f is None or F is None:
            return {"verified": None, "message": "Could not parse"}
        dF = diff(F, x)
        diff_simplified = simplify(dF - f)
        verified = diff_simplified == 0
        return {
            "verified": verified,
            "F_prime_latex": sp_latex(dF),
            "f_latex": sp_latex(f),
            "diff_latex": sp_latex(diff_simplified),
            "message": "d/dx(F) = f ✅" if verified else f"d/dx(F) ≠ f, difference = {sp_latex(diff_simplified)}"
        }
    except Exception as e:
        return {"verified": None, "message": str(e)}


# ══════════════════════════════════════════════════
# ASTERISK REMOVER
# ══════════════════════════════════════════════════

def clean_response(text: str) -> str:
    if not text: return text
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{3}(.+?)_{3}',   r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{2}(.+?)_{2}',   r'\1', text, flags=re.DOTALL)

    latex_inline  = re.findall(r'\\\(.*?\\\)', text, flags=re.DOTALL)
    latex_display = re.findall(r'\\\[.*?\\\]', text, flags=re.DOTALL)
    ph = {}
    for i, l in enumerate(latex_inline):
        k = f"LTXI{i}X"; ph[k] = l; text = text.replace(l, k, 1)
    for i, l in enumerate(latex_display):
        k = f"LTXD{i}X"; ph[k] = l; text = text.replace(l, k, 1)

    text = re.sub(r'(?<!\S)\*(.+?)\*(?!\S)', r'\1', text)
    text = re.sub(r'(?<=\d)\s*\*\s*(?=\d)', ' × ', text)
    text = re.sub(r'\*', '', text)

    for k, v in ph.items(): text = text.replace(k, v)
    return re.sub(r'\n{4,}', '\n\n\n', text).strip()


# ══════════════════════════════════════════════════
# AI CORE
# ══════════════════════════════════════════════════

def ask_ai(messages, system=None):
    if GROQ_AVAILABLE:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        if len(full) > 15: full = [full[0]] + full[-13:]
        for model in GROQ_MODELS:
            try:
                r = groq_client.chat.completions.create(
                    model=model, messages=full, max_tokens=4000, temperature=0.2)
                return clean_response(r.choices[0].message.content)
            except Exception as e:
                if any(x in str(e).lower() for x in ["429","rate_limit","model_not_active","does not exist"]):
                    continue
                raise

    if GEMINI_AVAILABLE:
        try:
            parts = ([f"SYSTEM:\n{system}\n\n"] if system else []) + \
                    [f"{'Student' if m['role']=='user' else 'Assistant'}: {m['content']}\n" for m in messages]
            r = gemini_client.models.generate_content(model="gemini-2.5-flash", contents="".join(parts))
            return clean_response(r.text)
        except Exception as e:
            print(f"Gemini error: {e}")

    return "⚠️ AI temporarily unavailable. Please try again!"


def ask_simple(prompt, system=None):
    return ask_ai([{"role": "user", "content": prompt}], system=system)


def ask_ai_with_image(messages, image_b64=None, image_type=None, system=None):
    if GEMINI_AVAILABLE and image_b64 and image_type:
        try:
            prompt_parts = []
            if system:
                prompt_parts.append(f"SYSTEM:\n{system}\n")
            for m in messages:
                role = "Student" if m.get("role") == "user" else "Assistant"
                prompt_parts.append(f"{role}: {m.get('content', '')}\n")
            prompt_parts.append("Analyse the uploaded image and answer step by step.")
            contents = [
                {"text": "\n".join(prompt_parts)},
                {"inline_data": {"mime_type": image_type, "data": image_b64}}
            ]
            r = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=contents)
            return clean_response(r.text)
        except Exception as e:
            print(f"Gemini image error: {e}")

    fallback = list(messages)
    if image_b64:
        fallback.append({"role": "user", "content": "User uploaded an image. Solve step by step."})
    return ask_ai(fallback, system=system)


# ══════════════════════════════════════════════════
# SYSTEM PROMPTS — Upgraded for accuracy
# ══════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are MathSphere — an expert Mathematics professor for graduate students, created by Anupam Nigam.

MATHEMATICAL ACCURACY RULES — ABSOLUTE, NO EXCEPTIONS:
1. ALWAYS compute numerical examples to self-verify before writing your answer
2. NEVER skip algebraic steps — show every manipulation
3. State ALL assumptions explicitly (domain, convergence conditions, branches)
4. If you are not 100% certain of a result, say so clearly
5. For calculus: always verify by differentiating your integral result
6. For algebra: always substitute back to verify solutions
7. For series: always state radius/interval of convergence
8. Box the final answer clearly using: \\[\\boxed{{...}}\\]
9. Never present a single answer when multiple solutions exist

FORMAT EVERY RESPONSE:
📌 [Topic Name]
💡 Real-life Application: [1 sentence]
📖 Given information and assumptions
📐 Step-by-step solution (numbered, every step shown)
✅ Final answer in box: \\[\\boxed{{...}}\\]
⚠️ Common mistakes on this type of problem
📚 {TEACHER_YOUTUBE}

STYLE RULES:
1. NEVER use asterisks * or ** — use CAPS for emphasis
2. ALL mathematics MUST be in LaTeX: \\(inline\\) or \\[display\\]
3. Use warm Hinglish tone: "Dekho...", "Samajh aaya?", "Bohot achha!"
4. Temperature is set low — be precise, not creative with math facts
5. HTML tags allowed: <br> <hr>

CONFIDENCE LEVELS — always end with one of:
[CONFIDENCE: HIGH] — you are certain and verified
[CONFIDENCE: MEDIUM] — standard result, verify if critical
[CONFIDENCE: LOW] — complex problem, please verify independently"""


ASK_ANUPAM_PROMPT = f"""You are Ask Anupam — an all-purpose AI tutor by Anupam Nigam.

ACCURACY RULES:
1. For ANY mathematical calculation: compute it carefully, then verify by substituting back
2. NEVER use asterisks in output
3. Show Step 1, Step 2, Step 3... for every math problem
4. Box every final mathematical answer: \\[\\boxed{{...}}\\]
5. ALL math expressions MUST be in proper LaTeX notation
6. If image uploaded: read carefully, state what you see, then solve step by step
7. State confidence: [CONFIDENCE: HIGH/MEDIUM/LOW] at end of math solutions
8. For definitions/concepts: be concise. For calculations: be thorough.

TONE: Friendly, precise, and confident. Like a helpful older sibling who is a math professor."""


VERIFIER_PROMPT = """You are a mathematical answer verifier. Your ONLY job is to:
1. Check if the student's answer is mathematically correct
2. Identify the EXACT step where an error occurs if wrong
3. Provide the correct answer with full working

Be extremely precise. Do not be encouraging — be accurate.
Format: Step-by-step verification with LaTeX. Final verdict: CORRECT or INCORRECT.
Box the correct answer: \\[\\boxed{...}\\]"""


ERROR_ANALYSIS_PROMPT = """You are a mathematical error analyst — like a professor marking exam scripts.
Your job:
1. Read the student's solution carefully
2. Identify EVERY error, no matter how small
3. Classify each error (conceptual/arithmetic/sign/notation/missing step)
4. Show the correct version of each wrong step
5. Give an overall assessment

Be specific. "Error in Step 3: you applied product rule but forgot the second term."
Use LaTeX for all mathematics."""


# ══════════════════════════════════════════════════
# MATHEMATICIANS DATA
# ══════════════════════════════════════════════════

MATHEMATICIANS = {
    "Srinivasa Ramanujan": {
        "period": "1887–1920", "country": "India",
        "fields": ["Number Theory", "Infinite Series", "Modular Forms"],
        "contribution": "Discovered 3900+ results with almost no formal training. His partition function work is used in string theory and black hole physics today.",
        "keyresults": "Ramanujan tau function, Hardy-Ramanujan number 1729, Rogers-Ramanujan identities, mock theta functions",
        "quote": "An equation for me has no meaning unless it expresses a thought of God",
        "image": "https://upload.wikimedia.org/wikipedia/commons/0/02/Srinivasa_Ramanujan_-_OPC_-_1.jpg",
        "impact": "Black hole physics, string theory, partition function applications worth billions in research",
        "resources": ["Wikipedia: https://en.wikipedia.org/wiki/Srinivasa_Ramanujan",
                      "MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Ramanujan.html"]
    },
    "Leonhard Euler": {
        "period": "1707–1783", "country": "Switzerland",
        "fields": ["Analysis", "Graph Theory", "Number Theory", "Topology"],
        "contribution": "Most prolific mathematician ever: 800+ papers. Founded graph theory, created e, π, i notation, solved Basel problem.",
        "keyresults": "Euler identity \\(e^{i\\pi}+1=0\\), Euler formula V-E+F=2, Basel problem \\(\\sum 1/n^2 = \\pi^2/6\\), Königsberg bridges",
        "quote": "Mathematics is the queen of sciences",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d7/Leonhard_Euler.jpg",
        "impact": "Internet networking, electrical engineering, quantum mechanics, every branch of modern math",
        "resources": ["Euler Archive: https://scholarlycommons.pacific.edu/euler/"]
    },
    "Carl Friedrich Gauss": {
        "period": "1777–1855", "country": "Germany",
        "fields": ["Number Theory", "Statistics", "Differential Geometry", "Algebra"],
        "contribution": "Prince of Mathematics. Proved Fundamental Theorem of Algebra at age 21. Invented least squares, Gaussian distribution, modular arithmetic.",
        "keyresults": "FTA, bell curve \\(N(\\mu,\\sigma^2)\\), Gauss-Bonnet theorem, quadratic reciprocity, prime number estimates",
        "quote": "Mathematics is the queen of the sciences and number theory is the queen of mathematics",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Gauss_1840_by_Jensen.jpg",
        "impact": "MRI scanners, GPS systems, machine learning, $1T+ in annual economic applications",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Gauss.html"]
    },
    "Emmy Noether": {
        "period": "1882–1935", "country": "Germany",
        "fields": ["Abstract Algebra", "Theoretical Physics", "Ring Theory"],
        "contribution": "Revolutionised abstract algebra. Noether's theorem connecting symmetry to conservation laws is arguably the most important result in theoretical physics.",
        "keyresults": "Noether's Theorem, Noetherian rings, ascending chain condition, ideal theory, invariant theory",
        "quote": "My methods are really methods of working and thinking; this is why they have crept in everywhere anonymously",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e9/Emmy_Noether_%281882-1935%29.jpg",
        "impact": "All modern physics, conservation of energy/momentum, quantum mechanics, general relativity",
        "resources": ["Stanford Encyclopedia: https://plato.stanford.edu/entries/noether/"]
    },
    "Isaac Newton": {
        "period": "1642–1727", "country": "England",
        "fields": ["Calculus", "Physics", "Classical Mechanics", "Optics"],
        "contribution": "Invented calculus, discovered gravity, formulated three laws of motion.",
        "keyresults": "Calculus, F=ma, universal gravitation \\(F=Gm_1m_2/r^2\\), binomial theorem, Principia Mathematica",
        "quote": "If I have seen further, it is by standing on the shoulders of giants",
        "image": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Principia_Mathematica_1687.jpg",
        "impact": "All classical mechanics, aerospace, civil engineering, space exploration",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Newton.html"]
    },
    "Bernhard Riemann": {
        "period": "1826–1866", "country": "Germany",
        "fields": ["Complex Analysis", "Riemannian Geometry", "Number Theory"],
        "contribution": "Riemann hypothesis (still unsolved!). Riemann integral. Differential geometry enabling Einstein's general relativity.",
        "keyresults": "Riemann hypothesis, Riemann integral, Riemann surfaces, curvature tensor, zeta function",
        "quote": "If only I had the theorems! Then I could find the proofs easily enough",
        "image": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Georg_Friedrich_Bernhard_Riemann.jpg",
        "impact": "Internet cryptography, quantum physics, general relativity, $1M Millennium Prize still unclaimed",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Riemann.html"]
    },
    "David Hilbert": {
        "period": "1862–1943", "country": "Germany",
        "fields": ["Functional Analysis", "Mathematical Logic", "Abstract Algebra"],
        "contribution": "Led formalism movement. His 23 unsolved problems shaped all of 20th-century mathematics.",
        "keyresults": "Hilbert spaces, Hilbert's 23 problems (1900), formalism, metamathematics, spectral theory",
        "quote": "We must know, we will know",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/da/Hilbert.jpg",
        "impact": "Quantum mechanics foundation, optimization, AI/ML mathematical basis",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Hilbert.html"]
    },
    "Georg Cantor": {
        "period": "1845–1918", "country": "Germany",
        "fields": ["Set Theory", "Mathematical Logic", "Foundations of Mathematics"],
        "contribution": "Invented set theory. Proved there are different sizes of infinity.",
        "keyresults": "Set theory, transfinite cardinals \\(\\aleph_0, \\aleph_1\\), diagonal argument, continuum hypothesis, power sets",
        "quote": "The true infinite, the truly infinite, is the Deity",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Georg_Cantor2.jpg",
        "impact": "Foundations of all mathematics, computer science theory, philosophy of infinity",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Cantor.html"]
    },
    "Terence Tao": {
        "period": "1975–present", "country": "Australia",
        "fields": ["Number Theory", "Harmonic Analysis", "PDE", "Combinatorics"],
        "contribution": "Mozart of mathematics. Fields Medal 2006. Solved Green-Tao theorem on primes in arithmetic progressions.",
        "keyresults": "Green-Tao theorem, compressed sensing, Navier-Stokes regularity progress",
        "quote": "What mathematics achieves is remarkable — it describes all patterns of the universe",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Terence_Tao.jpg",
        "impact": "Medical imaging, signal processing, AI mathematics, number theory",
        "resources": ["Tao's Blog: https://terrytao.wordpress.com/"]
    },
    "Maryam Mirzakhani": {
        "period": "1977–2017", "country": "Iran",
        "fields": ["Differential Geometry", "Topology", "Teichmüller Theory"],
        "contribution": "FIRST WOMAN to win Fields Medal (2014). Revolutionary work on dynamics and geometry of Riemann surfaces.",
        "keyresults": "Weil-Petersson volume formulas, moduli space dynamics, Teichmüller geodesics",
        "quote": "The beauty of mathematics only shows itself to more patient followers",
        "image": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Maryam_Mirzakhani.jpg",
        "impact": "String theory, quantum gravity, breaking gender barriers — inspired millions",
        "resources": ["Wikipedia: https://en.wikipedia.org/wiki/Maryam_Mirzakhani"]
    },
    "Alan Turing": {
        "period": "1912–1954", "country": "England",
        "fields": ["Computability Theory", "Cryptography", "Artificial Intelligence"],
        "contribution": "Father of computer science. Turing machine defines computation. Cracked Enigma code in WWII.",
        "keyresults": "Turing machine, halting problem undecidability, Turing test, Enigma decryption",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Alan_Turing_Aged_16.jpg",
        "impact": "All computation, software industry, AI, cybersecurity, saved 14M+ WWII lives",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Turing.html"]
    },
    "Kurt Gödel": {
        "period": "1906–1978", "country": "Austria-Hungary / USA",
        "fields": ["Mathematical Logic", "Set Theory", "Foundations"],
        "contribution": "Incompleteness theorems proved some truths are unprovable within any consistent formal system.",
        "keyresults": "First and Second Incompleteness Theorems, Gödel completeness theorem, constructible universe L",
        "quote": "Either mathematics is too big for the human mind, or the human mind is more than a machine",
        "image": "https://upload.wikimedia.org/wikipedia/commons/8/84/KurtGodel.jpg",
        "impact": "Limits of artificial intelligence, philosophy of mind, halting problem",
        "resources": ["Stanford Encyclopedia: https://plato.stanford.edu/entries/goedel/"]
    },
    "Henri Poincaré": {
        "period": "1854–1912", "country": "France",
        "fields": ["Topology", "Dynamical Systems", "Celestial Mechanics"],
        "contribution": "Last universal mathematician. Founded topology, chaos theory.",
        "keyresults": "Poincaré conjecture (solved 2003), algebraic topology, chaos theory, homology groups",
        "quote": "Mathematics is the art of giving the same name to different things",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Poincare1.jpg",
        "impact": "General relativity, weather prediction (chaos), GPS systems",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Poincare.html"]
    },
    "Ada Lovelace": {
        "period": "1815–1852", "country": "England",
        "fields": ["Algorithm Design", "Computing", "Mathematical Logic"],
        "contribution": "World's first computer programmer. Wrote the first algorithm for Babbage's Analytical Engine.",
        "keyresults": "First computer program, loop and conditional concepts, algorithm for Bernoulli numbers",
        "quote": "The Analytical Engine has no pretensions whatever to originate anything.",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg",
        "impact": "All of computer science, software industry, AI, every program ever written",
        "resources": ["Wikipedia: https://en.wikipedia.org/wiki/Ada_Lovelace"]
    },
    "Augustin-Louis Cauchy": {
        "period": "1789–1857", "country": "France",
        "fields": ["Real Analysis", "Complex Analysis", "Mathematical Rigour"],
        "contribution": "Brought rigour to calculus. Established epsilon-delta definitions. Cauchy sequences, integral formula, residue theorem.",
        "keyresults": "Cauchy sequences, Cauchy integral theorem, \\(\\epsilon\\)-\\(\\delta\\) definitions, Cauchy-Schwarz inequality",
        "quote": "I prefer the man of genius who laboreth without ceasing to perfect his works",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d8/Augustin-Louis_Cauchy_1901.jpg",
        "impact": "Foundation of all modern rigorous mathematics",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Cauchy.html"]
    },
    "Gottfried Leibniz": {
        "period": "1646–1716", "country": "Germany",
        "fields": ["Calculus", "Logic", "Philosophy", "Combinatorics"],
        "contribution": "Co-invented calculus. Invented ∫ integral notation, d/dx derivative notation, and ∞ symbol.",
        "keyresults": "Calculus notation (\\(\\int\\), \\(\\frac{d}{dx}\\)), Leibniz rule, binary number system",
        "quote": "There are no wholly useless truths",
        "image": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Gottfried_Wilhelm_Leibniz%2C_Bernhard_Christoph_Francke.jpg",
        "impact": "All mathematics notation, computer science (binary), programming foundations",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Leibniz.html"]
    }
}


# ══════════════════════════════════════════════════
# MATH PROJECTS
# ══════════════════════════════════════════════════

MATH_PROJECTS = [
    {"id":1,"title":"Machine Learning Classification using Linear Algebra","math":["Linear Algebra","Eigenvalues","SVD","Gradient Descent"],"desc":"Build a complete ML classifier using SVD for dimensionality reduction and linear regression for prediction.","real":"Google Search ranking, Netflix recommendations, Amazon product suggestions","companies":"Google, Netflix, Amazon, IBM Watson, Microsoft Azure ML","salary":"Data Scientist $120K+ | ML Engineer $140K+ | AI Researcher $150K+","difficulty":"Advanced"},
    {"id":2,"title":"Cryptography: RSA Encryption with Number Theory","math":["Number Theory","Prime Numbers","Modular Arithmetic","Euler's Theorem"],"desc":"Implement full RSA encryption using prime factorization. Generate key pairs, encrypt and decrypt messages.","real":"HTTPS (all of internet), WhatsApp E2E encryption, banking","companies":"Apple, Google, Microsoft, all banks, NSA","salary":"Cryptographer $115K+ | Security Engineer $130K+ | CISO $200K+","difficulty":"Intermediate"},
    {"id":3,"title":"3D Graphics Engine using Matrix Transformations","math":["Rotation Matrices","Homogeneous Coordinates","Quaternions","Projections"],"desc":"Build a software 3D renderer using transformation matrices. Implement rotation, scaling, projection and lighting.","real":"Video games (Unity, Unreal Engine), Pixar/DreamWorks films, VR/AR headsets","companies":"Unity, Epic Games, Pixar, Valve, Meta VR, NVIDIA","salary":"Game Developer $110K+ | Graphics Programmer $135K+ | VR Engineer $125K+","difficulty":"Advanced"},
    {"id":4,"title":"Signal Processing: Audio Analysis with Fourier Transform","math":["Fourier Analysis","FFT Algorithm","Complex Numbers","Convolution"],"desc":"Use FFT to analyse audio signals. Build noise filters, equalizers, pitch detection.","real":"Spotify audio processing, noise-cancelling headphones, MRI scanners","companies":"Spotify, Apple, Bose, Sony, Siemens Healthcare","salary":"Audio Engineer $100K+ | DSP Engineer $125K+ | Biomedical Engineer $105K+","difficulty":"Advanced"},
    {"id":5,"title":"Portfolio Optimisation using Lagrange Multipliers","math":["Calculus","Lagrange Multipliers","Convex Optimisation","Covariance Matrices"],"desc":"Maximise portfolio returns while minimising risk using Markowitz mean-variance optimisation.","real":"Goldman Sachs, JP Morgan, BlackRock ($10T AUM), all hedge funds","companies":"Goldman Sachs, JP Morgan, BlackRock, Citadel, Renaissance Technologies","salary":"Quant Analyst $150K+ | Portfolio Manager $200K+ | Hedge Fund Manager $500K+","difficulty":"Intermediate"},
    {"id":6,"title":"Disease Spread Modelling using Differential Equations (SIR)","math":["ODE Systems","SIR/SEIR Models","Numerical Integration","Phase Portraits"],"desc":"Build SIR/SEIR epidemic models using differential equations. Simulate vaccine distribution.","real":"COVID-19 modelling (WHO, CDC, governments worldwide)","companies":"WHO, CDC, ICMR, NIH, McKinsey Health","salary":"Epidemiologist $85K+ | Public Health Modeller $95K+ | Research Scientist $105K+","difficulty":"Intermediate"},
    {"id":7,"title":"PageRank Algorithm using Eigenvalues and Markov Chains","math":["Graph Theory","Eigenvalues/Eigenvectors","Markov Chains","Power Iteration"],"desc":"Implement Google's PageRank. Build a web graph and use power iteration to find the principal eigenvector.","real":"Google Search (handles 99B+ searches/day), Microsoft Bing","companies":"Google, Microsoft, Baidu, Yandex","salary":"Search Engineer $145K+ | Ranking Scientist $155K+ | Graph Data Scientist $130K+","difficulty":"Advanced"},
    {"id":8,"title":"Numerical ODE Solver: Runge-Kutta Methods","math":["Numerical Analysis","RK4","Error Analysis","Stability Theory"],"desc":"Implement and compare Euler, RK2, RK4 ODE solvers. Analyse accuracy vs step-size.","real":"Weather forecasting, aircraft simulation, drug pharmacokinetics","companies":"Boeing, Airbus, NOAA, NASA","salary":"Computational Scientist $120K+ | CFD Engineer $135K+ | Simulation Engineer $125K+","difficulty":"Advanced"},
    {"id":9,"title":"Natural Language Processing using Probability and Vector Spaces","math":["Vector Spaces","Probability Theory","Bayes Theorem","Information Theory"],"desc":"Build text classifier using TF-IDF, Naive Bayes, and word embeddings.","real":"ChatGPT (OpenAI), Gmail spam filter, Google Translate, Siri/Alexa","companies":"OpenAI, Google, Meta AI, Amazon Alexa","salary":"NLP Engineer $135K+ | AI Researcher $155K+ | Conversational AI Lead $165K+","difficulty":"Advanced"},
    {"id":10,"title":"Quantum Computing using Linear Algebra and Complex Numbers","math":["Complex Vector Spaces","Unitary Matrices","Tensor Products","Quantum Gates"],"desc":"Simulate quantum circuits using matrix multiplication. Implement Hadamard, CNOT gates.","real":"IBM Quantum Network, Google Sycamore (quantum supremacy), drug discovery","companies":"IBM, Google Quantum AI, IonQ, Microsoft Azure Quantum","salary":"Quantum Engineer $185K+ | Quantum Researcher $175K+","difficulty":"Very Advanced"},
]


# ══════════════════════════════════════════════════
# THEOREMS
# ══════════════════════════════════════════════════

THEOREMS = {
    "Pythagorean Theorem": {
        "statement": "In a right triangle with legs \\(a, b\\) and hypotenuse \\(c\\): \\[a^2 + b^2 = c^2\\]",
        "proof_sketch": "Construct squares on each side. The two large squares (side a+b) have equal area. Rearranging the inner triangles proves the result. Alternatively: similar triangles formed by the altitude from the right angle each satisfy the relation.",
        "formal_proof": "Let \\(\\triangle ABC\\) have right angle at \\(C\\). Drop altitude \\(CD\\) to hypotenuse. Then \\(\\triangle ACD \\sim \\triangle ABC\\), giving \\(AC^2 = AD \\cdot AB\\). Similarly \\(BC^2 = DB \\cdot AB\\). Adding: \\(AC^2 + BC^2 = AB \\cdot (AD+DB) = AB^2\\). ✅",
        "applications": "Distance formula in \\(\\mathbb{R}^n\\), complex modulus, physics (Pythagoras in energy), GPS triangulation",
        "difficulty": "Basic",
        "exam_relevance": "JAM, GATE, CSIR — appears in geometry, vector spaces, metric spaces"
    },
    "Fundamental Theorem of Calculus": {
        "statement": "If \\(f\\) is continuous on \\([a,b]\\) and \\(F'=f\\), then \\[\\int_a^b f(x)\\,dx = F(b) - F(a)\\]",
        "proof_sketch": "Define \\(G(x) = \\int_a^x f(t)\\,dt\\). Show \\(G'(x) = f(x)\\) using the mean value theorem for integrals. Then \\(G = F + C\\), so \\(\\int_a^b f = G(b)-G(a) = F(b)-F(a)\\).",
        "formal_proof": "By MVT: \\(\\frac{G(x+h)-G(x)}{h} = \\frac{1}{h}\\int_x^{x+h}f(t)\\,dt = f(c_h)\\) for some \\(c_h \\in (x,x+h)\\). As \\(h\\to 0\\), \\(c_h\\to x\\), and by continuity \\(G'(x)=f(x)\\). ✅",
        "applications": "All of integral calculus, physics (work = ∫F·dx), economics (revenue from marginal revenue)",
        "difficulty": "Core",
        "exam_relevance": "CRITICAL for JAM Section A, GATE MA, CSIR Part B — appears every year"
    },
    "Euler's Identity": {
        "statement": "\\[e^{i\\pi} + 1 = 0\\] connecting the five fundamental constants: \\(e, i, \\pi, 1, 0\\).",
        "proof_sketch": "Taylor series: \\(e^x = \\sum x^n/n!\\). Substitute \\(x = i\\theta\\). Real parts give \\(\\cos\\theta\\), imaginary parts give \\(i\\sin\\theta\\). So \\(e^{i\\theta} = \\cos\\theta + i\\sin\\theta\\). At \\(\\theta=\\pi\\): \\(e^{i\\pi} = -1\\). ✅",
        "formal_proof": "\\(e^{i\\pi} = \\sum_{n=0}^\\infty \\frac{(i\\pi)^n}{n!} = \\cos\\pi + i\\sin\\pi = -1 + 0i = -1\\). ✅",
        "applications": "Complex analysis (Cauchy's theorem), quantum mechanics (wave functions \\(\\psi=Ae^{i\\omega t}\\)), signal processing",
        "difficulty": "Advanced",
        "exam_relevance": "Complex Analysis for JAM, GATE, CSIR — direct formula applications and residue theorem"
    },
    "Cauchy-Schwarz Inequality": {
        "statement": "For vectors \\(u, v\\) in an inner product space: \\[|\\langle u, v \\rangle|^2 \\leq \\langle u, u \\rangle \\cdot \\langle v, v \\rangle\\] Equality iff \\(u, v\\) are linearly dependent.",
        "proof_sketch": "Consider \\(f(t) = \\langle u - tv, u - tv \\rangle \\geq 0\\) for all real \\(t\\). This quadratic in \\(t\\) has non-negative values, so its discriminant \\(\\leq 0\\). Expanding gives the inequality.",
        "formal_proof": "\\(0 \\leq \\|u-tv\\|^2 = \\|u\\|^2 - 2t\\langle u,v\\rangle + t^2\\|v\\|^2\\). Setting \\(t = \\frac{\\langle u,v\\rangle}{\\|v\\|^2}\\) gives \\(0 \\leq \\|u\\|^2 - \\frac{|\\langle u,v\\rangle|^2}{\\|v\\|^2}\\). ✅",
        "applications": "Linear algebra, quantum mechanics (Heisenberg uncertainty principle), statistics (correlation ≤ 1), ML (cosine similarity)",
        "difficulty": "Intermediate",
        "exam_relevance": "EXTREMELY important for JAM, GATE, CSIR — appears in linear algebra, functional analysis, probability"
    },
    "Banach Fixed Point Theorem": {
        "statement": "Let \\((X, d)\\) be a complete metric space and \\(T: X \\to X\\) a contraction (\\(d(Tx, Ty) \\leq k\\,d(x,y)\\) for \\(k < 1\\)). Then \\(T\\) has a UNIQUE fixed point.",
        "proof_sketch": "Start with any \\(x_0\\). Define \\(x_{n+1} = T(x_n)\\). The sequence is Cauchy: \\(d(x_m, x_n) \\leq \\frac{k^n}{1-k}d(x_1, x_0)\\to 0\\). By completeness it converges to some \\(x^*\\). Continuity of \\(T\\) gives \\(T(x^*) = x^*\\). ✅",
        "formal_proof": "Uniqueness: if \\(Tp=p, Tq=q\\) then \\(d(p,q) = d(Tp,Tq) \\leq k\\,d(p,q)\\), so \\((1-k)d(p,q) \\leq 0\\), hence \\(d(p,q)=0\\). ✅",
        "applications": "Newton-Raphson convergence, Picard's existence theorem for ODEs, iterative linear system solvers, fractal geometry",
        "difficulty": "Hard",
        "exam_relevance": "CSIR Part B/C, GATE — appears in functional analysis, metric spaces, ODE existence"
    },
    "Prime Number Theorem": {
        "statement": "Let \\(\\pi(x)\\) = number of primes \\(\\leq x\\). Then \\[\\pi(x) \\sim \\frac{x}{\\ln x} \\quad \\text{as } x \\to \\infty\\]",
        "proof_sketch": "Deep analytic proof via Riemann zeta function \\(\\zeta(s) = \\sum n^{-s}\\). Analytic continuation. The non-vanishing of \\(\\zeta(s)\\) on \\(\\text{Re}(s)=1\\) (proved by Hadamard and de la Vallée Poussin in 1896) is the key.",
        "formal_proof": "Proved in 1896. Key step: \\(\\zeta(1+it) \\neq 0\\) for real \\(t \\neq 0\\), proved using \\(3+4\\cos\\theta+\\cos 2\\theta \\geq 0\\).",
        "applications": "RSA key generation (prime density), randomised primality testing, cryptographic protocol design",
        "difficulty": "Very Hard",
        "exam_relevance": "CSIR Part C — concept and implications. Number theory sections in JAM/GATE."
    }
}


# ══════════════════════════════════════════════════
# COMPETITION PROBLEMS
# ══════════════════════════════════════════════════

COMPETITION_PROBLEMS = {
    "IMO": [
        {"year": 2019, "number": 1, "problem": "Determine all functions \\(f: \\mathbb{Z} \\to \\mathbb{Z}\\) such that \\[f(2a)+2f(b)=f(f(a+b))\\] for all integers \\(a, b\\).", "difficulty": "Hard", "hint": "Substituting \\(a=0,b=0\\) gives \\(f(0)+2f(0)=f(f(0))\\). Try \\(f\\equiv c\\) (constant) and \\(f(x)=2x+c\\).", "answer": "\\(f(x) = 2x+c\\) for any constant \\(c \\in \\mathbb{Z}\\), or \\(f \\equiv 0\\)."},
        {"year": 2021, "number": 2, "problem": "Show that the equation \\(x^6 + x^3 + 1 = 3y^2\\) has no integer solutions.", "difficulty": "Hard", "hint": "Consider both sides modulo 9. Cubes mod 9 are only \\(\\{0, 1, 8\\}\\).", "answer": "The LHS mod 9 is never a quadratic residue times 3 mod 9 — contradiction shows no solutions exist."},
        {"year": 2022, "number": 1, "problem": "Let \\(ABCDE\\) be a convex pentagon such that \\(BC=DE\\). Assume that there is a point \\(T\\) inside the pentagon such that \\(TB=TD\\), \\(TC=TE\\) and \\(\\angle ABT = \\angle CDT = \\angle EAT\\). Prove that line \\(AB\\) is parallel to \\(CD\\).", "difficulty": "Medium", "hint": "Use the angle condition to show spiral similarities. Triangles ABT and CDT are similar.", "answer": "The equal angle conditions imply \\(\\triangle ABT \\sim \\triangle CDT\\). This forces AB ∥ CD."}
    ],
    "PUTNAM": [
        {"year": 2020, "session": "A1", "problem": "How many positive integers \\(N\\) satisfy both: (i) \\(N\\) is a multiple of 5 (ii) The decimal representation of \\(N\\) contains no digit other than 5 or 0?", "difficulty": "Medium", "hint": "Such numbers look like 5, 50, 55, 500, 505, 550, 555, ... Count by number of digits.", "answer": "There are 31 such positive integers."},
        {"year": 2019, "session": "B3", "problem": "Let \\(f: [0,1]\\to\\mathbb{R}\\) be continuous with \\(\\int_0^1 f(x)\\,dx = 0\\). Prove: \\[\\int_0^1 f(x)^2\\,dx \\geq 3\\left(\\int_0^1 xf(x)\\,dx\\right)^2\\]", "difficulty": "Very Hard", "hint": "Use Cauchy-Schwarz with Legendre polynomials on [0,1].", "answer": "Write \\(f = c(2x-1) + g\\) where \\(\\int g = \\int g(2x-1) = 0\\). Apply Pythagoras in \\(L^2[0,1]\\)."},
        {"year": 2018, "session": "A2", "problem": "Let \\(S_1, S_2, \\ldots, S_{2^n-1}\\) be the nonempty subsets of \\(\\{1,2,\\ldots,n\\}\\). Find \\[\\sum_{i=1}^{2^n-1} (-1)^{|S_i|+1} \\frac{1}{\\max(S_i)}\\]", "difficulty": "Hard", "hint": "Group subsets by their maximum element \\(k\\).", "answer": "The sum equals \\(H_n = 1 + \\frac{1}{2} + \\cdots + \\frac{1}{n}\\), the \\(n\\)th harmonic number."}
    ],
    "AIME": [
        {"year": 2021, "number": 1, "problem": "Zou and Ceci both roll a fair six-sided die. What is the probability that they both roll the same number?", "difficulty": "Easy", "hint": "Fix Zou's roll. What is the probability Ceci matches?", "answer": "\\(\\frac{1}{6}\\)"},
        {"year": 2020, "number": 10, "problem": "There is a unique angle \\(\\theta\\) between \\(0°\\) and \\(90°\\) such that \\(\\tan(2^n\\theta)\\) is positive when \\(n\\) is a multiple of 3 and negative otherwise. Find \\(\\theta\\) as \\(p/q\\) degrees in lowest terms, and find \\(p+q\\).", "difficulty": "Hard", "hint": "Think about the angle doubling map on \\(\\mathbb{R}/180°\\mathbb{Z}\\). You need a periodic orbit of length 3.", "answer": "\\(\\theta = \\frac{400}{7}\\) degrees, so \\(p+q = 407\\)."},
        {"year": 2019, "number": 15, "problem": "Line segment \\(\\overline{AD}\\) is trisected by points \\(B\\) and \\(C\\) so that \\(AB=BC=CD=2\\). Three semicircles of radius 1 have their diameters on \\(\\overline{AD}\\). A circle of radius 2 has its center on \\(F\\). The area of the region inside the large circle but outside the three semicircles is \\(\\frac{m}{n}\\pi\\). Find \\(m+n\\).", "difficulty": "Very Hard", "hint": "Use inclusion-exclusion. Compute overlap areas of the large circle with the semicircles.", "answer": "\\(m+n = 32\\)"}
    ]
}


# ══════════════════════════════════════════════════
# LEARNING PATHS
# ══════════════════════════════════════════════════

LEARNING_PATHS = {
    "Undergraduate to PhD": {
        "overview": "BSc Mathematics → IIT JAM → MSc IIT → CSIR NET JRF → PhD → Research/Faculty",
        "total_time": "Typically 10–12 years from BSc to independent researcher",
        "salary_journey": "₹0 (student) → ₹4–6L (MSc TA) → ₹31K/month (JRF) → ₹70K/month (SRF) → ₹60–150L (faculty)",
        "stages": [
            {"name": "Bachelor's Degree (3–4 years)", "goal": "Build solid foundation in core mathematics", "subjects": ["Calculus & Analysis", "Linear Algebra", "Abstract Algebra", "Complex Analysis", "Topology", "Differential Equations"], "tips": "Maintain 70%+ marks. Start reading classic texts (Rudin, Artin) from 2nd year.", "outcome": "BSc Mathematics degree"},
            {"name": "IIT JAM Preparation (6–12 months)", "goal": "Clear IIT JAM for MSc at IIT/IISc", "subjects": ["Real Analysis (25%)", "Linear Algebra (20%)", "Calculus (20%)", "Group Theory (15%)", "Statistics (10%)"], "tips": "Solve last 15 years PYQs. Focus on Real Analysis and LA first. Do 3 mock tests per week in final 2 months.", "outcome": "Admission to IIT/IISc MSc"},
            {"name": "MSc at IIT/IISc (2 years)", "goal": "Master advanced mathematics and begin research exposure", "subjects": ["Functional Analysis", "Topology", "PDE", "Numerical Analysis", "Algebra", "Research Project"], "tips": "Attend seminars. Meet professors early. Start reading research papers. Aim for CGPA > 8.", "outcome": "MSc degree + research exposure"},
            {"name": "CSIR-UGC NET JRF (during/after MSc)", "goal": "Get fellowship for PhD funding", "subjects": ["All MSc topics", "Topology", "Functional Analysis", "Complex Analysis", "Part C proof-writing"], "tips": "CSIR Part C is the key differentiator — practice proof writing. Joint JRF gives ₹31,000/month.", "outcome": "JRF fellowship (₹31K/month) + Lectureship eligibility"},
            {"name": "PhD Research (4–5 years)", "goal": "Produce original mathematical research", "subjects": ["Specialised coursework", "Research problem", "Paper writing", "Conference presentations"], "tips": "Choose advisor carefully. Publish at least 2 papers. Collaborate internationally.", "outcome": "PhD degree + published research"},
            {"name": "Postdoc / Faculty / Industry", "goal": "Independent career in mathematics", "subjects": ["Independent research", "Grants", "Teaching", "Collaboration"], "tips": "Apply to 20+ positions. TIFR, IIT, IISER, ICTS are top choices.", "outcome": "Faculty (₹80–150L CTC) or Industry (₹60–200L CTC)"}
        ]
    },
    "JAM Fast Track": {
        "overview": "Focused 1-year JAM preparation for working/final-year students",
        "total_time": "8–12 months intensive preparation",
        "salary_journey": "₹0 → IIT MSc → ₹31K JRF → ₹40–80L career",
        "stages": [
            {"name": "Month 1–2: Foundation", "goal": "Revise BSc syllabus systematically", "subjects": ["Real Analysis basics", "Linear Algebra", "Calculus", "Group Theory fundamentals"], "tips": "Make formula cards. Solve 20 problems per day minimum.", "outcome": "Solid conceptual foundation"},
            {"name": "Month 3–5: Topic Mastery", "goal": "Deep dive into high-weightage topics", "subjects": ["Real Analysis (sequences, series, continuity, Riemann integral)", "Linear Algebra (all topics)", "Complex Analysis basics"], "tips": "Solve Arora & Sharma JAM book completely.", "outcome": "Mastery of 80% of JAM syllabus"},
            {"name": "Month 6–8: Previous Year Papers", "goal": "Analyse and solve last 15 years JAM PYQs", "subjects": ["Timed PYQ practice", "Error analysis", "Weak area revision"], "tips": "Note every mistake. Revise that topic immediately.", "outcome": "Pattern recognition and exam readiness"},
            {"name": "Month 9–12: Mock Tests & Revision", "goal": "Exam simulation and final preparation", "subjects": ["Full mock tests (weekly)", "Rapid revision", "Last-minute formula sheets"], "tips": "Take test in exam-like conditions. Aim for 70+ marks.", "outcome": "JAM score → IIT MSc admission"}
        ]
    },
    "GATE Mathematics Track": {
        "overview": "GATE MA → PSU jobs / IIT MTech / research positions",
        "total_time": "1 year prep → career of 30+ years",
        "salary_journey": "₹0 → ₹15–20L (PSU/MTech entry) → ₹30–50L (5 years) → ₹60–100L (10 years)",
        "stages": [
            {"name": "GATE Preparation (8–12 months)", "goal": "Score 650+ in GATE MA", "subjects": ["Calculus (30%)", "Linear Algebra (20%)", "Complex Analysis (15%)", "Probability (15%)", "Numerical Analysis (10%)", "ODE/PDE (10%)"], "tips": "GATE has NAT questions — no negative marking for these. Use NPTEL courses.", "outcome": "GATE score for PSU / IIT MTech"},
            {"name": "Entry Level (2–5 years)", "goal": "Build technical expertise", "subjects": ["Domain-specific mathematics", "Programming (Python/R)", "Data Analysis"], "tips": "PSUs: BHEL, GAIL, ONGC hire GATE scorers. ₹15–20L starting.", "outcome": "Career with ₹15–25L CTC"},
            {"name": "Mid-Career Growth (5–10 years)", "goal": "Senior technical or managerial roles", "subjects": ["Advanced analytics", "Team leadership", "Specialisation"], "tips": "Switch to data science / quant finance for high salaries.", "outcome": "₹30–60L CTC in core or ₹60–100L in data/quant"}
        ]
    }
}

EXAM_FORMULAS = {
    "Calculus": {
        "JAM": [
            "Derivative: \\(f'(x) = \\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}\\)",
            "Product Rule: \\((uv)' = u'v + uv'\\)",
            "Quotient Rule: \\(\\left(\\frac{u}{v}\\right)' = \\frac{u'v-uv'}{v^2}\\)",
            "Chain Rule: \\(\\frac{dy}{dx} = \\frac{dy}{du}\\cdot\\frac{du}{dx}\\)",
            "Power Rule: \\(\\int x^n\\,dx = \\frac{x^{n+1}}{n+1}+C\\quad (n\\neq-1)\\)",
            "Integration by Parts: \\(\\int u\\,dv = uv - \\int v\\,du\\)",
            "MVT: \\(f'(c) = \\frac{f(b)-f(a)}{b-a}\\) for some \\(c\\in(a,b)\\)",
            "FTC: \\(\\int_a^b f'(x)\\,dx = f(b)-f(a)\\)",
            "Taylor: \\(f(x) = \\sum_{n=0}^\\infty \\frac{f^{(n)}(a)}{n!}(x-a)^n\\)",
        ],
        "GATE": [
            "Green's Theorem: \\(\\oint_C(P\\,dx+Q\\,dy)=\\iint_D\\left(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\right)dA\\)",
            "Stokes: \\(\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}=\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}\\)",
            "Lagrange Multipliers: \\(\\nabla f = \\lambda\\nabla g\\) at constrained extrema",
        ]
    }
}

REALWORLD = [
    {"concept": "Fourier Transform", "application": "MRI Medical Imaging", "explanation": "MRI scanners record raw k-space data (Fourier-encoded radio frequency signals from hydrogen atoms). The Fourier Transform reconstructs this into detailed 3D anatomical images in milliseconds.", "companies": "Siemens Healthineers, GE Healthcare, Philips, Canon Medical", "impact": "Diagnoses cancer, brain tumours, spinal injuries without ionising radiation. ~100M MRI scans performed globally per year.", "salary": "Biomedical Engineer $105K+ | MRI Physicist $120K+"},
    {"concept": "Eigenvalues and Linear Algebra", "application": "Google PageRank", "explanation": "Google models the web as a directed graph. PageRank is the principal eigenvector of a massive stochastic matrix. The power iteration method computes it iteratively.", "companies": "Google, Microsoft Bing, DuckDuckGo, Baidu", "impact": "Handles 8.5 billion searches per day.", "salary": "Search Quality Engineer $145K+ | Ranking Scientist $160K+"},
    {"concept": "Differential Equations", "application": "COVID-19 Pandemic Modelling", "explanation": "Governments used SIR/SEIR systems of ODEs to model disease spread. Parameters: β (transmission rate), γ (recovery rate). R₀ = β/γ determines epidemic vs endemic.", "companies": "WHO, CDC, NHS, ICMR, McKinsey Health", "impact": "Directly shaped lockdown decisions affecting billions. Estimated to have saved 10–50 million lives.", "salary": "Epidemiologist $90K+ | Public Health Analyst $100K+"},
    {"concept": "Number Theory and RSA", "application": "Internet Security (HTTPS/TLS)", "explanation": "RSA encryption relies on the hardness of factoring large semiprime numbers. If p, q are large primes, computing p×q is easy but recovering p, q from N=pq is infeasible.", "companies": "Apple, Google, Amazon, all banks, NSA", "impact": "Protects every online transaction. Global e-commerce ($6T+/year) depends on it.", "salary": "Cryptographer $115K+ | Security Engineer $135K+"},
    {"concept": "Optimisation Calculus", "application": "Quantitative Finance", "explanation": "Markowitz mean-variance optimisation uses Lagrange multipliers to find minimum-variance portfolio for a given expected return. The efficient frontier is a parabola in (σ, μ) space.", "companies": "Goldman Sachs, Citadel, Renaissance Technologies, BlackRock", "impact": "Controls $100+ trillion in global financial assets.", "salary": "Quantitative Analyst $160K+ | Portfolio Manager $300K+"},
]

RESEARCH_HUB = {
    "Pure Mathematics": ["Analytic Number Theory and the Riemann Hypothesis", "Abstract Algebra: Groups, Rings, Fields and Galois Theory", "Algebraic Topology and Homotopy Theory", "Differential Geometry and Riemannian Manifolds", "Algebraic Geometry (Schemes, Sheaves, Cohomology)", "Category Theory and Homological Algebra", "Mathematical Logic, Model Theory and Set Theory", "Representation Theory of Lie Groups"],
    "Applied Mathematics": ["Numerical Methods for PDEs (FEM, FDM, Spectral Methods)", "Convex and Non-Convex Optimisation Theory", "Dynamical Systems, Ergodic Theory and Chaos", "Fluid Dynamics (Navier-Stokes, Turbulence)", "Mathematical Biology (Reaction-Diffusion, Population Dynamics)", "Financial Mathematics (Stochastic Calculus, Black-Scholes)", "Control Theory and Optimal Control", "Mathematical Imaging and Compressed Sensing"],
    "Probability and Statistics": ["Stochastic Processes and Brownian Motion", "Statistical Learning Theory and PAC Learning", "Bayesian Non-Parametrics and Gaussian Processes", "High-Dimensional Statistics and Random Matrix Theory", "Causal Inference and Potential Outcomes", "Information Theory (Shannon Entropy, Channel Capacity)", "Extreme Value Theory and Heavy-Tailed Distributions", "Spatial Statistics and Geostatistics"],
    "Computational Mathematics": ["Quantum Algorithms (Shor, Grover, HHL)", "Algorithmic Game Theory and Mechanism Design", "Compressed Sensing and Sparse Recovery", "Topological Data Analysis and Persistent Homology", "Geometric Deep Learning (Graph Neural Networks)", "Scientific Machine Learning (Physics-Informed NNs)", "High Performance Computing and Parallel Algorithms", "Symbolic Computation and Computer Algebra Systems"],
    "Analysis and Geometry": ["Harmonic Analysis and Wavelets", "Partial Differential Equations (Existence, Regularity)", "Functional Analysis and Operator Algebras (C*-algebras)", "Complex Analysis in Several Variables", "Symplectic Geometry and Mirror Symmetry", "Sub-Riemannian Geometry", "Geometric Measure Theory", "Metric Geometry and Alexandrov Spaces"]
}


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())

def parse_json_block(raw: str):
    if not raw: return None
    raw = raw.strip()
    try: return json.loads(raw)
    except Exception: pass
    m = re.search(r"```json\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", raw)
    if m:
        try: return json.loads(m.group(1))
        except Exception: return None
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try: return json.loads(m.group(1))
        except Exception: return None
    return None

def ai_dynamic_mathematicians(count=18):
    prompt = f"""Return ONLY valid JSON array with {count} mathematicians from diverse eras and countries.
Each item keys: name, period, country, fields (array of 3-5), contribution, keyresults, quote, image, impact, resources (array of 'Label: URL').
Use real public links. No markdown, no extra text."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data: return data
    return []

def ai_dynamic_projects(count=24):
    prompt = f"""Return ONLY valid JSON array with {count} mathematics projects useful for students.
Each item keys: title, difficulty, math (array), desc, real, companies, salary, links (array of 'Label: URL').
No markdown, no extra text."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data: return data
    return []


# ══════════════════════════════════════════════════
# ROUTES — Original
# ══════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "groq": GROQ_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "sympy": SYMPY_AVAILABLE,
        "mathematicians": len(MATHEMATICIANS),
        "projects": len(MATH_PROJECTS),
        "theorems": len(THEOREMS)
    })

# ── Chat ─────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data       = request.get_json()
        messages   = data.get("messages", [])
        mode       = str(data.get("mode", "normal"))
        image_b64  = data.get("image_b64")
        image_type = data.get("image_type")
        if not messages:
            return jsonify({"error": "messages required"}), 400
        clean = [{"role": m["role"], "content": str(m["content"])}
                 for m in messages if m.get("role") in ("user","assistant") and m.get("content")]
        if len(clean) > 16: clean = clean[-14:]
        system = ASK_ANUPAM_PROMPT if mode == "ask_anupam" else SYSTEM_PROMPT
        answer = ask_ai_with_image(clean, image_b64=image_b64, image_type=image_type, system=system)

        # Extract confidence level
        confidence = "HIGH"
        if "[CONFIDENCE: MEDIUM]" in answer:
            confidence = "MEDIUM"
        elif "[CONFIDENCE: LOW]" in answer:
            confidence = "LOW"
        answer = re.sub(r'\[CONFIDENCE: (HIGH|MEDIUM|LOW)\]', '', answer).strip()

        return jsonify({"answer": answer, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════
# NEW ROUTES — SymPy Verification Engine
# ══════════════════════════════════════════════════

@app.route("/api/sympy/compute", methods=["POST"])
def sympy_compute_route():
    """Direct SymPy computation endpoint."""
    if not SYMPY_AVAILABLE:
        return jsonify({"error": "SymPy not available on this server. Install: pip install sympy"}), 503
    data = request.get_json()
    problem_type = data.get("type", "simplify")
    expression   = data.get("expression", "")
    variable     = data.get("variable", "x")
    extra        = data.get("extra", {})
    if not expression:
        return jsonify({"error": "expression required"}), 400
    result = sympy_compute(problem_type, expression, variable, extra)
    return jsonify(result)


@app.route("/api/verify/answer", methods=["POST"])
def verify_answer():
    """
    Verify a student's answer against a problem.
    Three-layer check: SymPy symbolic → numerical → AI explanation.
    """
    data            = request.get_json()
    problem         = data.get("problem", "")
    student_answer  = data.get("student_answer", "")
    problem_type    = data.get("problem_type", "")  # differentiate, integrate, solve, etc.
    expression      = data.get("expression", "")
    variable        = data.get("variable", "x")

    result = {
        "sympy_verified": None,
        "sympy_result": None,
        "sympy_steps": [],
        "numerical_verified": None,
        "ai_explanation": "",
        "final_verdict": "UNVERIFIED",
        "correct_answer_latex": "",
        "error_location": ""
    }

    # Layer 1: SymPy symbolic check
    if SYMPY_AVAILABLE and expression and problem_type:
        try:
            computed = sympy_compute(problem_type, expression, variable)
            if computed.get("success"):
                result["correct_answer_latex"] = computed["result_latex"]
                result["sympy_steps"]          = computed.get("steps", [])

                # Verify student answer against computed
                correct_expr_str = computed["result_str"].replace(" + C", "").strip()
                verify = sympy_compute("verify_answer", expression, variable, {
                    "student_answer": student_answer,
                    "computed": correct_expr_str
                })
                if verify.get("success"):
                    result["sympy_verified"] = verify.get("is_correct", False)

                # Layer 2: Numerical check for calculus
                if problem_type == "integrate":
                    f_str   = expression
                    F_str   = student_answer
                    num_check = verify_integral(f_str, F_str, variable)
                    result["numerical_verified"] = num_check.get("verified")
                    result["numerical_details"]  = num_check

                elif problem_type == "differentiate":
                    df_str    = student_answer
                    num_check = numerical_verify_derivative(expression, df_str, variable)
                    result["numerical_verified"] = num_check.get("verified")
                    result["numerical_details"]  = num_check

        except Exception as e:
            result["sympy_error"] = str(e)

    # Determine verdict
    sv = result["sympy_verified"]
    nv = result["numerical_verified"]
    if sv is True or nv is True:
        result["final_verdict"] = "CORRECT"
    elif sv is False or nv is False:
        result["final_verdict"] = "INCORRECT"
    else:
        result["final_verdict"] = "UNVERIFIED"

    # Layer 3: AI explanation
    prompt = f"""Problem: {problem}
Student's answer: {student_answer}
SymPy computed answer: {result.get('correct_answer_latex', 'N/A')}
Symbolic verification: {result['sympy_verified']}
Numerical verification: {result['numerical_verified']}

Provide:
1. Clear verdict: CORRECT or INCORRECT
2. If incorrect: exactly where the student went wrong (step by step)
3. The correct complete solution with all steps shown
4. The boxed final answer: \\[\\boxed{{...}}\\]
5. One key tip to avoid this error in future

Use LaTeX for all math. Be precise, not vague."""

    result["ai_explanation"] = ask_simple(prompt, system=VERIFIER_PROMPT)
    return jsonify(result)


@app.route("/api/verify/steps", methods=["POST"])
def verify_steps():
    """Validate a student's step-by-step solution."""
    data            = request.get_json()
    problem         = data.get("problem", "")
    student_steps   = data.get("steps", [])  # list of step strings

    if not student_steps:
        return jsonify({"error": "steps required"}), 400

    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(student_steps)])

    prompt = f"""A student is solving this problem:
{problem}

Their step-by-step work:
{steps_text}

Analyse EVERY step carefully:
1. For each step: state CORRECT or ERROR with specific reason
2. If ERROR: show the correct version of that step with full LaTeX working
3. Classify each error: [CONCEPTUAL | ARITHMETIC | SIGN | NOTATION | MISSING STEP | WRONG RULE]
4. Give overall score: X/{len(student_steps)} steps correct
5. Final correct solution from scratch
6. Top 2 things this student needs to review

Format each step analysis as:
Step N: [CORRECT / ERROR — Error type]
[Explanation with LaTeX if needed]"""

    analysis = ask_simple(prompt, system=ERROR_ANALYSIS_PROMPT)

    # Quick SymPy check if first and last steps parseable
    sympy_checks = []
    if SYMPY_AVAILABLE and len(student_steps) >= 2:
        for i, step in enumerate(student_steps):
            eq_match = re.search(r'=\s*(.+)$', step.strip())
            if eq_match:
                expr_str = eq_match.group(1).strip()
                simplified = safe_parse(expr_str)
                if simplified is not None:
                    sympy_checks.append({
                        "step": i + 1,
                        "expression": expr_str,
                        "sympy_parsed": sp_latex(simplified) if SYMPY_AVAILABLE else expr_str
                    })

    return jsonify({
        "analysis": analysis,
        "sympy_checks": sympy_checks,
        "total_steps": len(student_steps)
    })


@app.route("/api/verify/check", methods=["POST"])
def quick_check():
    """
    Quick answer checker — student says 'I got X for problem Y, is it right?'
    """
    data           = request.get_json()
    problem        = data.get("problem", "")
    student_answer = data.get("answer", "")

    if not problem or not student_answer:
        return jsonify({"error": "problem and answer required"}), 400

    # Try to auto-detect and compute with SymPy
    math_type, expr, extra = detect_math_type(problem)
    sympy_result = None
    correct_latex = ""

    if SYMPY_AVAILABLE and math_type and expr:
        variable = extra.get("variable", "x")
        computed = sympy_compute(math_type, expr, variable, extra)
        if computed.get("success"):
            sympy_result = computed
            correct_latex = computed["result_latex"]

    # Ask AI to verify with context
    context = f"\nSymPy computed answer: \\({correct_latex}\\)" if correct_latex else ""
    prompt = f"""Student problem: {problem}
Student's answer: {student_answer}{context}

Give:
1. VERDICT: CORRECT / INCORRECT / PARTIALLY CORRECT
2. If wrong: exact error and correct answer \\[\\boxed{{...}}\\]
3. If correct: confirm and add one interesting insight
4. Confidence in your verdict: HIGH / MEDIUM / LOW

Keep response concise but mathematically precise. All math in LaTeX."""

    ai_response = ask_simple(prompt, system=VERIFIER_PROMPT)

    verdict = "UNVERIFIED"
    if "CORRECT" in ai_response.upper() and "INCORRECT" not in ai_response.upper():
        verdict = "CORRECT"
    elif "INCORRECT" in ai_response.upper():
        verdict = "INCORRECT"
    elif "PARTIALLY" in ai_response.upper():
        verdict = "PARTIAL"

    return jsonify({
        "verdict": verdict,
        "correct_answer_latex": correct_latex,
        "sympy_steps": sympy_result.get("steps", []) if sympy_result else [],
        "ai_explanation": ai_response,
        "sympy_available": SYMPY_AVAILABLE and sympy_result is not None
    })


@app.route("/api/analyze/error", methods=["POST"])
def analyze_error():
    """Deep error analysis — identify exactly what went wrong in a student's solution."""
    data           = request.get_json()
    problem        = data.get("problem", "")
    wrong_solution = data.get("wrong_solution", "")

    if not problem or not wrong_solution:
        return jsonify({"error": "problem and wrong_solution required"}), 400

    prompt = f"""Problem: {problem}

Student's complete (wrong) solution:
{wrong_solution}

Provide a detailed error report:
1. SUMMARY: What is the main conceptual/procedural issue?
2. ERROR BREAKDOWN: Go through each step and mark each as CORRECT or WRONG
3. FIRST ERROR: Identify the very first step that goes wrong
4. ERROR TYPE: [Conceptual | Arithmetic | Sign | Chain rule | Product rule | Integration constant | Other]
5. CORRECT SOLUTION: Solve it correctly from scratch, showing every step
6. CORRECT ANSWER: \\[\\boxed{{...}}\\]
7. PREVENTION TIP: How to avoid this mistake

All math in LaTeX. Be a professor, not a cheerleader — be precise."""

    analysis = ask_simple(prompt, system=ERROR_ANALYSIS_PROMPT)
    return jsonify({"analysis": analysis})


@app.route("/api/sympy/graph", methods=["POST"])
def sympy_graph():
    """Generate data points for graphing a function."""
    if not SYMPY_AVAILABLE:
        return jsonify({"error": "SymPy not available"}), 503

    data       = request.get_json()
    f_str      = data.get("function", "x**2")
    x_min      = float(data.get("x_min", -10))
    x_max      = float(data.get("x_max", 10))
    num_points = min(int(data.get("points", 200)), 500)

    try:
        x    = Symbol("x")
        f    = safe_parse(f_str)
        if f is None:
            return jsonify({"error": "Could not parse function"}), 400

        points = []
        step   = (x_max - x_min) / num_points
        for i in range(num_points + 1):
            xv = x_min + i * step
            try:
                yv = float(f.subs(x, xv))
                if abs(yv) < 1e10:  # filter asymptotes
                    points.append({"x": round(xv, 6), "y": round(yv, 6)})
                else:
                    points.append({"x": round(xv, 6), "y": None})
            except Exception:
                points.append({"x": round(xv, 6), "y": None})

        # Also compute derivative and critical points
        df         = diff(f, x)
        df_str     = sp_latex(simplify(df))
        try:
            crit = solve(df, x)
            crit_real = [float(c) for c in crit if c.is_real and x_min <= float(c) <= x_max]
        except Exception:
            crit_real = []

        return jsonify({
            "points": points,
            "function_latex": sp_latex(f),
            "derivative_latex": df_str,
            "critical_points": crit_real,
            "x_min": x_min,
            "x_max": x_max
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── All original routes ───────────────────────────

@app.route("/api/mathematician")
def mathematician_random():
    dyn = ai_dynamic_mathematicians(count=12)
    if dyn:
        return jsonify(random.choice(dyn))
    name, d = random.choice(list(MATHEMATICIANS.items()))
    return jsonify({"name": name, **d})

@app.route("/api/mathematicians")
def mathematician_list():
    dyn = ai_dynamic_mathematicians(count=18)
    if dyn:
        slim = [{"name": x.get("name","Unknown"), "period": x.get("period","N/A"), "country": x.get("country","N/A"), "fields": x.get("fields",[])} for x in dyn]
        return jsonify({"mathematicians": slim, "total": len(slim), "source": "dynamic"})
    return jsonify({"mathematicians": [{"name": n, "period": d["period"], "country": d["country"], "fields": d["fields"]} for n, d in MATHEMATICIANS.items()], "total": len(MATHEMATICIANS), "source": "local"})

@app.route("/api/mathematician/<name>")
def mathematician_detail(name):
    q = normalize_name(name)
    for n, d in MATHEMATICIANS.items():
        if q and (q in normalize_name(n) or normalize_name(n) in q):
            return jsonify({"name": n, **d, "source": "local"})
    prompt = f"""Return ONLY valid JSON for mathematician: {name}
Keys: name, period, country, fields (array), contribution, keyresults, quote, image, impact, resources (array of 'Label: URL')."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, dict) and data.get("name"):
        data["source"] = "dynamic"
        return jsonify(data)
    return jsonify({"error": "Not found"}), 404

DYNAMIC_PROJECTS_CACHE = []

@app.route("/api/projects")
def projects_list():
    global DYNAMIC_PROJECTS_CACHE
    dyn = ai_dynamic_projects(count=24)
    if dyn:
        DYNAMIC_PROJECTS_CACHE = [{"id": i, **p} for i, p in enumerate(dyn, start=1)]
        return jsonify({"projects": DYNAMIC_PROJECTS_CACHE, "total": len(DYNAMIC_PROJECTS_CACHE), "source": "dynamic"})
    return jsonify({"projects": MATH_PROJECTS, "total": len(MATH_PROJECTS), "source": "local"})

@app.route("/api/project/<int:pid>", methods=["POST"])
def project_detail(pid):
    source = DYNAMIC_PROJECTS_CACHE if DYNAMIC_PROJECTS_CACHE else MATH_PROJECTS
    p = next((x for x in source if int(x.get("id", -1)) == pid), None)
    if not p: return jsonify({"error": "Not found"}), 404
    links = ", ".join(p.get("links", [])) if isinstance(p.get("links"), list) else ""
    prompt = f"""Explain this maths project for students.
Project: {p.get('title')}
Math topics: {', '.join(p.get('math', []))}
Description: {p.get('desc')}
Real-world use: {p.get('real')}

Give Step 1, Step 2, Step 3... implementation and important formulas in LaTeX."""
    return jsonify({"project": p, "explanation": ask_simple(prompt, system=ASK_ANUPAM_PROMPT)})

@app.route("/api/theorems")
def theorems_list():
    return jsonify({"theorems": list(THEOREMS.keys()), "total": len(THEOREMS)})

@app.route("/api/theorem/<name>")
def theorem_detail(name):
    for n, d in THEOREMS.items():
        if name.lower() in n.lower():
            return jsonify({"name": n, **d})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/competition/<cat>")
def competition(cat):
    probs = COMPETITION_PROBLEMS.get(cat.upper(), [])
    if not probs: return jsonify({"error": "Category not found. Use IMO, PUTNAM, or AIME"}), 404
    return jsonify({"category": cat.upper(), "problems": probs, "total": len(probs)})

@app.route("/api/learning-paths")
def learning_paths_list():
    return jsonify({"paths": [{"name": n, "overview": d["overview"]} for n, d in LEARNING_PATHS.items()], "total": len(LEARNING_PATHS)})

@app.route("/api/learning-path/<name>")
def learning_path_detail(name):
    for n, d in LEARNING_PATHS.items():
        if name.lower().replace("-"," ") in n.lower():
            return jsonify({"name": n, **d})
    return jsonify({"error": "Not found"}), 404

@app.route("/api/formula", methods=["POST"])
def formula():
    data  = request.get_json()
    topic = data.get("topic", "Calculus")
    exam  = data.get("exam",  "JAM")
    stored = EXAM_FORMULAS.get(topic, {}).get(exam, [])
    stored_block = ("\n\nKNOWN FORMULAS TO INCLUDE:\n" + "\n".join(f"• {f}" for f in stored)) if stored else ""
    prompt = f"""Generate a COMPLETE, exam-ready formula sheet for: {topic}
Target exam: {exam}{stored_block}

For each formula:
📌 [Formula Name]
\\[ LaTeX formula \\]
When to use: [1 sentence]
Condition: [any restrictions]
Exam tip: [common mistake or trick]

Include AT LEAST 18 formulas. ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic", "Calculus")
    prompt = f"""Give exactly 10 RAPID REVISION POINTS for: {topic} (graduate exam level)

For each:
[N]. [TOPIC IN CAPS]
Definition: [with LaTeX]
Key formula: \\[ ... \\]
Exam trap: [common wrong answer]
Memory trick: [how to remember]

ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic", "Calculus")
    prompt = f"""Create a DEEP concept map for: {topic}

📌 Core Definition + LaTeX
💡 Key Sub-concepts (6-8) each with LaTeX + intuition
📐 How they connect (arrows with reason)
⭐ Top 5 theorems / results
🌍 3 real-world applications
📚 Prerequisites and what builds on this

ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/latex", methods=["POST"])
def latex_gen():
    text = request.get_json().get("text", "")
    prompt = f"""Generate professional LaTeX code for: {text}

Include:
1. Complete compilable code snippet
2. Explanation of each command
3. How to compile (pdflatex or overleaf)
4. Alternative simpler version if complex"""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    d = request.get_json()
    prompt = f"""Generate ONE rigorous MCQ for topic: {d.get('topic','Calculus')}
Difficulty: {d.get('difficulty','medium')}
Question {d.get('q_num',1)} of {d.get('total',5)}

EXACT FORMAT:
Q: [question text with LaTeX]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter A/B/C/D]
EXPLANATION: [full step-by-step solution with LaTeX]

ALL math in LaTeX."""
    raw = ask_simple(prompt)
    lines = raw.strip().split("\n")
    ans_line = next((l for l in lines if l.strip().startswith("ANSWER:")), "ANSWER: A")
    ans = ans_line.replace("ANSWER:", "").strip()[:1].upper() or "A"
    explanation = ""
    if "EXPLANATION:" in raw:
        explanation = raw.split("EXPLANATION:", 1)[1].strip()
    question = "\n".join(l for l in lines if not l.strip().startswith("ANSWER:"))
    question = question.split("EXPLANATION:", 1)[0].strip()
    return jsonify({"question": question.strip(), "answer": ans, "explanation": explanation})

@app.route("/api/pyq")
def pyq():
    exam   = request.args.get("exam", "JAM")
    topics = {"JAM": ["Real Analysis","Linear Algebra","Calculus","Group Theory","Complex Analysis"], "GATE": ["Calculus","Linear Algebra","Complex Analysis","PDE","ODE"], "CSIR": ["Real Analysis","Topology","Algebra","Functional Analysis","Complex Analysis"]}
    topic  = random.choice(topics.get(exam, topics["JAM"]))
    year   = random.randint(2014, 2023)
    prompt = f"""Generate a realistic {exam} PYQ for {topic} (~year {year}).

FORMAT:
Question: [challenging problem with LaTeX]
Solution: [complete step-by-step with ALL LaTeX formulas]
Key Concept: [theorem tested]
Exam Tip: [approach for similar questions]

ALL math in LaTeX."""
    raw   = ask_simple(prompt, system=SYSTEM_PROMPT)
    lines = raw.split('\n')
    q = next((l.replace("Question:","").strip() for l in lines if l.startswith("Question:")), raw[:300])
    a = next((l.replace("Solution:","").strip()  for l in lines if l.startswith("Solution:")),  "See full answer above.")
    return jsonify({"q": q, "a": a, "topic": topic, "year": year, "exam": exam})

@app.route("/api/challenge")
def challenge():
    challenges = [
        "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
        "Find all critical points of \\(f(x)=x^3-3x+2\\) and classify them using the second derivative test.",
        "Compute the eigenvalues and eigenvectors of \\(A=\\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}\\).",
        "Evaluate \\(\\int x^2 e^x\\,dx\\) fully using integration by parts.",
        "Solve \\(\\frac{dy}{dx}+2y=4x\\) with initial condition \\(y(0)=1\\).",
        "Prove the Cauchy-Schwarz inequality in \\(\\mathbb{R}^n\\).",
        "Show every convergent sequence is a Cauchy sequence.",
        "Compute \\(\\sum_{n=1}^\\infty\\frac{1}{n^2}\\) using Fourier series of \\(f(x)=x\\) on \\([-\\pi,\\pi]\\).",
        "Prove \\(\\text{tr}(AB)=\\text{tr}(BA)\\) for any \\(n\\times n\\) matrices \\(A, B\\).",
        "Prove the intermediate value theorem using the completeness of \\(\\mathbb{R}\\).",
        "Find the radius of convergence of \\(\\sum_{n=0}^\\infty \\frac{n!}{n^n} x^n\\).",
        "Prove that a continuous function on a closed bounded interval attains its maximum.",
    ]
    return jsonify({"challenge": random.choice(challenges)})

@app.route("/api/realworld")
def realworld_random():
    return jsonify(random.choice(REALWORLD))

@app.route("/api/realworld/<concept>")
def realworld_detail(concept):
    for item in REALWORLD:
        if concept.lower() in item["concept"].lower():
            return jsonify(item)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/research-hub")
def research_hub_index():
    return jsonify({"categories": list(RESEARCH_HUB.keys()), "total_topics": sum(len(v) for v in RESEARCH_HUB.values())})

@app.route("/api/research-hub/<cat>")
def research_hub_cat(cat):
    for k, v in RESEARCH_HUB.items():
        if cat.lower() in k.lower():
            return jsonify({"category": k, "topics": v, "total": len(v)})
    return jsonify({"error": "Category not found"}), 404

@app.route("/api/research", methods=["POST"])
def research_question():
    question = request.get_json().get("question", "")
    prompt = f"""Answer this research mathematics question in depth: {question}

📌 Overview of the research area
🔬 Current state (key open problems)
📐 Core mathematical tools + theorems with LaTeX
💡 Key researchers to follow
📚 Recommended papers and textbooks
🚀 How a student can get started

ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/books", methods=["POST"])
def books_search():
    d = request.get_json() or {}
    topic = (d.get("topic") or "Mathematics").strip()
    exam = (d.get("exam") or "").strip()
    prompt = f"""Return ONLY valid JSON array of best books for topic: {topic}. Exam: {exam or 'General'}.
Each item: name, author, level, why, link. 8-12 items. Real links only."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data:
        return jsonify({"books": data, "total": len(data), "source": "dynamic"})
    return jsonify({"books": [], "total": 0, "source": "none"})

@app.route("/api/exam/<exam>")
def exam_info(exam):
    info = {
        "JAM": {"full_name": "IIT JAM Mathematics", "conducting_body": "IITs (rotational)", "eligibility": "Bachelor's degree with Mathematics in at least 2 years", "pattern": "3 hours · 60 questions · 100 marks\nSection A: 30 MCQ (1 & 2 marks, ⅓ negative)\nSection B: 10 MSQ (2 marks, NO negative)\nSection C: 20 NAT (1 & 2 marks, NO negative)", "syllabus": "Real Analysis · Linear Algebra · Calculus · Differential Equations · Group Theory · Complex Analysis · Numerical Analysis · Statistics & Probability", "weightage": "Real Analysis 25% · Linear Algebra 20% · Calculus 20% · Group Theory 15% · Statistics 10%", "top_books": ["Rudin — Principles of Mathematical Analysis","Artin — Algebra","Churchill — Complex Variables","Apostol — Calculus Vol 1 & 2"], "strategy": "Solve 15 years PYQs. Strong in Real Analysis = 60% of rank. Take 1 full mock test per week in last 3 months.", "website": "https://jam.iitd.ac.in"},
        "GATE": {"full_name": "GATE Mathematics (MA)", "conducting_body": "IITs / IISc (rotational)", "eligibility": "Bachelor's in Mathematics/Statistics/CS or related", "pattern": "3 hours · 65 questions · 100 marks\nGeneral Aptitude: 15 marks\nCore MA: 85 marks (MCQ + MSQ + NAT)", "syllabus": "Calculus · Linear Algebra · Real Analysis · Complex Analysis · ODE · PDE · Abstract Algebra · Functional Analysis · Numerical Analysis · Probability", "weightage": "Calculus + LA: 30% · Real Analysis: 20% · Complex: 15% · Algebra: 15%", "top_books": ["Apostol — Calculus","Hoffman-Kunze — Linear Algebra","Conway — Complex Analysis","Dummit-Foote — Abstract Algebra"], "strategy": "NAT questions have no negative marking — attempt all. NPTEL videos are excellent.", "website": "https://gate.iitd.ac.in"},
        "CSIR": {"full_name": "CSIR UGC NET Mathematics", "conducting_body": "NTA (National Testing Agency)", "eligibility": "Master's in Mathematics with 55% (50% SC/ST/PwD)", "pattern": "3 hours · 200 marks total\nPart A: 20Q (General Science, 30 marks)\nPart B: 40Q (Core math, 70 marks, ⅓ negative)\nPart C: 60Q (Advanced, 100 marks, ⅓ negative, PROOF-BASED)", "syllabus": "Analysis (Real+Complex+Functional) · Algebra (Linear+Abstract) · Topology · ODE · PDE · Numerical Methods · Probability", "weightage": "Analysis: 30% · Algebra: 25% · Complex: 20% · Topology: 10%", "top_books": ["Rudin — Real & Complex Analysis","Dummit-Foote — Abstract Algebra","Munkres — Topology","Conway — Functions of One Complex Variable"], "strategy": "Part C is the key differentiator. Master proof writing. JRF = ₹31,000/month for PhD.", "website": "https://csirnet.nta.nic.in"}
    }
    return jsonify(info.get(exam, {"error": "Not found. Use JAM, GATE, or CSIR"}))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🧮 MathSphere v6.0 Professor Edition — port {port}")
    print(f"   ✅ SymPy: {SYMPY_AVAILABLE}")
    print(f"   ✅ Groq: {GROQ_AVAILABLE}")
    print(f"   ✅ Gemini: {GEMINI_AVAILABLE}")
    print(f"   👥 {len(MATHEMATICIANS)} Mathematicians")
    print(f"   🚀 {len(MATH_PROJECTS)} Projects")
    print(f"   📐 {len(THEOREMS)} Theorems")
    print(f"   📺 {TEACHER_YOUTUBE}\n")
    app.run(host="0.0.0.0", port=port, debug=False)