"""
MathSphere v8.0 — Production-Ready Backend
==========================================
By Anupam Nigam | youtube.com/@pi_nomenal1729

WORKING ENDPOINTS:
✅ /api/chat - Ask Anupam (context-aware, natural responses)
✅ /api/formula - Formula sheets with proper notation
✅ /api/graph - Graph visualization with complete analysis
✅ /api/mathematician - Dynamic unlimited mathematicians
✅ /api/projects/generate - 5 projects per topic with code
✅ /api/theorem/prove - Complete rigorous proofs
✅ /api/competition/problems - 30-40 problems per visit
✅ /api/quiz/generate - 30+ questions per session
✅ /api/verify-solution - Verify math answers (Phase 1)
✅ /api/solution-paths - Multiple solution methods (Phase 1)
✅ /api/common-mistakes - Error analysis (Phase 1)
"""

import os
import re
import json
import random
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

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
        print("✅ Groq connected")
    except Exception as e:
        print(f"⚠️ Groq init failed: {e}")
        GROQ_AVAILABLE = False

if GEMINI_AVAILABLE:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini connected")
    except Exception as e:
        print(f"⚠️ Gemini init failed: {e}")
        GEMINI_AVAILABLE = False

# ════ SymPy Setup — FIXED: catches ALL errors, not just ImportError ════
try:
    print(f"🔍 Python version: {sys.version}")
    print(f"🔍 Attempting sympy import...")
    
    import sympy as sp
    print(f"🔍 sympy version: {sp.__version__}")
    
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
    print("✅ SymPy loaded successfully")

except ImportError as e:
    print(f"⚠️ SymPy ImportError: {e}")
except Exception as e:
    # THIS IS THE KEY FIX — was silently missing other errors before
    print(f"⚠️ SymPy failed with unexpected error: {type(e).__name__}: {e}")

TEACHER_YOUTUBE = "https://youtube.com/@pi_nomenal1729"
TEACHER_WEBSITE = "https://www.anupamnigam.com"

SYMPY_TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
) if SYMPY_AVAILABLE else None

def safe_parse(expr_str: str):
    """Parse mathematical expression safely"""
    if not SYMPY_AVAILABLE:
        return None
    try:
        expr_str = expr_str.strip()
        expr_str = re.sub(r'\^', '**', expr_str)
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        return parse_expr(expr_str, transformations=SYMPY_TRANSFORMATIONS)
    except Exception:
        try:
            return sympify(expr_str)
        except Exception:
            return None

# ════ RESPONSE CLEANING ════
def clean_response(text: str) -> str:
    """Remove asterisks, preserve LaTeX"""
    if not text:
        return text
    
    # Preserve LaTeX blocks
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
    
    # Remove asterisks
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*', '', text)
    
    # Restore LaTeX
    for key, latex in latex_blocks:
        text = text.replace(key, latex)
    
    return text.strip()

# ════ AI CORE ════
def ask_ai(messages, system=None, temperature=0.2):
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
                    temperature=temperature
                )
                return clean_response(r.choices[0].message.content)
            except Exception as e:
                if any(x in str(e).lower() for x in ["429", "rate_limit", "does not exist"]):
                    continue
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
            print(f"Gemini error: {e}")
    
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
            print(f"Gemini image error: {e}")
    
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

RESPONSE STYLE:
- Natural language, warm, helpful tone
- Math format: "Step 1: ..., Step 2: ..., Final: \\[\\boxed{{...}}\\]"
- For images: "I see [problem description]. Let me solve each..."
- CRISP answers unless depth requested

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
- What we need to show
- Given information
- Strategy/approach

Step 2: [Key insight]
[Derivation with ALL formulas in LaTeX]

Step 3: [Continue building]
[More steps with complete working]

[Continue until proof complete]

Final Step: [Conclusion]
Therefore: \\[\\boxed{{Conclusion}}\\]
✓ QED

📝 DETAILED PROOF (Alternative approach):
[Show complete alternative proof]

💡 INTUITIVE EXPLANATION:
[Explain why theorem is true in simple terms]

🔍 KEY LEMMAS USED:
- Lemma 1: [Statement with LaTeX]
- Lemma 2: ...

🎯 SPECIAL CASES:
- Case 1: ...
- Case 2: ...

💎 EXTENSIONS & GENERALIZATIONS:
[Related theorems and how this generalizes]

⚠️ COMMON PROOF MISTAKES:
[What students often get wrong]

📚 HISTORICAL CONTEXT:
[Who discovered it, when, why it matters]

🌍 APPLICATIONS:
[Real-world and theoretical uses]

Make proofs COMPLETE and RIGOROUS - not abbreviated."""

VERIFY_PROMPT = """Verify mathematical solutions completely.

For each solution provided:

1. SUBSTITUTION CHECK:
   - Plug answer back into original equation
   - Show all calculations
   - Verify it satisfies the condition

2. ALTERNATIVE METHOD CHECK:
   - Solve using different approach
   - Compare answers
   - If different: identify error

3. DOMAIN/RANGE CHECK:
   - Verify answer within domain
   - Check any restrictions
   - Confirm range compliance

4. BOUNDARY CHECK:
   - Test edge cases
   - Check special values
   - Verify limits

Result:
✅ VERIFIED - Answer is correct
❌ ERROR FOUND - [Explanation and correction]

Confidence: HIGH/MEDIUM/LOW"""

# ════ ROUTES ════

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "groq": GROQ_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "sympy": SYMPY_AVAILABLE,
        "version": "8.0",
        "python": sys.version
    })

# ════ CHAT ENDPOINT ════
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        image_b64 = data.get("image_b64")
        image_type = data.get("image_type")
        
        if not messages:
            return jsonify({"error": "messages required"}), 400
        
        clean = [{"role": m["role"], "content": str(m["content"])}
                 for m in messages if m.get("role") in ("user", "assistant")]
        if len(clean) > 30:
            clean = clean[-30:]
        
        answer = ask_ai_with_image(clean, image_b64=image_b64, image_type=image_type, 
                                   system=ASK_ANUPAM_PROMPT)
        
        return jsonify({"answer": answer, "confidence": "HIGH"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ FORMULA SHEET ════
@app.route("/api/formula", methods=["POST"])
def formula():
    try:
        data = request.get_json()
        topic = data.get("topic", "Calculus")
        exam = data.get("exam", "JAM")
        
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
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ GRAPH PLOTTER ════
@app.route("/api/graph", methods=["POST"])
def graph_plotter():
    try:
        data = request.get_json()
        expr_str = data.get("expression", "x**2")
        graph_type = data.get("type", "2d")
        
        if not SYMPY_AVAILABLE:
            prompt = f"""Analyze function: f(x) = {expr_str}

Provide COMPLETE analysis using LATEX notation:

📌 DOMAIN: \\[...\\]
📌 RANGE: \\[...\\]

🎯 CRITICAL POINTS:
- Point 1: \\(x = ..., y = ...\\)
- Type: [max/min/inflection]

📍 INTERCEPTS:
- x-intercepts: \\([...,...,...]\\)
- y-intercept: \\([...]\\)

🔄 ASYMPTOTES:
- Vertical: \\(x = ...\\)
- Horizontal: \\(y = ...\\)

📈 BEHAVIOR:
- As \\(x \\to \\infty\\): ...
- As \\(x \\to -\\infty\\): ...

🔢 DERIVATIVE: \\[f'(x) = ...\\]

Use PROPER mathematical notation throughout."""
            
            analysis = ask_simple(prompt, temperature=0.1)
            return jsonify({
                "sympy": False,
                "expression": expr_str,
                "analysis": analysis
            })
        
        # With SymPy
        f = safe_parse(re.sub(r'\^', '**', expr_str.strip()))
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
            
            analysis_prompt = f"""COMPLETE MATHEMATICAL ANALYSIS of f(x) = {expr_str}

Use PROPER LaTeX notation for everything:

📌 DOMAIN: State using set notation \\[...\\]
📌 RANGE: State using set notation \\[...\\]

🎯 CRITICAL POINTS (where \\(f'(x) = 0\\)):
- If exists: \\(x = ..., f(x) = ...\\)
- Classification: [local max/min/neither]

📍 INTERCEPTS:
- x-intercepts: \\([x_1, x_2, ...]\\)
- y-intercept: \\(f(0) = ...\\)

🔄 ASYMPTOTES:
- Vertical: \\(x = ...\\)
- Horizontal: \\(\\lim_{{x \\to \\infty}} f(x) = ...\\)

🔢 DERIVATIVE: \\[f'(x) = {df_latex}\\]

🔹 SECOND DERIVATIVE: \\[f''(x) = ...\\]

📊 CONCAVITY:
- Concave up: \\(x \\in (...)\\)
- Concave down: \\(x \\in (...)\\)

📈 BEHAVIOR:
- As \\(x \\to \\infty\\): \\(f(x) \\to ...\\)
- As \\(x \\to -\\infty\\): \\(f(x) \\to ...\\)

Format EVERYTHING in proper mathematical notation."""
            
            analysis = ask_simple(analysis_prompt, temperature=0.1)
            
            return jsonify({
                "sympy": True,
                "type": "2d",
                "points": points,
                "expression": expr_str,
                "latex": sp_latex(f),
                "derivative_latex": df_latex if df_latex else "",
                "critical_points": critical,
                "analysis": analysis
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ MATHEMATICIAN EXPLORER ════
@app.route("/api/mathematician", methods=["GET", "POST"])
def mathematician():
    try:
        name = None
        if request.method == "GET":
            name = request.args.get("name")
        else:
            name = request.get_json().get("name") if request.get_json() else None
        
        if not name:
            name = random.choice([
                "Gauss", "Euler", "Ramanujan", "Emmy Noether", "Alan Turing",
                "Terence Tao", "Maryam Mirzakhani", "Kurt Gödel"
            ])
        
        prompt = f"""Generate COMPLETE biography of mathematician: {name}

Return ONLY valid JSON:
{{
  "name": "Full name",
  "period": "Birth–Death years",
  "country": "Country",
  "fields": ["Field1", "Field2"],
  "biography": "3-4 paragraph detailed biography",
  "major_contributions": ["Contribution 1", "Contribution 2"],
  "famous_quote": "Famous quote",
  "key_achievements": {{"theorem1": "Description", "theorem2": "Description"}},
  "impact_today": "How work impacts modern world",
  "learning_resources": ["Book/Paper 1", "Resource 2"],
  "wikipedia": "Wikipedia URL"
}}

Make biography COMPREHENSIVE."""
        
        response = ask_simple(prompt, temperature=0.3)
        try:
            data = json.loads(response)
            return jsonify(data)
        except:
            return jsonify({"name": name, "biography": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ PROJECTS ════
@app.route("/api/projects/generate", methods=["POST"])
def projects_generate():
    try:
        data = request.get_json()
        topic = data.get("topic", "Machine Learning")
        
        prompt = f"""Generate 5 DETAILED projects for topic: {topic}

Return ONLY valid JSON array:
[{{
  "number": 1,
  "title": "Project title",
  "difficulty": "Beginner/Intermediate/Advanced",
  "description": "3-4 sentence description",
  "math_concepts": ["Concept 1: formula", "Concept 2: formula"],
  "step_by_step": [
    "Step 1: [detailed subtitle] - Full explanation with math",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ...",
    "Step 5: ..."
  ],
  "code_snippet": "Complete working Python code",
  "expected_outcome": "What you'll build",
  "career_salary": "Job title + salary range",
  "resources": ["Book", "Course", "GitHub"]
}}]

Make COMPLETE with code and formulas."""
        
        response = ask_simple(prompt, temperature=0.4)
        projects = None
        try: projects = json.loads(response)
        except: pass
        if not projects:
            import re as _re
            m = _re.search(r'\[[\s\S]*\]', response)
            if m:
                try: projects = json.loads(m.group(0))
                except: pass
        if projects and isinstance(projects, list):
            return jsonify({"topic": topic, "projects": projects})
        return jsonify({"topic": topic, "projects": [{"title": "Projects for " + topic, "description": response[:800], "difficulty": "Various", "step_by_step": [], "math_concepts": [], "resources": [], "code_snippet": ""}]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ THEOREM PROVER ════
@app.route("/api/theorem/prove", methods=["POST"])
def theorem_prove():
    try:
        data = request.get_json()
        theorem_name = data.get("theorem", "Pythagorean Theorem")
        
        prompt = f"""Prove theorem COMPLETELY: {theorem_name}

OUTPUT REQUIRED:

📌 THEOREM: {theorem_name}

📖 STATEMENT:
\\[Mathematical statement\\]

✅ PROOF (step-by-step from scratch):

Step 1: [Setup and definitions]
Step 2: [Key insight]
Step 3: [Main argument]
[Continue with ALL steps]
Final: Conclusion - QED ✓

💡 INTUITIVE EXPLANATION:
[Why the theorem is true]

🔍 KEY LEMMAS:
[Lemmas used]

🎯 APPLICATIONS:
[Real-world uses]

⚠️ COMMON MISTAKES:
[What students get wrong]

Make proof COMPLETE and RIGOROUS."""
        
        proof = ask_simple(prompt, system=THEOREM_PROMPT, temperature=0.1)
        return jsonify({"theorem": theorem_name, "proof": proof})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ COMPETITION PROBLEMS ════
@app.route("/api/competition/problems", methods=["POST"])
def competition_problems():
    try:
        data = request.get_json()
        category = data.get("category", "IMO")
        count = int(data.get("count", 30))
        
        prompt = f"""Generate {count} {category} problems with COMPLETE solutions.

For EACH problem:

**Problem [N]:**
\\[Problem statement with full LaTeX\\]

Difficulty: [Easy/Medium/Hard]
Topics: [Topics tested]
Hint: [Strategic hint]

**SOLUTION:**
Step 1: [Analysis]
Step 2: [Approach]
[All steps with LaTeX]
Final Answer: \\[\\boxed{{...}}\\]

**Insight:** [Why this works]

Generate {count} complete problems."""
        
        problems_text = ask_simple(prompt, temperature=0.2)
        return jsonify({
            "category": category,
            "count": count,
            "problems": problems_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ QUIZ GENERATION ════
@app.route("/api/quiz/generate", methods=["POST"])
def quiz_generate():
    try:
        data = request.get_json()
        topic = data.get("topic", "Calculus")
        count = int(data.get("count", 30))
        
        prompt = f"""Generate {count} exam-style questions for {topic}.

For EACH question:

**Question [N]:**
\\[Problem with full LaTeX\\]

**SOLUTION:**
Step 1: [Full working with LaTeX]
Step 2: [Continue]
...
**Answer:** \\[\\boxed{{...}}\\]

**Explanation:** [Why correct]

Generate {count} varied questions with COMPLETE solutions."""
        
        questions = ask_simple(prompt, temperature=0.3)
        return jsonify({
            "topic": topic,
            "count": count,
            "questions": questions
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ PHASE 1: ACCURACY IMPROVEMENTS ════

@app.route("/api/verify-solution", methods=["POST"])
def verify_solution():
    try:
        data = request.get_json()
        problem = data.get("problem")
        solution = data.get("solution")
        
        prompt = f"""VERIFY this solution:

Problem: {problem}
Solution: {solution}

Check:
1. Substitute back - does it work?
2. Alternative method - get same answer?
3. Domain/range - valid?
4. Edge cases - handled?

Result:
✅ VERIFIED - Correct
❌ ERROR - [Correction]"""
        
        verification = ask_simple(prompt, system=VERIFY_PROMPT, temperature=0.1)
        return jsonify({"verification": verification})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/solution-paths", methods=["POST"])
def solution_paths():
    try:
        data = request.get_json()
        problem = data.get("problem")
        
        prompt = f"""Show 3-5 DIFFERENT methods to solve:

Problem: {problem}

For EACH method:

**METHOD 1: [Approach name]**
- Steps: \\[...\\]
- Difficulty: [Easy/Medium/Hard]
- When to use: ...
- Time: X minutes

[Repeat for methods 2-5]

**COMPARISON:** Which is best? Why?"""
        
        paths = ask_simple(prompt, temperature=0.3)
        return jsonify({"methods": paths})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/common-mistakes", methods=["POST"])
def common_mistakes():
    try:
        data = request.get_json()
        topic = data.get("topic")
        
        prompt = f"""Common mistakes in {topic}:

For EACH mistake (7-10 total):

**Mistake [N]:**
❌ Wrong: \\[...\\]
✅ Correct: \\[...\\]
💡 Why: [Why students make this mistake]
🔧 Fix: [How to avoid it]"""
        
        mistakes = ask_simple(prompt, temperature=0.2)
        return jsonify({"mistakes": mistakes})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════ EXAM INFO ════
@app.route("/api/exam/<exam>", methods=["GET"])
def exam_info(exam):
    try:
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
        return jsonify({"error": str(e)}), 500

# ════ ERROR HANDLERS ════
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"""
════════════════════════════════════════
🧮 MathSphere v8.0 - Production Backend
════════════════════════════════════════
✅ Groq:   {GROQ_AVAILABLE}
✅ Gemini: {GEMINI_AVAILABLE}
✅ SymPy:  {SYMPY_AVAILABLE}
🐍 Python: {sys.version}
════════════════════════════════════════
📺 {TEACHER_YOUTUBE}
════════════════════════════════════════
    """)
    app.run(host="0.0.0.0", port=port, debug=False)