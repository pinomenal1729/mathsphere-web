"""
MathSphere Web v8.0 — Professor Edition (ENHANCED)
====================================================
By Anupam Nigam | youtube.com/@pi_nomenal1729

MAJOR IMPROVEMENTS:
✅ Ask Anupam: Dynamic, context-aware chat with natural responses
✅ Graph Plotter: Embedded visualization with detailed analysis
✅ Mathematicians: Unlimited dynamic generation via AI
✅ Projects: Detailed multi-step exploration
✅ Theorems: Complete proofs with step-by-step derivations
✅ Competitions: Unlimited problems (30-40 per session)
✅ Quiz/Mock Test: Dynamic question generation (30+ per session)
✅ Image Solving: Line-by-line detailed solutions
✅ Proper LaTeX rendering in all responses
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
        Sum, Product, cos, sin, tan, exp, log, Abs,
        factorial, gcd, lcm, isprime, factorint,
        Derivative, Integral
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor
    )
    SYMPY_AVAILABLE = True
    print("✅ SymPy loaded")
except ImportError:
    print("⚠️ SymPy not installed")

GROQ_MODELS     = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TEACHER_YOUTUBE = "https://youtube.com/@pi_nomenal1729"
TEACHER_WEBSITE = "https://www.anupamnigam.com"

SYMPY_TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
) if SYMPY_AVAILABLE else None

def safe_parse(expr_str: str):
    if not SYMPY_AVAILABLE: return None
    try:
        expr_str = expr_str.strip()
        expr_str = re.sub(r'\^', '**', expr_str)
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        return parse_expr(expr_str, transformations=SYMPY_TRANSFORMATIONS)
    except Exception:
        try: return sympify(expr_str)
        except Exception: return None


# ══════════════════════════════════════════════════
# ASTERISK REMOVER & CLEANER
# ══════════════════════════════════════════════════

def clean_response(text: str) -> str:
    """Remove markdown asterisks but preserve LaTeX"""
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
# AI CORE — IMPROVED
# ══════════════════════════════════════════════════

def ask_ai(messages, system=None, temperature=0.2):
    """Core AI function with fallback support"""
    if GROQ_AVAILABLE:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        if len(full) > 25: full = [full[0]] + full[-23:]  # Keep more context
        for model in GROQ_MODELS:
            try:
                r = groq_client.chat.completions.create(
                    model=model, messages=full, max_tokens=4000, temperature=temperature)
                return clean_response(r.choices[0].message.content)
            except Exception as e:
                if any(x in str(e).lower() for x in ["429","rate_limit","model_not_active","does not exist"]):
                    continue
                raise

    if GEMINI_AVAILABLE:
        try:
            parts = ([f"SYSTEM:\n{system}\n\n"] if system else []) + \
                    [f"{'Student' if m['role']=='user' else 'Assistant'}: {m['content']}\n" for m in messages]
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents="".join(parts),
                generation_config={"temperature": temperature})
            return clean_response(r.text)
        except Exception as e:
            print(f"Gemini error: {e}")

    return "⚠️ AI temporarily unavailable. Please try again!"


def ask_simple(prompt, system=None, temperature=0.2):
    """Single-turn AI call"""
    return ask_ai([{"role": "user", "content": prompt}], system=system, temperature=temperature)


def ask_ai_with_image(messages, image_b64=None, image_type=None, system=None):
    """AI with image analysis"""
    if GEMINI_AVAILABLE and image_b64 and image_type:
        try:
            prompt_parts = []
            if system:
                prompt_parts.append(f"SYSTEM:\n{system}\n")
            for m in messages:
                role = "Student" if m.get("role") == "user" else "Assistant"
                prompt_parts.append(f"{role}: {m.get('content', '')}\n")
            prompt_parts.append("Analyse the uploaded image step by step. Solve every question/problem in the image completely.")
            contents = [
                {"text": "\n".join(prompt_parts)},
                {"inline_data": {"mime_type": image_type, "data": image_b64}}
            ]
            r = gemini_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=contents,
                generation_config={"temperature": 0.1})
            return clean_response(r.text)
        except Exception as e:
            print(f"Gemini image error: {e}")

    fallback = list(messages)
    if image_b64:
        fallback.append({"role": "user", "content": "User uploaded an image. Solve step by step with complete details."})
    return ask_ai(fallback, system=system)


def parse_json_block(raw: str):
    """Extract JSON from response"""
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


# ══════════════════════════════════════════════════
# SYSTEM PROMPTS — IMPROVED
# ══════════════════════════════════════════════════

ASK_ANUPAM_PROMPT = f"""You are Ask Anupam — an AI tutor by Anupam Nigam.

CORE RULES:
1. ALWAYS maintain full conversation context — read ALL previous messages
2. For EVERY question: answer naturally, conversationally, and CRISP (not verbose)
3. For math questions: solve step-by-step with complete working + verification
4. For image uploads: solve EVERY question line-by-line with full details
5. NEVER use asterisks — use CAPS or bold formatting instead
6. ALL mathematics in proper LaTeX: \\(inline\\) or \\[display\\]
7. Box final answers: \\[\\boxed{{...}}\\]
8. Handle ANY topic: academics, advice, explanations, creative tasks, code, debugging, etc.
9. State confidence at end of math: [CONFIDENCE: HIGH/MEDIUM/LOW]

RESPONSE STYLE:
- Keep answers CONCISE unless depth is requested
- Use natural language with warm, helpful tone
- For math: "Step 1: ..., Step 2: ..., Final Answer: \\[\\boxed{{...}}\\]"
- For images: "I can see: [what's in image]. Let me solve each problem..."
- For non-math: Just answer directly and naturally

REMEMBER: This is a CHAT interface like ChatGPT. Be conversational, not robotic."""

SYSTEM_PROMPT = f"""You are MathSphere — an expert Mathematics professor, created by Anupam Nigam.

MATHEMATICAL ACCURACY RULES:
1. ALWAYS compute numerical examples to self-verify
2. NEVER skip algebraic steps — show EVERY manipulation
3. State ALL assumptions explicitly
4. If not 100% certain, say so clearly
5. For calculus: always verify by differentiating integral results
6. For algebra: substitute back to verify solutions
7. Box final answers: \\[\\boxed{{...}}\\]
8. Maintain conversation context — refer to previous messages

FORMAT EVERY RESPONSE:
📌 [Topic Name]
💡 Real-life Application: [1 sentence]
📖 Given information and assumptions
📐 Step-by-step solution (EVERY step shown)
✅ Final answer: \\[\\boxed{{...}}\\]
⚠️ Common mistakes on this type
📚 {TEACHER_YOUTUBE}

STYLE RULES:
1. NO asterisks — use CAPS for emphasis
2. ALL mathematics in LaTeX
3. Warm, encouraging tone
4. Temperature is set low — be precise
5. HTML tags allowed: <br> <hr>

[CONFIDENCE: HIGH] or [CONFIDENCE: MEDIUM] or [CONFIDENCE: LOW]"""

MATHEMATICIAN_PROMPT = f"""You are a biographical AI expert. Generate a COMPLETE, detailed biography.

RETURN ONLY VALID JSON:
{{
  "name": "Full name",
  "period": "Birth–Death years",
  "country": "Country",
  "fields": ["Field1", "Field2"],
  "bio": "2-3 paragraph detailed biography",
  "major_contributions": ["Contribution 1", "Contribution 2"],
  "key_results": "Important theorems/results (with LaTeX formulas)",
  "famous_quote": "One famous quote",
  "image_url": "Wikipedia image URL (search on Wikipedia)",
  "impact_today": "How their work impacts modern world",
  "learning_resources": ["Resource 1", "Resource 2"],
  "wikipedia": "Wikipedia URL"
}}

Be comprehensive and accurate. Include mathematical depth."""

PROJECT_PROMPT = """You are a project advisor. Generate detailed, actionable projects.

RETURN ONLY VALID JSON:
{{
  "title": "Project title",
  "difficulty": "Beginner/Intermediate/Advanced",
  "description": "2-3 sentence detailed description",
  "math_concepts": ["Concept1 (with formula)", "Concept2"],
  "real_world_use": "Specific companies and industries",
  "step_by_step": ["Step 1: ...", "Step 2: ...", ...],
  "code_example": "Python snippet showing implementation",
  "learning_outcomes": ["Outcome 1", "Outcome 2"],
  "resources": ["Book", "Course", "Tutorial"],
  "career_impact": "Salary range and job titles"
}}

Be detailed. Include code. Make it actionable."""

THEOREM_PROOF_PROMPT = """You are a rigorous mathematics educator. Prove theorems completely.

PROOF FORMAT:
📌 THEOREM NAME
📖 Statement: \\[...\\]
✅ PROOF (complete with every step):
  Step 1: [Introduction/Setup]
  Step 2: [Key insight]
  Step 3: [Main argument]
  ...
  Final Step: [Conclusion - QED]
  
📐 Formal Proof (with all LaTeX formulas)
💡 Intuition behind the proof
🎯 Key Lemmas used
📚 Extensions and related theorems
⚠️ Common proof mistakes
🔍 Historical context

Make proofs COMPLETE and RIGOROUS."""

COMPETITION_PROMPT = """You are a competition math expert. Generate RIGOROUS olympiad-style problems.

EXACT FORMAT FOR EACH PROBLEM:
Problem [N]:
[Statement with full LaTeX]

Difficulty: [Easy/Medium/Hard]
Key Concepts: [Topics tested]
Hint: [Hint leading toward solution]

SOLUTION (COMPLETE):
Step 1: [Analysis]
Step 2: [Approach]
...
Final Answer: \\[\\boxed{{...}}\\]

⚠️ Common mistakes
🎯 Technique for similar problems
📚 Related theorems

GENERATE 30-40 PROBLEMS. Be comprehensive."""

MOCK_TEST_PROMPT = """You are a rigorous exam question generator. Create exam-style questions.

FOR EACH QUESTION:
Question [N]:
[2-3 sentence problem statement with LaTeX]

Options (if MCQ):
A) ...  B) ...  C) ...  D) ...

SOLUTION:
Step 1: [Identify topic and approach]
Step 2: [Work through calculations]
...
Answer: \\[\\boxed{{...}}\\]

⚠️ Why other options are wrong
💡 Key insight
⏱️ Expected time: X minutes

GENERATE 30-40 QUESTIONS. Ensure variety of topics."""

# ══════════════════════════════════════════════════
# STATIC DATA (Keep as is)
# ══════════════════════════════════════════════════

MATHEMATICIANS = {
    "Srinivasa Ramanujan": {
        "period": "1887–1920", "country": "India",
        "fields": ["Number Theory", "Infinite Series", "Modular Forms"],
        "contribution": "Discovered 3900+ results with almost no formal training.",
    },
    "Leonhard Euler": {
        "period": "1707–1783", "country": "Switzerland",
        "fields": ["Analysis", "Graph Theory", "Number Theory"],
        "contribution": "800+ papers. Founded graph theory, created e, π, i notation.",
    },
}

THEOREMS = {
    "Pythagorean Theorem": {
        "statement": "In a right triangle: \\(a^2 + b^2 = c^2\\)",
        "applications": "Distance formula, GPS, complex numbers",
    },
}

EXAM_INFO = {
    "JAM": {"full_name": "IIT JAM Mathematics", "pattern": "3 hours · 60 questions · 100 marks"},
    "GATE": {"full_name": "GATE Mathematics", "pattern": "3 hours · 65 questions · 100 marks"},
    "CSIR": {"full_name": "CSIR UGC NET", "pattern": "3 hours · 200 marks (Parts A/B/C)"}
}


# ══════════════════════════════════════════════════
# ROUTES
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
        "version": "8.0"
    })


# ── CHAT (Full context memory, natural responses) ──

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Ask Anupam endpoint - IMPROVED
    Maintains full conversation context
    Natural, crisp responses for any topic
    """
    try:
        data       = request.get_json()
        messages   = data.get("messages", [])
        image_b64  = data.get("image_b64")
        image_type = data.get("image_type")
        
        if not messages:
            return jsonify({"error": "messages required"}), 400
        
        # Keep full conversation (up to 30 messages for context)
        clean = [{"role": m["role"], "content": str(m["content"])}
                 for m in messages if m.get("role") in ("user","assistant") and m.get("content")]
        if len(clean) > 30: clean = clean[-30:]
        
        # For image uploads: ensure system prompt emphasizes line-by-line solving
        system = ASK_ANUPAM_PROMPT
        if image_b64:
            system += "\n\nPRIORITY: Solve every question in the image COMPLETELY, line by line, with all details."
        
        answer = ask_ai_with_image(clean, image_b64=image_b64, image_type=image_type, system=system)

        # Extract confidence
        confidence = "HIGH"
        if "[CONFIDENCE: MEDIUM]" in answer: confidence = "MEDIUM"
        elif "[CONFIDENCE: LOW]" in answer: confidence = "LOW"
        answer = re.sub(r'\[CONFIDENCE: (HIGH|MEDIUM|LOW)\]', '', answer).strip()

        return jsonify({"answer": answer, "confidence": confidence})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ── GRAPH PLOTTER (with visualization data) ──

@app.route("/api/graph", methods=["POST"])
def graph_data():
    """
    IMPROVED: Returns complete graph data with analysis
    Includes domain, range, intercepts, critical points
    Returns data in format ready for frontend plotting
    """
    if not SYMPY_AVAILABLE:
        data = request.get_json()
        expr = data.get("expression", "x**2")
        prompt = f"""Analyze the function: f(x) = {expr}

📊 COMPLETE ANALYSIS:
1. Domain: [exact notation]
2. Range: [exact notation]
3. Asymptotes: [vertical/horizontal/oblique]
4. Intercepts: x-intercepts and y-intercepts (with coordinates)
5. Critical Points: [local max/min with values]
6. Behavior: [as x→±∞]
7. Periodic?: [period if applicable]
8. Symmetry: [even/odd/neither]
9. Sketch description: [detailed description of shape]

ALL coordinates in LaTeX. Make it complete."""
        return jsonify({
            "sympy": False,
            "expression": expr,
            "analysis": ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.1)
        })
    
    data = request.get_json()
    expr_str = data.get("expression", "x**2")
    graph_type = data.get("type", "2d")
    x_min = float(data.get("x_min", -5))
    x_max = float(data.get("x_max", 5))
    y_min = float(data.get("y_min", -5))
    y_max = float(data.get("y_max", 5))
    
    try:
        x = Symbol("x")
        y = Symbol("y")
        
        if graph_type == "3d":
            expr_str_clean = re.sub(r'\^', '**', expr_str.strip())
            f = safe_parse(expr_str_clean)
            if f is None:
                return jsonify({"error": "Could not parse expression"}), 400
            
            points_x, points_y, points_z = [], [], []
            steps = 30
            for i in range(steps+1):
                xv = x_min + i*(x_max-x_min)/steps
                for j in range(steps+1):
                    yv = y_min + j*(y_max-y_min)/steps
                    try:
                        zv = float(f.subs([(x, xv), (y, yv)]))
                        if abs(zv) < 1e8:
                            points_x.append(round(xv,4))
                            points_y.append(round(yv,4))
                            points_z.append(round(zv,4))
                    except: pass
            
            return jsonify({
                "sympy": True, "type": "3d",
                "x": points_x, "y": points_y, "z": points_z,
                "expression": expr_str,
                "latex": sp_latex(safe_parse(expr_str_clean)) if safe_parse(expr_str_clean) else expr_str
            })
        else:
            # 2D plot with COMPLETE analysis
            f = safe_parse(re.sub(r'\^', '**', expr_str.strip()))
            if f is None:
                return jsonify({"error": "Could not parse expression"}), 400
            
            points = []
            num_pts = 300
            step = (x_max - x_min) / num_pts
            for i in range(num_pts+1):
                xv = x_min + i*step
                try:
                    yv = float(f.subs(x, xv))
                    if abs(yv) < 1e6:
                        points.append({"x": round(xv,4), "y": round(yv,4)})
                    else:
                        points.append({"x": round(xv,4), "y": None})
                except:
                    points.append({"x": round(xv,4), "y": None})
            
            # Derivative and critical points
            try:
                df = diff(f, x)
                df_latex = sp_latex(simplify(df))
                crit = sorted([float(c) for c in solve(df, x) if c.is_real and x_min <= float(c) <= x_max])
            except:
                df_latex = ""
                crit = []
            
            # Get analysis from AI
            analysis_prompt = f"""Analyze function: f(x) = {expr_str}
Domain: [{x_min}, {x_max}]

Provide:
1. Exact Domain & Range
2. All Intercepts (x and y with coordinates)
3. Critical Points & their nature (max/min/inflection)
4. Asymptotes (if any)
5. Behavior at boundaries
6. Key feature description

Use LaTeX. Be precise."""
            
            analysis = ask_simple(analysis_prompt, system=ASK_ANUPAM_PROMPT, temperature=0.1)
            
            return jsonify({
                "sympy": True, "type": "2d",
                "points": points,
                "expression": expr_str,
                "latex": sp_latex(f),
                "derivative_latex": df_latex,
                "critical_points": crit,
                "analysis": analysis
            })
    except Exception as e:
        return jsonify({"error": f"Graph error: {str(e)}"}), 500


# ── MATHEMATICIANS (DYNAMIC, unlimited) ──

@app.route("/api/mathematician", methods=["GET", "POST"])
def mathematician():
    """
    IMPROVED: Dynamic mathematician generation
    Can request specific mathematician or get random
    Returns comprehensive biography
    """
    data = request.get_json() or {}
    name = request.args.get("name") or data.get("name")
    
    if not name:
        # Random mathematician
        name = random.choice([
            "Srinivasa Ramanujan", "Leonhard Euler", "Carl Friedrich Gauss",
            "Emmy Noether", "Isaac Newton", "Bernhard Riemann", "Alan Turing",
            "Terence Tao", "Maryam Mirzakhani", "Kurt Gödel"
        ])
    
    prompt = f"""Generate a DETAILED biography of {name}.

Return ONLY valid JSON:
{{
  "name": "Full name",
  "period": "Birth–Death",
  "country": "Country/Countries",
  "fields": ["Field1", "Field2", "Field3"],
  "biography": "3-4 paragraph detailed biography",
  "major_contributions": ["Contribution 1 (with formula/theorem)", "Contribution 2"],
  "famous_quote": "Most famous quote (if available)",
  "key_achievements": {{"theorem1": "Description with LaTeX", "theorem2": "..."}},
  "impact_today": "How their work is used today (companies, fields, etc)",
  "interesting_facts": ["Fact 1", "Fact 2", "Fact 3"],
  "learning_resources": ["Book/Paper 1", "Resource 2"],
  "wikipedia": "Wikipedia search URL"
}}

Be comprehensive, accurate, and include mathematical depth."""
    
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.3)
    data_out = parse_json_block(raw)
    
    if isinstance(data_out, dict) and data_out.get("name"):
        return jsonify(data_out)
    
    # Fallback
    return jsonify({
        "name": name,
        "error": "Could not generate full biography. Try another mathematician.",
        "raw_response": raw[:500]
    })

@app.route("/api/mathematicians/list")
def mathematicians_list():
    """Suggest mathematician names"""
    names = [
        "Srinivasa Ramanujan", "Leonhard Euler", "Carl Friedrich Gauss",
        "Emmy Noether", "Isaac Newton", "Bernhard Riemann", "Alan Turing",
        "Terence Tao", "Maryam Mirzakhani", "Kurt Gödel", "David Hilbert",
        "André Weil", "Paul Erdős", "Pierre de Fermat", "Sophie Germain"
    ]
    return jsonify({"mathematicians": names, "note": "You can ask for any mathematician!"})


# ── PROJECTS (Detailed, actionable) ──

@app.route("/api/projects/generate", methods=["POST"])
def projects_generate():
    """
    IMPROVED: Generate 5 detailed projects for any topic
    Each project has step-by-step implementation
    """
    data = request.get_json()
    topic = data.get("topic", "Machine Learning")
    
    prompt = f"""Generate 5 DETAILED, actionable projects for: {topic}

Return ONLY valid JSON array:
[{{
  "number": 1,
  "title": "Project title",
  "difficulty": "Beginner/Intermediate/Advanced",
  "description": "3-4 sentence detailed description of what you build",
  "math_concepts": ["Concept 1 (with key formula in LaTeX)", "Concept 2"],
  "real_world_applications": ["Company/Industry 1", "Company/Industry 2"],
  "step_by_step": [
    "Step 1: [Detailed subtitle] - Explain what to do",
    "Step 2: [Detailed subtitle] - Include math concepts",
    ...
  ],
  "code_snippet": "Python code example (complete, runnable)",
  "expected_outcome": "What you'll have built at the end",
  "estimated_time": "X weeks/months",
  "learning_outcomes": ["Outcome 1", "Outcome 2", "Outcome 3"],
  "resources": ["Book/Paper", "YouTube Course", "GitHub Tutorial"],
  "career_salary": "Job title + salary range (₹/$/€)",
  "difficulty_explanation": "Why this difficulty level"
}}]

Be COMPREHENSIVE. Include complete code. Make it actionable and inspiring."""
    
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.4)
    projects = parse_json_block(raw)
    
    if isinstance(projects, list) and projects:
        return jsonify({"topic": topic, "projects": projects, "count": len(projects)})
    
    return jsonify({"topic": topic, "projects": [], "error": "Could not generate projects", "raw": raw[:1000]})


# ── THEOREMS (Complete proofs) ──

@app.route("/api/theorem/prove", methods=["POST"])
def theorem_prove():
    """
    IMPROVED: Complete, rigorous theorem proofs
    Step-by-step derivation from scratch
    """
    data = request.get_json()
    theorem_name = data.get("theorem", "Pythagorean Theorem")
    
    prompt = f"""Provide a COMPLETE, RIGOROUS proof of: {theorem_name}

FORMAT (EXACT):

📌 THEOREM: {theorem_name}

📖 STATEMENT:
\\[Exact mathematical statement in LaTeX\\]

✅ COMPLETE PROOF (from scratch):
Step 1: [Setup/Definitions]
  - Define all variables
  - State what we need to prove
  
Step 2: [First key step]
  - Show calculations
  - Include relevant formulas in LaTeX
  
Step 3: [Continue building]
  ...
  
Final Step: [Conclusion]
  Therefore \\[\\boxed{{Conclusion}}\\]
  ✓ QED

📐 INTUITIVE EXPLANATION:
[Explain why this theorem is true in simple terms]

🔍 KEY LEMMAS USED:
[List any lemmas or earlier theorems required]

💡 VARIATIONS & EXTENSIONS:
[Related theorems and generalizations]

⚠️ COMMON PROOF MISTAKES:
[What students often get wrong]

📚 HISTORICAL CONTEXT:
[Who discovered it, when, why it matters]

🎯 APPLICATIONS:
[Real-world and theoretical uses]

Make the proof COMPLETE, DETAILED, and EDUCATIONAL."""
    
    proof = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.1)
    
    return jsonify({
        "theorem": theorem_name,
        "proof": proof,
        "has_image": False
    })


# ── COMPETITIONS (30-40 problems per category) ──

@app.route("/api/competition/problems", methods=["POST"])
def competition_problems():
    """
    IMPROVED: Generate 30-40 competition problems
    Each with complete solution
    User can ask for specific category (IMO, Putnam, AIME)
    """
    data = request.get_json()
    category = data.get("category", "IMO")
    difficulty = data.get("difficulty", "mixed")
    count = int(data.get("count", 30))
    
    prompt = f"""Generate {count} competition problems from {category}.
Difficulty: {difficulty}

FORMAT FOR EACH (STRICT):

**Problem [N]:**
[Full problem statement with LaTeX]

Difficulty: [Easy/Medium/Hard]
Topics: [Topics tested]
Hint: [Strategic hint]

**SOLUTION:**
Step 1: [Analyze what's being asked]
Step 2: [Key insight/approach]
Step 3: [Calculation/proof]
...
Final Answer: \\[\\boxed{{Answer}}\\]

**Insight:** [Why this approach works]
**Technique:** [Method to use for similar problems]
**Mistakes:** [What competitors commonly get wrong]

---

Generate {count} complete problems with solutions. Be rigorous."""
    
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.2)
    
    return jsonify({
        "category": category,
        "difficulty": difficulty,
        "count": count,
        "problems_raw": raw,
        "note": "Problems displayed in order. Each has complete solution."
    })

@app.route("/api/problem/<int:pid>/solve", methods=["POST"])
def problem_solve(pid):
    """Solve a specific competition problem with detailed explanation"""
    data = request.get_json()
    problem_statement = data.get("problem", "")
    
    if not problem_statement:
        return jsonify({"error": "problem statement required"}), 400
    
    prompt = f"""A mathematician is stuck on this competition problem:

{problem_statement}

Provide COMPLETE solution:
Step 1: [Understand what's being asked]
Step 2: [Key insight]
Step 3: [Full derivation/proof]
...
Final Answer: \\[\\boxed{{...}}\\]

⚠️ Why this is the right approach
💡 Technique: How to solve similar problems
📚 Related theorems and concepts

Be THOROUGH and RIGOROUS."""
    
    solution = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.1)
    
    return jsonify({
        "problem_id": pid,
        "problem": problem_statement,
        "solution": solution
    })


# ── QUIZ / MOCK TEST (30+ dynamic questions) ──

@app.route("/api/quiz/generate", methods=["POST"])
def quiz_generate():
    """
    IMPROVED: Generate 30-40 quiz questions
    Each with complete solution
    Covers multiple topics
    """
    data = request.get_json()
    topic = data.get("topic", "Calculus")
    difficulty = data.get("difficulty", "medium")
    count = int(data.get("count", 30))
    exam = data.get("exam", "General")
    
    prompt = f"""Generate {count} exam-style questions for {topic}.
Difficulty: {difficulty}
Exam: {exam}

FORMAT FOR EACH QUESTION:

**Question [N]:**
[2-3 sentence problem with full LaTeX]

**Type:** MCQ/Short Answer/Proof
**Topics:** [Skills tested]
**Time:** X minutes

**SOLUTION:**
Step 1: [Identify what's asked]
Step 2: [Apply relevant theorem/formula]
Step 3: [Calculate/Derive]
...
**Answer:** \\[\\boxed{{...}}\\]

**Explanation:** [Why this is correct]
**Common Mistakes:** [What students get wrong]
**Key Insight:** [Main idea to understand]

---

Generate {count} varied questions. Include mix of easy/medium/hard."""
    
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.3)
    
    return jsonify({
        "topic": topic,
        "difficulty": difficulty,
        "count": count,
        "exam": exam,
        "questions_raw": raw
    })

@app.route("/api/question/solve", methods=["POST"])
def question_solve():
    """Solve any single question with complete explanation"""
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "question required"}), 400
    
    prompt = f"""A student is stuck on:

{question}

Provide COMPLETE solution:
📌 What's being asked
📖 Identify relevant concepts
📐 Step-by-step working
✅ Final answer: \\[\\boxed{{...}}\\]
💡 Key insight
⚠️ Common mistakes
🎯 Similar problems to practice

Be thorough and clear."""
    
    solution = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.1)
    
    return jsonify({
        "question": question,
        "solution": solution
    })


# ── PYQ (Previous Year Questions) ──

@app.route("/api/pyq/generate", methods=["POST"])
def pyq_generate():
    """Generate realistic previous year questions"""
    data = request.get_json()
    exam = data.get("exam", "JAM")
    topic = data.get("topic", "Real Analysis")
    year = data.get("year", 2024)
    count = int(data.get("count", 10))
    
    prompt = f"""Generate {count} realistic PYQ-style questions for {exam} {year}, topic: {topic}.

FORMAT:

**Question [N]:** [Problem statement]
**Answer:** \\[\\boxed{{...}}\\]
**Solution:** [Step-by-step]

Generate {count} varied questions."""
    
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT, temperature=0.2)
    
    return jsonify({
        "exam": exam,
        "topic": topic,
        "year": year,
        "count": count,
        "questions": raw
    })


# ── CHALLENGE OF THE DAY ──

@app.route("/api/challenge")
def challenge():
    """Random daily challenge"""
    challenges = [
        "Prove \\(\\sqrt{2}\\) is irrational",
        "Find all critical points of \\(f(x)=x^3-3x+2\\)",
        "Compute \\(\\int_0^\\pi x\\sin x\\,dx\\)",
        "Eigenvalues of \\(\\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}\\)",
        "Solve \\(\\frac{dy}{dx}+2y=4x, y(0)=1\\)",
    ]
    return jsonify({"challenge": random.choice(challenges)})


# ── EXAM INFO ──

@app.route("/api/exam/<exam>")
def exam_info(exam):
    """Exam information and resources"""
    return jsonify(EXAM_INFO.get(exam, {"error": "Exam not found. Use JAM, GATE, or CSIR"}))


# ── GRAPH PLOTTING (Interactive visualization) ──

@app.route("/api/graph/interactive", methods=["POST"])
def graph_interactive():
    """Interactive graph with all features visible"""
    # Returns data ready for frontend graphing library (Plotly, Chart.js, etc)
    data = request.get_json()
    expression = data.get("expression", "x**2")
    
    # Returns complete data
    return graph_data()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🧮 MathSphere v8.0 — Enhanced Backend")
    print(f"   ✅ SymPy: {SYMPY_AVAILABLE}")
    print(f"   ✅ Groq: {GROQ_AVAILABLE}")
    print(f"   ✅ Gemini: {GEMINI_AVAILABLE}")
    print(f"   📺 {TEACHER_YOUTUBE}\n")
    app.run(host="0.0.0.0", port=port, debug=False)