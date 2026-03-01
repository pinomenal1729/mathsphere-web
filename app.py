"""
MathSphere Web v7.0 — Professor Edition
========================================
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
        if len(full) > 20: full = [full[0]] + full[-18:]
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


# ══════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are MathSphere — an expert Mathematics professor for graduate students, created by Anupam Nigam.

MATHEMATICAL ACCURACY RULES — ABSOLUTE, NO EXCEPTIONS:
1. ALWAYS compute numerical examples to self-verify before writing your answer
2. NEVER skip algebraic steps — show every manipulation
3. State ALL assumptions explicitly (domain, convergence conditions, branches)
4. If you are not 100% certain of a result, say so clearly
5. For calculus: always verify by differentiating your integral result
6. For algebra: always substitute back to verify solutions
7. Box the final answer clearly using: \\[\\boxed{{...}}\\]
8. MAINTAIN CONVERSATION CONTEXT — always refer back to previous questions if relevant

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
6. REMEMBER previous messages in conversation for follow-up questions

CONFIDENCE LEVELS — always end with one of:
[CONFIDENCE: HIGH] — you are certain and verified
[CONFIDENCE: MEDIUM] — standard result, verify if critical
[CONFIDENCE: LOW] — complex problem, please verify independently"""


ASK_ANUPAM_PROMPT = f"""You are Ask Anupam — an all-purpose AI tutor by Anupam Nigam.

ACCURACY RULES:
1. For ANY mathematical calculation: compute carefully, then verify by substituting back
2. NEVER use asterisks in output
3. Show Step 1, Step 2, Step 3... for every math problem
4. Box every final mathematical answer: \\[\\boxed{{...}}\\]
5. ALL math expressions MUST be in proper LaTeX notation
6. If image uploaded: read carefully, state what you see, then solve step by step
7. State confidence: [CONFIDENCE: HIGH/MEDIUM/LOW] at end of math solutions
8. MAINTAIN CONTEXT: Always read the full conversation history and answer follow-up questions in context
9. If a follow-up question refers to previous work (e.g., "what about x=2?" or "now integrate this"), connect it to the previous answer

TONE: Friendly, precise, confident. Like a helpful older sibling who is a math professor."""


FORMULA_PROMPT = """You are a mathematical formula sheet generator. Your ONLY job is to output a CLEAN, COMPLETE formula sheet.

STRICT RULES:
1. Output ONLY formulas — no lengthy explanations
2. Each formula must be in LaTeX: \\[ formula \\] or \\( formula \\)
3. One short label per formula (3-5 words max)
4. Group by sub-topic with a heading
5. ABSOLUTELY NO paragraphs, NO long descriptions
6. Include ALL standard formulas for the topic and exam level
7. Format example:
   DERIVATIVES
   \\[ \\frac{d}{dx}(x^n) = nx^{n-1} \\]  Power Rule
   \\[ \\frac{d}{dx}(\\sin x) = \\cos x \\]  Sine Derivative
   
8. Be COMPREHENSIVE — minimum 25 formulas
9. No introduction text, no conclusion text — just the formulas"""


CONCEPT_MAP_PROMPT = """You are a mathematical concept map generator. Create a STRUCTURED, VISUAL concept map.

FORMAT (strictly follow this):
Use this exact structure for each node:

CORE CONCEPT
└── [Precise definition in 1 line with LaTeX formula]

PREREQUISITES
├── [Concept 1] → [Why needed]
├── [Concept 2] → [Why needed]

KEY SUB-TOPICS
├── [Sub-topic 1]
│   ├── Definition: [LaTeX]
│   ├── Key Formula: \\[ formula \\]
│   └── Example: [one line]
├── [Sub-topic 2]
│   ├── Definition: [LaTeX]
│   └── Key Formula: \\[ formula \\]

CONNECTIONS
├── [This concept] → uses → [Other concept]
├── [This concept] → leads to → [Advanced topic]

KEY THEOREMS
├── [Theorem name]: \\[ statement \\]

REAL APPLICATIONS
├── [Application 1]: [Company/field]
├── [Application 2]: [Company/field]

EXAM RELEVANCE (JAM/GATE/CSIR)
└── [Which sections, what weightage]

Rules:
- ALL definitions must include LaTeX formulas
- Keep each line CONCISE (one line per item)
- Use tree structure with ├── └── │
- No long paragraphs anywhere"""


GRAPH_PROMPT = """You are a mathematical graph assistant. When given a function or equation to plot:
1. Identify the function type
2. Give key features: domain, range, asymptotes, intercepts, critical points
3. Provide the function in a format suitable for plotting
4. Describe the shape briefly
All math in LaTeX."""


# ══════════════════════════════════════════════════
# STATIC DATA
# ══════════════════════════════════════════════════

MATHEMATICIANS = {
    "Srinivasa Ramanujan": {
        "period": "1887–1920", "country": "India",
        "fields": ["Number Theory", "Infinite Series", "Modular Forms"],
        "contribution": "Discovered 3900+ results with almost no formal training. His partition function work is used in string theory and black hole physics today.",
        "keyresults": "Ramanujan tau function, Hardy-Ramanujan number 1729, Rogers-Ramanujan identities, mock theta functions",
        "quote": "An equation for me has no meaning unless it expresses a thought of God",
        "image": "https://upload.wikimedia.org/wikipedia/commons/0/02/Srinivasa_Ramanujan_-_OPC_-_1.jpg",
        "impact": "Black hole physics, string theory, partition function applications",
        "wiki": "https://en.wikipedia.org/wiki/Srinivasa_Ramanujan"
    },
    "Leonhard Euler": {
        "period": "1707–1783", "country": "Switzerland",
        "fields": ["Analysis", "Graph Theory", "Number Theory", "Topology"],
        "contribution": "Most prolific mathematician ever: 800+ papers. Founded graph theory, created e, π, i notation, solved Basel problem.",
        "keyresults": "Euler identity \\(e^{i\\pi}+1=0\\), Euler formula V-E+F=2, Basel problem \\(\\sum 1/n^2 = \\pi^2/6\\)",
        "quote": "Mathematics is the queen of sciences",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d7/Leonhard_Euler.jpg",
        "impact": "Internet networking, electrical engineering, quantum mechanics",
        "wiki": "https://en.wikipedia.org/wiki/Leonhard_Euler"
    },
    "Carl Friedrich Gauss": {
        "period": "1777–1855", "country": "Germany",
        "fields": ["Number Theory", "Statistics", "Differential Geometry", "Algebra"],
        "contribution": "Prince of Mathematics. Proved Fundamental Theorem of Algebra at age 21.",
        "keyresults": "FTA, bell curve \\(N(\\mu,\\sigma^2)\\), Gauss-Bonnet theorem, quadratic reciprocity",
        "quote": "Mathematics is the queen of the sciences and number theory is the queen of mathematics",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Gauss_1840_by_Jensen.jpg",
        "impact": "MRI scanners, GPS systems, machine learning",
        "wiki": "https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss"
    },
    "Emmy Noether": {
        "period": "1882–1935", "country": "Germany",
        "fields": ["Abstract Algebra", "Theoretical Physics", "Ring Theory"],
        "contribution": "Revolutionised abstract algebra. Noether's theorem connecting symmetry to conservation laws.",
        "keyresults": "Noether's Theorem, Noetherian rings, ascending chain condition",
        "quote": "My methods are really methods of working and thinking",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e9/Emmy_Noether_%281882-1935%29.jpg",
        "impact": "All modern physics, conservation laws, quantum mechanics",
        "wiki": "https://en.wikipedia.org/wiki/Emmy_Noether"
    },
    "Isaac Newton": {
        "period": "1642–1727", "country": "England",
        "fields": ["Calculus", "Physics", "Classical Mechanics"],
        "contribution": "Invented calculus, discovered gravity, formulated three laws of motion.",
        "keyresults": "Calculus, F=ma, universal gravitation \\(F=Gm_1m_2/r^2\\), binomial theorem",
        "quote": "If I have seen further, it is by standing on the shoulders of giants",
        "image": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Principia_Mathematica_1687.jpg",
        "impact": "Aerospace, civil engineering, space exploration",
        "wiki": "https://en.wikipedia.org/wiki/Isaac_Newton"
    },
    "Bernhard Riemann": {
        "period": "1826–1866", "country": "Germany",
        "fields": ["Complex Analysis", "Riemannian Geometry", "Number Theory"],
        "contribution": "Riemann hypothesis (still unsolved). Riemann integral. Differential geometry enabling Einstein's general relativity.",
        "keyresults": "Riemann hypothesis, Riemann integral, Riemann surfaces, zeta function",
        "quote": "If only I had the theorems! Then I could find the proofs easily enough",
        "image": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Georg_Friedrich_Bernhard_Riemann.jpg",
        "impact": "Internet cryptography, general relativity, $1M Millennium Prize unclaimed",
        "wiki": "https://en.wikipedia.org/wiki/Bernhard_Riemann"
    },
    "Alan Turing": {
        "period": "1912–1954", "country": "England",
        "fields": ["Computability Theory", "Cryptography", "Artificial Intelligence"],
        "contribution": "Father of computer science. Turing machine defines computation. Cracked Enigma in WWII.",
        "keyresults": "Turing machine, halting problem undecidability, Turing test",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Alan_Turing_Aged_16.jpg",
        "impact": "All computation, software industry, AI, cybersecurity",
        "wiki": "https://en.wikipedia.org/wiki/Alan_Turing"
    },
    "Terence Tao": {
        "period": "1975–present", "country": "Australia",
        "fields": ["Number Theory", "Harmonic Analysis", "PDE", "Combinatorics"],
        "contribution": "Fields Medal 2006. Solved Green-Tao theorem on primes in arithmetic progressions.",
        "keyresults": "Green-Tao theorem, compressed sensing, Navier-Stokes progress",
        "quote": "What mathematics achieves is remarkable — it describes all patterns of the universe",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Terence_Tao.jpg",
        "impact": "Medical imaging, signal processing, AI mathematics",
        "wiki": "https://en.wikipedia.org/wiki/Terence_Tao"
    },
    "Maryam Mirzakhani": {
        "period": "1977–2017", "country": "Iran",
        "fields": ["Differential Geometry", "Topology", "Teichmüller Theory"],
        "contribution": "FIRST WOMAN to win Fields Medal (2014). Revolutionary work on dynamics of Riemann surfaces.",
        "keyresults": "Weil-Petersson volumes, moduli space dynamics",
        "quote": "The beauty of mathematics only shows itself to more patient followers",
        "image": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Maryam_Mirzakhani.jpg",
        "impact": "String theory, quantum gravity, inspiring millions",
        "wiki": "https://en.wikipedia.org/wiki/Maryam_Mirzakhani"
    },
    "Kurt Gödel": {
        "period": "1906–1978", "country": "Austria/USA",
        "fields": ["Mathematical Logic", "Set Theory", "Foundations"],
        "contribution": "Incompleteness theorems proved some truths are unprovable within any consistent formal system.",
        "keyresults": "First and Second Incompleteness Theorems, constructible universe L",
        "quote": "Either mathematics is too big for the human mind, or the human mind is more than a machine",
        "image": "https://upload.wikimedia.org/wikipedia/commons/8/84/KurtGodel.jpg",
        "impact": "Limits of AI, philosophy of mind, halting problem",
        "wiki": "https://en.wikipedia.org/wiki/Kurt_G%C3%B6del"
    },
}

THEOREMS = {
    "Pythagorean Theorem": {
        "statement": "In a right triangle with legs \\(a, b\\) and hypotenuse \\(c\\): \\[a^2 + b^2 = c^2\\]",
        "proof_sketch": "Construct squares on each side. The two large squares have equal area. Rearranging the inner triangles proves the result.",
        "formal_proof": "Let \\(\\triangle ABC\\) have right angle at \\(C\\). Drop altitude \\(CD\\) to hypotenuse. \\(\\triangle ACD \\sim \\triangle ABC\\), giving \\(AC^2 = AD \\cdot AB\\). Similarly \\(BC^2 = DB \\cdot AB\\). Adding: \\(AC^2 + BC^2 = AB^2\\). ✅",
        "applications": "Distance formula in \\(\\mathbb{R}^n\\), complex modulus, GPS triangulation",
        "difficulty": "Basic",
        "exam_relevance": "JAM, GATE, CSIR — geometry, vector spaces, metric spaces"
    },
    "Fundamental Theorem of Calculus": {
        "statement": "If \\(f\\) is continuous on \\([a,b]\\) and \\(F'=f\\), then \\[\\int_a^b f(x)\\,dx = F(b) - F(a)\\]",
        "proof_sketch": "Define \\(G(x) = \\int_a^x f(t)\\,dt\\). Show \\(G'(x) = f(x)\\) using MVT. Then \\(G = F + C\\).",
        "formal_proof": "By MVT: \\(G'(x) = \\lim_{h\\to0}\\frac{1}{h}\\int_x^{x+h}f(t)\\,dt = f(c_h) \\to f(x)\\). ✅",
        "applications": "All integral calculus, physics (work = ∫F·dx), economics",
        "difficulty": "Core",
        "exam_relevance": "CRITICAL for JAM Section A, GATE MA, CSIR Part B — appears every year"
    },
    "Euler's Identity": {
        "statement": "\\[e^{i\\pi} + 1 = 0\\]",
        "proof_sketch": "Taylor series: \\(e^{i\\theta} = \\cos\\theta + i\\sin\\theta\\) (Euler's formula). At \\(\\theta=\\pi\\): \\(e^{i\\pi} = -1\\). ✅",
        "formal_proof": "\\(e^{i\\pi} = \\sum_{n=0}^\\infty \\frac{(i\\pi)^n}{n!} = \\cos\\pi + i\\sin\\pi = -1\\). ✅",
        "applications": "Complex analysis, quantum mechanics, signal processing",
        "difficulty": "Advanced",
        "exam_relevance": "Complex Analysis for JAM, GATE, CSIR"
    },
    "Cauchy-Schwarz Inequality": {
        "statement": "\\[|\\langle u, v \\rangle|^2 \\leq \\langle u, u \\rangle \\cdot \\langle v, v \\rangle\\]",
        "proof_sketch": "Consider \\(f(t) = \\langle u - tv, u - tv \\rangle \\geq 0\\). Discriminant of this quadratic \\(\\leq 0\\).",
        "formal_proof": "Setting \\(t = \\frac{\\langle u,v\\rangle}{\\|v\\|^2}\\): \\(0 \\leq \\|u\\|^2 - \\frac{|\\langle u,v\\rangle|^2}{\\|v\\|^2}\\). ✅",
        "applications": "Quantum mechanics (Heisenberg uncertainty), statistics, ML",
        "difficulty": "Intermediate",
        "exam_relevance": "EXTREMELY important for JAM, GATE, CSIR — linear algebra and functional analysis"
    },
    "Banach Fixed Point Theorem": {
        "statement": "Let \\((X, d)\\) be complete and \\(T: X \\to X\\) a contraction (\\(k < 1\\)). Then \\(T\\) has a UNIQUE fixed point.",
        "proof_sketch": "Start \\(x_0\\), iterate \\(x_{n+1}=T(x_n)\\). Sequence is Cauchy → converges to \\(x^*\\) with \\(T(x^*)=x^*\\).",
        "formal_proof": "Uniqueness: \\(d(p,q) = d(Tp,Tq) \\leq k\\,d(p,q)\\) with \\(k<1\\) forces \\(d(p,q)=0\\). ✅",
        "applications": "Newton-Raphson convergence, Picard's ODE existence, iterative solvers",
        "difficulty": "Hard",
        "exam_relevance": "CSIR Part B/C, GATE — functional analysis, metric spaces"
    },
    "Prime Number Theorem": {
        "statement": "\\[\\pi(x) \\sim \\frac{x}{\\ln x} \\quad \\text{as } x \\to \\infty\\]",
        "proof_sketch": "Via Riemann zeta function \\(\\zeta(s)\\). Non-vanishing of \\(\\zeta(s)\\) on \\(\\text{Re}(s)=1\\) proved by Hadamard (1896).",
        "formal_proof": "Key: \\(\\zeta(1+it) \\neq 0\\) for \\(t \\neq 0\\), proved using \\(3+4\\cos\\theta+\\cos 2\\theta \\geq 0\\).",
        "applications": "RSA key generation, prime density, cryptographic protocols",
        "difficulty": "Very Hard",
        "exam_relevance": "CSIR Part C, number theory sections"
    }
}

COMPETITION_PROBLEMS = {
    "IMO": [
        {"year": 2023, "number": 1, "problem": "Determine all composite integers \\(n>1\\) that satisfy the following property: if \\(d_1, d_2, \\ldots, d_k\\) are all the positive divisors of \\(n\\) with \\(1 = d_1 < d_2 < \\cdots < d_k = n\\), then \\(d_i\\) divides \\(d_{i+1} + d_{i+2}\\) for every \\(1 \\leq i \\leq k-2\\).", "difficulty": "Hard", "hint": "Try small composites. Check \\(n = 4, 8, p^2\\). The key is what happens with prime power divisors.", "answer": "All prime squares \\(p^2\\) and \\(p^3\\) for prime \\(p\\), plus \\(n=1\\)."},
        {"year": 2022, "number": 2, "problem": "Let \\(\\mathbb{R}^+\\) denote the set of positive reals. Find all functions \\(f: \\mathbb{R}^+ \\to \\mathbb{R}^+\\) such that for each \\(x \\in \\mathbb{R}^+\\), there is exactly one \\(y \\in \\mathbb{R}^+\\) satisfying \\[xf(y) + yf(x) \\leq 2\\]", "difficulty": "Hard", "hint": "Try \\(f(x) = 1/x\\). Substitute and check AM-GM.", "answer": "\\(f(x) = \\frac{1}{x}\\) is the only solution."},
        {"year": 2021, "number": 1, "problem": "Let \\(n \\geq 100\\) be an integer. Ivan writes the numbers \\(n, n+1, \\ldots, 2n\\) each on different cards. He then shuffles these \\(n+1\\) cards, and divides them into two non-empty groups. Prove that at least one of the two groups contains two cards whose numbers sum to a perfect square.", "difficulty": "Medium", "hint": "Use pigeonhole. Find pairs that sum to squares among \\(\\{n, n+1, \\ldots, 2n\\}\\).", "answer": "Use that consecutive integers \\((2k-1)^2\\) and \\((2k)^2\\) both lie in range \\([2n, 4n]\\)."},
    ],
    "PUTNAM": [
        {"year": 2023, "session": "A1", "problem": "For a positive integer \\(n\\), let \\(f(n)\\) be the number of pairs \\((a,b)\\) of positive integers such that \\[\\frac{a}{b} = \\frac{n+2}{n}\\] Compute \\(\\sum_{n=1}^{100} f(n)\\).", "difficulty": "Medium", "hint": "Write \\(\\frac{a}{b}=\\frac{n+2}{n}\\) in lowest terms. The pair count depends on \\(\\gcd(n,n+2) = \\gcd(n,2)\\).", "answer": "The answer is \\(150\\)."},
        {"year": 2022, "session": "B3", "problem": "Let \\(p(x) = x^5 + x^4 + x^3 + x^2 + x + 1\\). Show that for each integer \\(n \\geq 0\\), \\(p(x^n) \\cdot p(x^{n+1}) \\cdots p(x^{n+5})\\) is divisible by \\(p(x^{n+5})\\) in \\(\\mathbb{Z}[x]\\).", "difficulty": "Hard", "hint": "Note \\(p(x) = (x^6-1)/(x-1)\\). Use cyclotomic factorization.", "answer": "Factor via 6th roots of unity and use divisibility of cyclotomic polynomials."},
    ],
    "AIME": [
        {"year": 2024, "number": 1, "problem": "Every morning Aya goes for a \\(9\\)-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of \\(s\\) km/hr, the walk takes 4 hours, including \\(t\\) minutes spent in the coffee shop. When she walks at \\(s+2\\) km/hr, the walk takes 2 hours and 24 minutes, including \\(t\\) minutes in the shop. Suppose Aya decides to walk at \\(s+\\frac{1}{2}\\) km/hr. Find the number of minutes the walk takes, including the \\(t\\) minutes in the shop.", "difficulty": "Medium", "hint": "Set up two equations. Walking time = distance/speed. Solve for \\(s\\) and \\(t\\).", "answer": "\\(204\\) minutes"},
        {"year": 2024, "number": 5, "problem": "Rectangles \\(ABCD\\) and \\(EFGH\\) are drawn such that \\(D, E, C, F\\) are collinear. Also, \\(A, D, H, G\\) all lie on a circle, \\(\\angle{ABC} = \\angle{EFG} = 90°\\), \\(AB = 4\\), \\(BC = 3\\), and \\(EF = 12\\). Find the length of \\(BE\\).", "difficulty": "Hard", "hint": "Use power of a point for the circle through \\(A, D, H, G\\).", "answer": "\\(BE = \\frac{30}{\\sqrt{13}}\\)"},
    ]
}

REALWORLD = [
    {"concept": "Fourier Transform", "application": "MRI Medical Imaging", "explanation": "MRI scanners record raw k-space data — Fourier-encoded radio frequency signals from hydrogen atoms. The Fourier Transform reconstructs this into detailed 3D anatomical images. The key formula is \\(\\hat{f}(\\xi) = \\int_{-\\infty}^{\\infty} f(x)e^{-2\\pi i x\\xi}\\,dx\\).", "companies": "Siemens Healthineers, GE Healthcare, Philips", "impact": "~100M MRI scans globally per year. Diagnoses cancer, brain tumours without radiation.", "salary": "Biomedical Engineer ₹12L+ | MRI Physicist $120K+"},
    {"concept": "Eigenvalues & Linear Algebra", "application": "Google PageRank", "explanation": "Google models the web as a directed graph. PageRank is the principal eigenvector of a stochastic matrix \\(A\\). Power iteration: \\(\\mathbf{r}_{k+1} = A\\mathbf{r}_k\\) until convergence.", "companies": "Google, Microsoft Bing, Baidu", "impact": "8.5 billion searches per day.", "salary": "Search Engineer $145K+ | Ranking Scientist $160K+"},
    {"concept": "Differential Equations (SIR Model)", "application": "Epidemic Modelling", "explanation": "\\(\\frac{dS}{dt} = -\\beta SI\\), \\(\\frac{dI}{dt} = \\beta SI - \\gamma I\\), \\(\\frac{dR}{dt} = \\gamma I\\). Basic reproduction number \\(R_0 = \\beta/\\gamma\\) determines pandemic vs endemic.", "companies": "WHO, CDC, ICMR, NIH", "impact": "Shaped COVID-19 lockdown decisions. Estimated 10–50M lives saved.", "salary": "Epidemiologist ₹8L+ | Public Health Analyst $100K+"},
    {"concept": "Number Theory & RSA", "application": "Internet Security (HTTPS)", "explanation": "RSA: choose primes \\(p, q\\). Set \\(n=pq\\), \\(\\phi(n)=(p-1)(q-1)\\). Public key \\(e\\), private \\(d = e^{-1} \\pmod{\\phi(n)}\\). Encrypt: \\(c = m^e \\pmod{n}\\).", "companies": "Apple, Google, Amazon, all banks", "impact": "Protects every HTTPS transaction. Global e-commerce \\$6T+/year.", "salary": "Cryptographer ₹15L+ | Security Engineer $135K+"},
    {"concept": "Optimisation & Lagrange Multipliers", "application": "Portfolio Optimisation", "explanation": "Minimise variance \\(\\sigma_p^2 = \\mathbf{w}^T\\Sigma\\mathbf{w}\\) subject to \\(\\mathbf{w}^T\\mathbf{\\mu} = \\mu_p\\) and \\(\\mathbf{1}^T\\mathbf{w}=1\\). The efficient frontier is a parabola in \\((\\sigma, \\mu)\\) space.", "companies": "Goldman Sachs, BlackRock, Citadel, Renaissance Technologies", "impact": "Controls $100+ trillion in global financial assets.", "salary": "Quant Analyst ₹25L+ | Portfolio Manager $300K+"},
]

RESEARCH_HUB = {
    "Pure Mathematics": ["Analytic Number Theory and the Riemann Hypothesis", "Abstract Algebra: Groups, Rings, Fields and Galois Theory", "Algebraic Topology and Homotopy Theory", "Differential Geometry and Riemannian Manifolds", "Algebraic Geometry (Schemes, Sheaves, Cohomology)", "Category Theory and Homological Algebra", "Mathematical Logic, Model Theory and Set Theory", "Representation Theory of Lie Groups"],
    "Applied Mathematics": ["Numerical Methods for PDEs (FEM, FDM, Spectral)", "Convex and Non-Convex Optimisation", "Dynamical Systems, Ergodic Theory and Chaos", "Fluid Dynamics (Navier-Stokes, Turbulence)", "Mathematical Biology (Reaction-Diffusion)", "Financial Mathematics (Stochastic Calculus, Black-Scholes)", "Control Theory and Optimal Control", "Mathematical Imaging and Compressed Sensing"],
    "Probability and Statistics": ["Stochastic Processes and Brownian Motion", "Statistical Learning Theory", "Bayesian Non-Parametrics", "High-Dimensional Statistics and Random Matrix Theory", "Causal Inference", "Information Theory (Shannon Entropy)", "Extreme Value Theory", "Spatial Statistics"],
    "Computational Mathematics": ["Quantum Algorithms (Shor, Grover, HHL)", "Algorithmic Game Theory", "Topological Data Analysis", "Geometric Deep Learning (Graph Neural Networks)", "Scientific Machine Learning (Physics-Informed NNs)", "Symbolic Computation and CAS", "High Performance Computing"],
}

EXAM_INFO = {
    "JAM": {"full_name": "IIT JAM Mathematics", "conducting_body": "IITs (rotational)", "eligibility": "Bachelor's degree with Mathematics in at least 2 years", "pattern": "3 hours · 60 questions · 100 marks\nSection A: 30 MCQ (1 & 2 marks, ⅓ negative)\nSection B: 10 MSQ (2 marks, NO negative)\nSection C: 20 NAT (1 & 2 marks, NO negative)", "syllabus": "Real Analysis · Linear Algebra · Calculus · Differential Equations · Group Theory · Complex Analysis · Numerical Analysis · Statistics", "weightage": "Real Analysis 25% · Linear Algebra 20% · Calculus 20% · Group Theory 15% · Statistics 10%", "top_books": ["Rudin — Principles of Mathematical Analysis","Artin — Algebra","Churchill — Complex Variables","Apostol — Calculus Vol 1 & 2"], "strategy": "Solve 15 years PYQs. Strong in Real Analysis = 60% of rank. Take 1 full mock test per week in last 3 months.", "website": "https://jam.iitd.ac.in"},
    "GATE": {"full_name": "GATE Mathematics (MA)", "conducting_body": "IITs / IISc (rotational)", "eligibility": "Bachelor's in Mathematics/Statistics/CS or related", "pattern": "3 hours · 65 questions · 100 marks\nGeneral Aptitude: 15 marks\nCore MA: 85 marks (MCQ + MSQ + NAT)", "syllabus": "Calculus · Linear Algebra · Real Analysis · Complex Analysis · ODE · PDE · Abstract Algebra · Functional Analysis · Numerical Analysis · Probability", "weightage": "Calculus + LA: 30% · Real Analysis: 20% · Complex: 15% · Algebra: 15%", "top_books": ["Apostol — Calculus","Hoffman-Kunze — Linear Algebra","Conway — Complex Analysis","Dummit-Foote — Abstract Algebra"], "strategy": "NAT questions have no negative marking — attempt all. NPTEL videos are excellent.", "website": "https://gate.iitd.ac.in"},
    "CSIR": {"full_name": "CSIR UGC NET Mathematics", "conducting_body": "NTA", "eligibility": "Master's in Mathematics with 55%", "pattern": "3 hours · 200 marks total\nPart A: 20Q (General Science, 30 marks)\nPart B: 40Q (Core math, 70 marks, ⅓ negative)\nPart C: 60Q (Advanced, 100 marks, proof-based)", "syllabus": "Analysis (Real+Complex+Functional) · Algebra (Linear+Abstract) · Topology · ODE · PDE · Numerical · Probability", "weightage": "Analysis: 30% · Algebra: 25% · Complex: 20% · Topology: 10%", "top_books": ["Rudin — Real & Complex Analysis","Dummit-Foote — Abstract Algebra","Munkres — Topology","Conway — Functions of One Complex Variable"], "strategy": "Part C is the key differentiator. Master proof writing. JRF = ₹31,000/month for PhD.", "website": "https://csirnet.nta.nic.in"}
}


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


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
        "mathematicians": len(MATHEMATICIANS),
        "theorems": len(THEOREMS),
        "version": "7.0"
    })

# ── Chat (with full memory) ───────────────────────

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
        
        # Keep last 20 messages for full conversation memory
        clean = [{"role": m["role"], "content": str(m["content"])}
                 for m in messages if m.get("role") in ("user","assistant") and m.get("content")]
        if len(clean) > 20: clean = clean[-20:]
        
        system = ASK_ANUPAM_PROMPT if mode == "ask_anupam" else SYSTEM_PROMPT
        answer = ask_ai_with_image(clean, image_b64=image_b64, image_type=image_type, system=system)

        confidence = "HIGH"
        if "[CONFIDENCE: MEDIUM]" in answer: confidence = "MEDIUM"
        elif "[CONFIDENCE: LOW]" in answer: confidence = "LOW"
        answer = re.sub(r'\[CONFIDENCE: (HIGH|MEDIUM|LOW)\]', '', answer).strip()

        return jsonify({"answer": answer, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Formula Sheet (pure formulas, no fluff) ──────

@app.route("/api/formula", methods=["POST"])
def formula():
    data  = request.get_json()
    topic = data.get("topic", "Calculus")
    exam  = data.get("exam", "JAM")
    prompt = f"""Generate a COMPLETE, exam-ready formula sheet.
Topic: {topic}
Exam: {exam}

Output ONLY formulas grouped by sub-topic. No paragraphs. No long explanations.
Each formula: LaTeX on its own line + short label (3-5 words).
Minimum 30 formulas. Start immediately with the first sub-topic heading.
Example format:
DERIVATIVES
\\[ \\frac{{d}}{{dx}}(x^n) = nx^{{n-1}} \\]  Power Rule
\\[ \\frac{{d}}{{dx}}(\\sin x) = \\cos x \\]  Sine Derivative

Now generate for {topic} ({exam}):"""
    return jsonify({"answer": ask_simple(prompt, system=FORMULA_PROMPT)})


# ── Concept Map ──────────────────────────────────

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic", "Calculus")
    prompt = f"""Create a precise mathematical concept map for: {topic}

Use tree structure with ├── └── │ symbols.
Include exact definitions with LaTeX, key formulas, connections to other topics.
Exam relevance for JAM/GATE/CSIR.
Be concise — one line per item. No long paragraphs."""
    return jsonify({"answer": ask_simple(prompt, system=CONCEPT_MAP_PROMPT)})


# ── Projects — Dynamic by topic ──────────────────

@app.route("/api/projects/topic", methods=["POST"])
def projects_by_topic():
    """Generate dynamic projects for any topic the user enters."""
    data = request.get_json()
    topic = data.get("topic", "")
    if not topic:
        return jsonify({"error": "topic required"}), 400
    
    prompt = f"""Generate 5 real-world projects for the mathematics topic: {topic}

Return ONLY valid JSON array. Each item:
{{
  "title": "Project title",
  "difficulty": "Beginner/Intermediate/Advanced",
  "math_used": ["topic1", "topic2"],
  "description": "2-sentence description of what you build",
  "real_world": "Specific company/industry that uses this",
  "steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
  "tools": ["Python/NumPy", "etc"],
  "salary": "Relevant job + salary range",
  "wiki_search": "Wikipedia search term for the core concept"
}}

No markdown, no extra text. Valid JSON only."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data_out = parse_json_block(raw)
    if isinstance(data_out, list) and data_out:
        return jsonify({"projects": data_out, "topic": topic})
    return jsonify({"projects": [], "topic": topic, "raw": raw})


@app.route("/api/projects", methods=["GET"])
def projects_list():
    """Default projects list."""
    default_topics = ["Machine Learning", "Cryptography", "Computer Graphics", "Signal Processing", "Finance"]
    projects = []
    for i, t in enumerate(default_topics):
        projects.append({
            "id": i+1, "title": f"{t} with Mathematics",
            "difficulty": "Intermediate", "math": ["Linear Algebra", "Calculus"],
            "desc": f"Apply mathematical concepts to {t}",
            "real": "Industry applications", "companies": "Tech companies",
            "salary": "₹15-40L+ | $120K+"
        })
    return jsonify({"projects": projects, "total": len(projects)})

@app.route("/api/project/<int:pid>", methods=["POST"])
def project_detail(pid):
    prompt = f"""Explain a mathematics project for project #{pid}.
Give:
- Overview
- Required mathematical concepts with LaTeX formulas
- Step-by-step implementation guide
- Real companies using this
- Learning resources (name + search term for Wikipedia/Google)
Be detailed and include all formulas."""
    return jsonify({"explanation": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Books — with level selector ──────────────────

@app.route("/api/books", methods=["POST"])
def books_search():
    d = request.get_json() or {}
    topic = (d.get("topic") or "").strip()
    level = (d.get("level") or "undergraduate").strip()
    exam  = (d.get("exam") or "").strip()
    
    if not topic:
        return jsonify({"needs_input": True, "message": "Please provide a topic"})
    
    prompt = f"""List the best books for:
Topic: {topic}
Level: {level}
Exam: {exam or 'General study'}

Return ONLY valid JSON array (8-12 items):
[{{"name": "Full title", "author": "Author name", "level": "beginner/undergraduate/graduate/research", "why": "One sentence why this book", "isbn": "ISBN if known", "search_term": "Google/Amazon search term to find this book"}}]

No markdown, no extra text."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data_out = parse_json_block(raw)
    if isinstance(data_out, list) and data_out:
        return jsonify({"books": data_out, "total": len(data_out), "topic": topic, "level": level})
    return jsonify({"books": [], "total": 0})


# ── Graph data endpoint ──────────────────────────

@app.route("/api/graph", methods=["POST"])
def graph_data():
    """Generate data points for 2D/3D plotting."""
    if not SYMPY_AVAILABLE:
        # Fallback: ask AI to describe the graph
        data = request.get_json()
        expr = data.get("expression", "x**2")
        prompt = f"""For the function f(x) = {expr}, describe:
1. Domain and range
2. Key features (asymptotes, intercepts, max/min)
3. Shape description
All math in LaTeX."""
        return jsonify({"sympy": False, "description": ask_simple(prompt, system=GRAPH_PROMPT)})
    
    data = request.get_json()
    expr_str = data.get("expression", "x**2")
    graph_type = data.get("type", "2d")  # 2d or 3d
    x_min = float(data.get("x_min", -5))
    x_max = float(data.get("x_max", 5))
    y_min = float(data.get("y_min", -5))
    y_max = float(data.get("y_max", 5))
    
    try:
        x = Symbol("x")
        y = Symbol("y")
        
        if graph_type == "3d":
            # Parse expression in x and y
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
            # 2D plot
            f = safe_parse(re.sub(r'\^', '**', expr_str.strip()))
            if f is None:
                return jsonify({"error": "Could not parse expression"}), 400
            
            points = []
            num_pts = 200
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
            
            # Compute derivative
            try:
                df = diff(f, x)
                df_latex = sp_latex(simplify(df))
                crit = [float(c) for c in solve(df, x) if c.is_real and x_min <= float(c) <= x_max]
            except:
                df_latex = ""
                crit = []
            
            return jsonify({
                "sympy": True, "type": "2d",
                "points": points,
                "expression": expr_str,
                "latex": sp_latex(f),
                "derivative_latex": df_latex,
                "critical_points": crit
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Mathematician routes ─────────────────────────

@app.route("/api/mathematician")
def mathematician_random():
    name, d = random.choice(list(MATHEMATICIANS.items()))
    return jsonify({"name": name, **d})

@app.route("/api/mathematicians")
def mathematician_list():
    return jsonify({"mathematicians": [{"name": n, "period": d["period"], "country": d["country"], "fields": d["fields"]} for n, d in MATHEMATICIANS.items()], "total": len(MATHEMATICIANS)})

@app.route("/api/mathematician/<name>")
def mathematician_detail(name):
    q = normalize_name(name)
    for n, d in MATHEMATICIANS.items():
        if q and (q in normalize_name(n) or normalize_name(n) in q):
            return jsonify({"name": n, **d})
    # AI fallback
    prompt = f"""Return ONLY valid JSON for mathematician: {name}
Keys: name, period, country, fields (array), contribution, keyresults, quote, image (Wikipedia URL), impact, wiki (Wikipedia URL)."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data_out = parse_json_block(raw)
    if isinstance(data_out, dict) and data_out.get("name"):
        return jsonify(data_out)
    return jsonify({"error": "Not found"}), 404


# ── Theorem routes ───────────────────────────────

@app.route("/api/theorems")
def theorems_list():
    return jsonify({"theorems": list(THEOREMS.keys()), "total": len(THEOREMS)})

@app.route("/api/theorem/<name>")
def theorem_detail(name):
    for n, d in THEOREMS.items():
        if name.lower() in n.lower():
            return jsonify({"name": n, **d})
    return jsonify({"error": "Not found"}), 404


# ── Competition routes ───────────────────────────

@app.route("/api/competition/<cat>")
def competition(cat):
    probs = COMPETITION_PROBLEMS.get(cat.upper(), [])
    if not probs: return jsonify({"error": "Category not found. Use IMO, PUTNAM, or AIME"}), 404
    return jsonify({"category": cat.upper(), "problems": probs, "total": len(probs)})


# ── Quiz ─────────────────────────────────────────

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    d = request.get_json()
    prompt = f"""Generate ONE rigorous MCQ for topic: {d.get('topic','Calculus')}
Difficulty: {d.get('difficulty','medium')}
Question {d.get('q_num',1)} of {d.get('total',5)}

EXACT FORMAT (no deviation):
Q: [question text with LaTeX]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter A/B/C/D]
EXPLANATION: [full step-by-step solution with LaTeX]

ALL math in LaTeX. Make it genuinely challenging."""
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


# ── PYQ ──────────────────────────────────────────

@app.route("/api/pyq")
def pyq():
    exam   = request.args.get("exam", "JAM")
    topics = {"JAM": ["Real Analysis","Linear Algebra","Calculus","Group Theory","Complex Analysis"],
              "GATE": ["Calculus","Linear Algebra","Complex Analysis","PDE","ODE"],
              "CSIR": ["Real Analysis","Topology","Algebra","Functional Analysis","Complex Analysis"]}
    topic  = random.choice(topics.get(exam, topics["JAM"]))
    year   = random.randint(2015, 2024)
    prompt = f"""Generate a realistic {exam} PYQ for {topic} (~year {year}).

FORMAT (exact):
Question: [challenging problem with LaTeX]
Solution: [complete step-by-step with ALL LaTeX formulas, box the final answer]
Key Concept: [theorem tested]
Exam Tip: [approach for similar questions]"""
    raw   = ask_simple(prompt, system=SYSTEM_PROMPT)
    lines = raw.split('\n')
    q = next((l.replace("Question:","").strip() for l in lines if l.startswith("Question:")), raw[:300])
    a = next((l.replace("Solution:","").strip()  for l in lines if l.startswith("Solution:")), "See full answer above.")
    return jsonify({"q": q, "a": a, "topic": topic, "year": year, "exam": exam})


# ── Challenge ────────────────────────────────────

@app.route("/api/challenge")
def challenge():
    challenges = [
        "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
        "Find all critical points of \\(f(x)=x^3-3x+2\\) and classify them.",
        "Compute eigenvalues and eigenvectors of \\(A=\\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}\\).",
        "Evaluate \\(\\int x^2 e^x\\,dx\\) using integration by parts.",
        "Solve \\(\\frac{dy}{dx}+2y=4x\\) with \\(y(0)=1\\).",
        "Prove the Cauchy-Schwarz inequality in \\(\\mathbb{R}^n\\).",
        "Show every convergent sequence is a Cauchy sequence.",
        "Compute \\(\\sum_{n=1}^\\infty\\frac{1}{n^2}\\) using Fourier series.",
        "Prove \\(\\text{tr}(AB)=\\text{tr}(BA)\\) for any \\(n\\times n\\) matrices.",
        "Find the radius of convergence of \\(\\sum_{n=0}^\\infty \\frac{n!}{n^n} x^n\\).",
        "Prove that a continuous function on \\([a,b]\\) attains its maximum.",
        "Prove that \\(e\\) is irrational.",
    ]
    return jsonify({"challenge": random.choice(challenges)})


# ── Realworld ────────────────────────────────────

@app.route("/api/realworld")
def realworld_random():
    return jsonify(random.choice(REALWORLD))


# ── Research Hub ─────────────────────────────────

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
    prompt = f"""Answer this research mathematics question: {question}

📌 Overview of the research area
🔬 Key open problems (with LaTeX statements)
📐 Core mathematical tools + theorems with LaTeX
💡 Key researchers and their contributions
📚 Recommended papers (title + author, searchable on arXiv/Google Scholar)
🚀 How a student can get started

ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Exam Info ────────────────────────────────────

@app.route("/api/exam/<exam>")
def exam_info(exam):
    return jsonify(EXAM_INFO.get(exam, {"error": "Not found. Use JAM, GATE, or CSIR"}))


# ── LaTeX ────────────────────────────────────────

@app.route("/api/latex", methods=["POST"])
def latex_gen():
    text = request.get_json().get("text", "")
    prompt = f"""Generate professional LaTeX code for: {text}

1. Complete compilable snippet
2. Brief explanation of each command
3. How to compile (pdflatex / Overleaf)"""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Mock test AI solve ───────────────────────────

@app.route("/api/solve-question", methods=["POST"])
def solve_question():
    """Solve a specific mock test question completely."""
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "question required"}), 400
    prompt = f"""A student is stuck on this exam question:

{question}

Provide COMPLETE solution:
📌 Identify the topic
📐 Full step-by-step working (show ALL steps)
✅ Final Answer: \\[\\boxed{{...}}\\]
⚠️ Key insight / trick for this type
💡 Similar question to practice

ALL math in LaTeX. Be thorough."""
    return jsonify({"solution": ask_simple(prompt, system=SYSTEM_PROMPT)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🧮 MathSphere v7.0 — port {port}")
    print(f"   ✅ SymPy: {SYMPY_AVAILABLE}")
    print(f"   ✅ Groq: {GROQ_AVAILABLE}")
    print(f"   ✅ Gemini: {GEMINI_AVAILABLE}")
    print(f"   📺 {TEACHER_YOUTUBE}\n")
    app.run(host="0.0.0.0", port=port, debug=False)