"""
MathSphere Web — Complete Backend
All features from Telegram bot converted to web
Flask + Groq + Gemini fallback
Deploy FREE on Render.com
By Anupam Nigam | youtube.com/@pi_nomenal1729

✅ FIXES APPLIED:
1. Aggressive chat history limiting (Issue #3)
2. Enhanced system prompt directive (Issue #3)
"""

import os
import json
import random
import base64
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── API CLIENTS ────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_AVAILABLE   = bool(GROQ_API_KEY)
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

groq_client   = None
gemini_client = None

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
        from google.genai import types as genai_types
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini connected")
    except Exception as e:
        print(f"⚠️ Gemini init failed: {e}")
        GEMINI_AVAILABLE = False

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# ── AI CORE ───────────────────────────────────────────────────

def ask_ai(messages, system=None):
    if GROQ_AVAILABLE:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        # ✅ FIX #3.1: AGGRESSIVE HISTORY LIMITING
        # Keep only 9 previous messages (4-5 exchanges) instead of 20
        if len(full) > 11:
            full = [full[0]] + full[-9:]
        for model in GROQ_MODELS:
            try:
                resp = groq_client.chat.completions.create(
                    model=model, messages=full, max_tokens=3000, temperature=0.3
                )
                return resp.choices[0].message.content
            except Exception as e:
                err = str(e)
                if any(x in err.lower() for x in ["429","rate_limit","model_not_active","does not exist"]):
                    continue
                raise e

    if GEMINI_AVAILABLE:
        try:
            parts = []
            if system:
                parts.append(f"SYSTEM:\n{system}\n\n")
            parts.append("CONVERSATION:\n")
            for m in messages:
                role = "Student" if m["role"] == "user" else "Assistant"
                parts.append(f"{role}: {m['content']}\n")
            resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents="".join(parts)
            )
            return resp.text
        except Exception as e:
            print(f"Gemini error: {e}")

    return "⚠️ AI temporarily unavailable. Please try again in a moment!"


def ask_simple(prompt, system=None):
    return ask_ai([{"role": "user", "content": prompt}], system=system)


def solve_image_with_gemini(image_b64, mime_type="image/jpeg"):
    if not GEMINI_AVAILABLE:
        return None
    try:
        from google.genai import types as gt
        img_part = gt.Part.from_bytes(data=base64.b64decode(image_b64), mime_type=mime_type)
        txt_part = gt.Part.from_text(text="""You are an expert mathematics teacher for graduation level students.
Look at this image carefully and:
1. Extract the exact mathematical question shown
2. Identify the topic and method to use
3. Solve it completely step by step — show EVERY step clearly
4. Verify your answer
5. State the final answer clearly

MATH FORMATTING RULES (very important):
- Write ALL math expressions using LaTeX syntax wrapped in delimiters
- Inline math: \\( ... \\)   e.g. \\( x^2 + 3x + 2 = 0 \\)
- Display math (equations on their own line): \\[ ... \\]
- For fractions use \\frac{a}{b}
- For integrals use \\int, \\int_a^b
- For summation use \\sum_{i=1}^{n}
- For square root use \\sqrt{x}
- For Greek letters use \\alpha, \\beta, \\pi, \\theta etc.
- Never write raw math without LaTeX delimiters

Start with: "Namaste! 🙏 Yeh problem dekh liya maine —"
Always add at the end: MathSphere: https://youtube.com/@pi_nomenal1729""")
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=[img_part, txt_part]
        )
        return resp.text
    except Exception as e:
        print(f"Gemini vision error: {e}")
        return None


def solve_with_sympy(problem_text):
    try:
        import sympy as sp
        x = sp.Symbol('x')
        pl = problem_text.lower()
        if "integrate" in pl or "integral" in pl:
            expr_str = pl.replace("integrate","").replace("integral","").replace("dx","").strip()
            result = sp.integrate(sp.sympify(expr_str), x)
            return f"✅ SymPy Verified Answer: \\( \\int = {sp.latex(result)} + C \\)"
        if "differentiate" in pl or "derivative" in pl:
            expr_str = pl.replace("differentiate","").replace("derivative","").replace("of","").strip()
            result = sp.diff(sp.sympify(expr_str), x)
            return f"✅ SymPy Verified Answer: \\( \\frac{{d}}{{dx}} = {sp.latex(result)} \\)"
        if "solve" in pl and "=" in problem_text:
            eq_str = pl.replace("solve","").strip()
            lhs, rhs = eq_str.split("=",1)
            eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            result = sp.solve(eq, x)
            return f"✅ SymPy Verified Answer: \\( x = {sp.latex(result)} \\)"
        if "simplify" in pl:
            expr_str = pl.replace("simplify","").strip()
            result = sp.simplify(sp.sympify(expr_str))
            return f"✅ SymPy Verified Answer: \\( {sp.latex(result)} \\)"
        return None
    except:
        return None

# ── STATIC DATA ────────────────────────────────────────────────

TEACHER_YOUTUBE   = "https://youtube.com/@pi_nomenal1729"
TEACHER_INSTAGRAM = "https://instagram.com/pi_nomenal1729"
TEACHER_WEBSITE   = "https://www.anupamnigam.com"

MATHEMATICIANS = [
    {"name":"Srinivasa Ramanujan","period":"1887–1920","country":"India",
     "contribution":"Discovered thousands of formulas in number theory, infinite series and continued fractions — largely without formal training. His notebooks still reveal new mathematics today!",
     "quote":"An equation for me has no meaning unless it expresses a thought of God."},
    {"name":"Leonhard Euler","period":"1707–1783","country":"Switzerland",
     "contribution":"Most prolific mathematician in history. Created graph theory, introduced π, e, i notation. Proved e^(iπ) + 1 = 0 — called the most beautiful equation.",
     "quote":"Mathematics is the queen of sciences."},
    {"name":"Emmy Noether","period":"1882–1935","country":"Germany",
     "contribution":"Revolutionized abstract algebra and theoretical physics. Noether's theorem connects symmetry and conservation laws — fundamental to modern physics.",
     "quote":"My methods are really methods of working and thinking; this is why they have crept in everywhere anonymously."},
    {"name":"Carl Friedrich Gauss","period":"1777–1855","country":"Germany",
     "contribution":"Prince of Mathematics. Contributed to number theory, statistics, analysis, differential geometry. Proved the fundamental theorem of algebra at age 21.",
     "quote":"Mathematics is the queen of the sciences and number theory is the queen of mathematics."},
    {"name":"Aryabhata","period":"476–550 AD","country":"India",
     "contribution":"Calculated π = 3.1416, explained solar and lunar eclipses mathematically, introduced zero and the place value system.",
     "quote":"Just as a boat in water, the earth floats in space."},
    {"name":"Maryam Mirzakhani","period":"1977–2017","country":"Iran",
     "contribution":"First woman to win the Fields Medal (2014). Revolutionized understanding of Riemann surfaces and their moduli spaces.",
     "quote":"The beauty of mathematics only shows itself to more patient followers."},
]

REAL_WORLD_APPS = [
    {"concept":"Fourier Transform","application":"MRI Machines",
     "explanation":"MRI machines use Fourier Transform to convert radio frequency signals into detailed images of your organs. Without this mathematics, modern medical imaging would not exist!"},
    {"concept":"Linear Algebra","application":"Google Search",
     "explanation":"Google's PageRank algorithm uses eigenvectors and matrices to rank billions of webpages. Every search you do involves massive linear algebra computation!"},
    {"concept":"Probability Theory","application":"Weather Forecasting",
     "explanation":"Weather predictions use Bayesian probability and stochastic processes. The 70% rain probability you see is a direct output of probabilistic mathematical models!"},
    {"concept":"Differential Equations","application":"COVID-19 Modelling",
     "explanation":"Governments used SIR differential equations to model pandemic spread and decide lockdown policies. Mathematics literally shaped global COVID response!"},
    {"concept":"Number Theory","application":"Internet Security",
     "explanation":"Every HTTPS website uses RSA encryption based on prime number theory. Your online banking and WhatsApp messages are protected by pure mathematics!"},
    {"concept":"Graph Theory","application":"Google Maps",
     "explanation":"Dijkstra's shortest path algorithm finds your fastest route in Google Maps. Every navigation app runs on mathematical graph theory!"},
    {"concept":"Statistics","application":"Netflix Recommendations",
     "explanation":"Netflix uses collaborative filtering with matrix factorization to recommend shows. Your personalized recommendations are powered by advanced statistics!"},
    {"concept":"Calculus","application":"Rocket Science",
     "explanation":"ISRO uses differential equations and calculus to calculate rocket trajectories. Every satellite launch requires solving complex calculus problems in real time!"},
]

PARADOXES = [
    {"name":"Zeno's Paradox",
     "statement":"Achilles can NEVER overtake a tortoise in a race if the tortoise has a head start. Every time Achilles reaches where the tortoise was, the tortoise has moved ahead. This repeats infinitely!",
     "teaser":"Motion itself is mathematically impossible according to this paradox. Yet here we are, moving!"},
    {"name":"0.999... = 1",
     "statement":"0.999... repeating forever is EXACTLY equal to 1. Not approximately — exactly! Most people refuse to believe this.",
     "teaser":"This looks wrong. It feels wrong. But mathematics proves it is absolutely right!"},
    {"name":"Russell's Paradox",
     "statement":"Consider the set of ALL sets that do NOT contain themselves. Does this set contain itself? If yes it should not. If no it should. This destroyed the foundation of mathematics in 1901!",
     "teaser":"One question broke all of mathematics and forced mathematicians to rebuild from scratch!"},
    {"name":"Hilbert's Infinite Hotel",
     "statement":"A hotel with infinitely many rooms is completely full. A new guest arrives. The manager accommodates them. Then infinitely many guests arrive — all get accommodated!",
     "teaser":"Infinity plus infinity equals infinity. But are all infinities equal?"},
    {"name":"Banach-Tarski Paradox",
     "statement":"You can take a solid sphere, cut it into finite pieces, and reassemble into TWO spheres the same size as the original!",
     "teaser":"Pure mathematics says you can duplicate a ball using only rotations!"},
    {"name":"Cantor's Different Infinities",
     "statement":"Some infinities are bigger than other infinities! The infinity of real numbers is strictly larger than the infinity of natural numbers.",
     "teaser":"Georg Cantor proved this and was called insane. He was right."},
    {"name":"Birthday Paradox",
     "statement":"In a group of just 23 people there is a 50% chance two people share the same birthday. With 70 people: 99.9%!",
     "teaser":"How can 23 people out of 365 days give 50% probability?"},
    {"name":"Monty Hall Problem",
     "statement":"You pick one of three doors. Host opens a goat door. Should you switch? YES — switching wins 2/3 of the time!",
     "teaser":"Even PhD mathematicians got this wrong when first published!"},
]

DAILY_CHALLENGES = [
    "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
    "If \\(f(x) = x^3 - 3x + 2\\), find all critical points and classify them as maxima or minima.",
    "Find eigenvalues and eigenvectors of the matrix \\(\\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}\\).",
    "Evaluate \\(\\int x^2 e^x \\, dx\\) using integration by parts.",
    "If group \\(G\\) has order 15, prove \\(G\\) is cyclic.",
    "Find the radius of convergence of \\(\\sum_{n=0}^{\\infty} \\frac{x^n}{n!}\\).",
    "Solve: \\(\\frac{dy}{dx} + 2y = 4x\\) with \\(y(0) = 1\\).",
    "Prove AM \\(\\geq\\) GM for positive reals \\(a\\) and \\(b\\).",
    "Find the Fourier series of \\(f(x) = x\\) on \\([-\\pi, \\pi]\\).",
    "Show that every finite integral domain is a field.",
    "Prove that the set of rational numbers \\(\\mathbb{Q}\\) is countable.",
    "Find all solutions of \\(z^4 = 1\\) in \\(\\mathbb{C}\\).",
    "Prove that the continuous image of a compact set is compact.",
    "Evaluate \\(\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n\\) from first principles.",
    "Show that the \\(p\\)-series \\(\\sum \\frac{1}{n^p}\\) converges if and only if \\(p > 1\\).",
    "Prove that every subgroup of a cyclic group is cyclic.",
]

PYQ_BANK = {
    "JAM": [
        {"q":"Let \\(f: \\mathbb{R} \\to \\mathbb{R}\\) be defined by \\(f(x) = x^2 \\sin(1/x)\\) for \\(x \\neq 0\\) and \\(f(0) = 0\\). Is \\(f\\) differentiable at \\(x = 0\\)?",
         "a":"TRUE — \\(f'(0) = \\lim_{h \\to 0} h \\sin(1/h) = 0\\) since \\(|h \\sin(1/h)| \\leq |h| \\to 0\\)","topic":"Real Analysis","year":"2023"},
        {"q":"The number of group homomorphisms from \\(\\mathbb{Z}_{12}\\) to \\(\\mathbb{Z}_8\\) is?",
         "a":"4 — Since \\(\\gcd(12,8) = 4\\), there are exactly 4 homomorphisms","topic":"Algebra","year":"2023"},
        {"q":"Evaluate \\(\\int_0^{\\infty} e^{-x^2} \\, dx\\)",
         "a":"\\(\\frac{\\sqrt{\\pi}}{2}\\) — Using Gaussian integral: \\(\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}\\), so half gives \\(\\frac{\\sqrt{\\pi}}{2}\\)","topic":"Calculus","year":"2022"},
        {"q":"Find eigenvalues of the matrix \\(\\begin{pmatrix} 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ 1 & -3 & 3 \\end{pmatrix}\\)",
         "a":"\\(\\lambda = 1\\) (triple root) — Characteristic polynomial is \\((\\lambda-1)^3 = 0\\)","topic":"Linear Algebra","year":"2022"},
        {"q":"Is the series \\(\\sum \\frac{n^2+1}{n^3+n+1}\\) convergent?",
         "a":"DIVERGENT — Compare with \\(\\frac{1}{n}\\) using limit comparison test, limit \\(= 1 \\neq 0\\)","topic":"Real Analysis","year":"2021"},
        {"q":"The radius of convergence of \\(\\sum n! \\cdot \\frac{x^n}{n^n}\\) is?",
         "a":"\\(e\\) — By ratio test: \\(\\lim|a_{n+1}/a_n| = 1/e\\), so \\(R = e\\)","topic":"Calculus","year":"2021"},
    ],
    "GATE": [
        {"q":"Let \\(T: \\mathbb{R}^3 \\to \\mathbb{R}^3\\) be linear with nullity 1. Vectors \\((1,0,1)\\) and \\((0,1,1)\\) are in null space. Find rank of \\(T\\).",
         "a":"Rank \\(= 2\\) — By rank-nullity theorem: rank + nullity \\(= 3\\), nullity \\(= 1\\), so rank \\(= 2\\)","topic":"Linear Algebra","year":"2023"},
        {"q":"The PDE \\(u_{xx} + 4u_{xy} + 4u_{yy} = 0\\) is classified as?",
         "a":"PARABOLIC — Discriminant \\(B^2 - 4AC = 16 - 16 = 0\\), so parabolic","topic":"PDE","year":"2023"},
        {"q":"Number of onto functions from \\(\\{1,2,3,4\\}\\) to \\(\\{a,b,c\\}\\) is?",
         "a":"36 — Inclusion-exclusion: \\(3^4 - \\binom{3}{1}2^4 + \\binom{3}{2}1^4 = 81 - 48 + 3 = 36\\)","topic":"Combinatorics","year":"2022"},
        {"q":"\\(\\oint_C \\frac{dz}{z^2+1}\\) where \\(C: |z|=2\\) counterclockwise equals?",
         "a":"\\(0\\) — Residues at \\(z=i\\) and \\(z=-i\\) are \\(\\frac{1}{2i}\\) and \\(\\frac{-1}{2i}\\), sum \\(= 0\\)","topic":"Complex Analysis","year":"2022"},
        {"q":"Find the Laplace transform of \\(t \\sin(at)\\)",
         "a":"\\(\\frac{2as}{(s^2+a^2)^2}\\) — Using \\(\\mathcal{L}\\{t f(t)\\} = -\\frac{d}{ds}[F(s)]\\) with \\(F(s) = \\frac{a}{s^2+a^2}\\)","topic":"ODE","year":"2021"},
    ],
    "CSIR": [
        {"q":"Which is NOT a metric on \\(\\mathbb{R}\\)? (a) \\(|x-y|\\) (b) \\(\\frac{|x-y|}{1+|x-y|}\\) (c) \\(|x^2-y^2|\\) (d) \\(\\sqrt{|x-y|}\\)",
         "a":"(c) — \\(d(x,y)=|x^2-y^2|\\) fails triangle inequality for some points","topic":"Topology","year":"2023"},
        {"q":"The group \\((\\mathbb{Z}/n\\mathbb{Z})^*\\) is cyclic for \\(n\\) of the form?",
         "a":"\\(n = 1, 2, 4, p^k, 2p^k\\) where \\(p\\) is an odd prime — these have primitive roots","topic":"Algebra","year":"2023"},
        {"q":"If \\(f\\) is entire and \\(|f(z)| \\leq |z|^2\\) for all \\(z\\), then \\(f(z) = ?\\)",
         "a":"\\(f(z) = az^2\\) for some constant \\(a\\) with \\(|a| \\leq 1\\) — By Cauchy estimates and Liouville","topic":"Complex Analysis","year":"2022"},
        {"q":"A normed space is Banach iff every absolutely convergent series is convergent. True or False?",
         "a":"TRUE — This is a standard characterization theorem of Banach spaces","topic":"Functional Analysis","year":"2022"},
    ]
}

EXAM_INFO = {
    "JAM": {
        "full_name":"Joint Admission Test for Masters (JAM)",
        "conducting_body":"IITs on rotation",
        "eligibility":"Bachelor's degree with Mathematics",
        "pattern":"3 hours, 60 questions, 100 marks\nSection A: 30 MCQs\nSection B: 10 MSQs\nSection C: 20 Numerical",
        "syllabus":"Real Analysis, Linear Algebra, Calculus, Differential Equations, Vector Calculus, Statistics, Probability",
        "weightage":"Real Analysis: 25-30%\nLinear Algebra: 20-25%\nCalculus: 15-20%\nDiff Equations: 10-15%",
        "books":"Rudin for Real Analysis | Gilbert Strang for Linear Algebra | Arora & Sharma for practice",
        "website":"https://jam.iitd.ac.in"
    },
    "GATE": {
        "full_name":"Graduate Aptitude Test in Engineering — Mathematics",
        "conducting_body":"IITs and IISc on rotation",
        "eligibility":"Bachelor's degree in Mathematics",
        "pattern":"3 hours, 65 questions, 100 marks\nGeneral Aptitude: 15 marks\nMathematics: 85 marks",
        "syllabus":"Calculus, Linear Algebra, Real Analysis, Complex Analysis, Algebra, ODE, PDE, Probability, Statistics",
        "weightage":"Calculus + Linear Algebra: 35-40%\nReal + Complex Analysis: 20-25%\nAlgebra: 15%",
        "books":"Kreyszig for Engineering Mathematics | Dummit & Foote for Abstract Algebra",
        "website":"https://gate.iitd.ac.in"
    },
    "CSIR": {
        "full_name":"CSIR UGC NET Mathematical Sciences",
        "conducting_body":"National Testing Agency (NTA)",
        "eligibility":"MSc Mathematics with minimum 55%",
        "pattern":"3 hours, 3 parts\nPart A: General Aptitude 30 marks\nPart B: Core Maths 70 marks\nPart C: Advanced Maths 100 marks",
        "syllabus":"Analysis, Linear Algebra, Complex Analysis, Algebra, Topology, Differential Equations, Probability",
        "weightage":"Analysis + Topology: 30-35%\nAlgebra + Linear Algebra: 25-30%\nComplex Analysis: 15-20%",
        "books":"Munkres for Topology | Kreyszig for Functional Analysis | Royden for Measure Theory",
        "website":"https://csirnet.nta.nic.in"
    }
}

# ── PROMPTS ────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are MathSphere — a warm, expert Mathematics teacher for graduation level students, created by Anupam Nigam (youtube.com/@pi_nomenal1729).

═══════════════════════════════════════
LANGUAGE & TONE (VERY IMPORTANT):
═══════════════════════════════════════
- ALWAYS start your FIRST response in a conversation with a warm Hinglish greeting like:
  "Namaste! 🙏 Aao, aaj yeh concept milke samjhte hain!" or
  "Arre waah! Bahut achha question hai yeh! Chalo dekhte hain..." or
  "Haan haan! Yeh topic bohot important hai — samjho dhyan se!"
- Throughout responses, naturally mix Hindi/Hinglish phrases like:
  "Dekho...", "Samajh aaya?", "Yeh important hai!", "Bohot achha!",
  "Dhyan rakhna", "Simple hai — suno", "Yaad rakho yeh point"
- Keep it mostly English but with warm Hindi touches — like a friendly Indian teacher
- For casual messages ("thanks", "ok") → brief Hinglish response only
- NEVER be robotic or formal — be warm like a real desi teacher

═══════════════════════════════════════
MATH FORMATTING (CRITICAL — FOLLOW EXACTLY):
═══════════════════════════════════════
- ALL mathematical expressions MUST use LaTeX syntax
- Inline math → wrap in \\( ... \\)   Example: \\( x^2 + 3x + 2 = 0 \\)
- Display equations → wrap in \\[ ... \\]  Example: \\[ \\int_0^1 x^2 dx = \\frac{{1}}{{3}} \\]
- NEVER write raw math like x^2 or sqrt(x) — always use LaTeX
- Use: \\frac{{a}}{{b}}, \\sqrt{{x}}, \\int, \\sum, \\lim, \\infty, \\alpha, \\beta, \\pi, \\theta
- Use \\mathbb{{R}}, \\mathbb{{Z}}, \\mathbb{{Q}}, \\mathbb{{C}} for number sets
- Use \\leq, \\geq, \\neq, \\approx, \\in, \\subset, \\forall, \\exists
- For matrices: \\begin{{pmatrix}} a & b \\\\ c & d \\end{{pmatrix}}

═══════════════════════════════════════
ACCURACY RULES:
═══════════════════════════════════════
- Think step by step BEFORE answering
- Double-check every calculation
- For computations: verify the answer by substitution or another method
- If unsure: say "Let me verify this..." and check again
- Never guess — be mathematically rigorous

═══════════════════════════════════════
TEACHING STRUCTURE (for math questions):
═══════════════════════════════════════

◆━━━━━━━━━━━━━━━━━━◆
📌 [Topic Name]
◆━━━━━━━━━━━━━━━━━━◆

💡 Real Life Analogy: [relatable Indian example]
📖 Definition: [precise mathematical definition with LaTeX]
📐 Step-by-Step Solution: [numbered steps, every step in LaTeX]
✏️ Verification: [verify the answer]
📝 Try This Yourself: [one practice problem]

📚 Resources:
► MathSphere: {TEACHER_YOUTUBE}
◆━━━━━━━━━━━━━━━━━━◆

Always include: {TEACHER_YOUTUBE}"""


QUIZ_PROMPT = """You are a mathematics quiz generator for graduation level students.
Generate ONE multiple choice question.
Topic: {topic}
Difficulty: {difficulty}
Question {q_num} of {total}

IMPORTANT RULES:
- The question must be mathematically correct and well-formed
- All 4 options must be plausible but only ONE is correct
- Use LaTeX for all math: \\( ... \\) for inline, \\[ ... \\] for display
- Double-check that your stated ANSWER is actually correct
- The explanation must clearly justify why the answer is correct

REPLY ONLY IN THIS EXACT FORMAT — nothing else:
Q: [question text with LaTeX math]
A) [option with LaTeX]
B) [option with LaTeX]
C) [option with LaTeX]
D) [option with LaTeX]
ANSWER: [A or B or C or D]
EXPLANATION: [clear one-sentence explanation with LaTeX]"""

PROOF_PROMPT = f"""You are running an interactive Proof Builder session in MathSphere.
You are a friendly Indian math teacher — use Hinglish naturally.

Rules:
- Break proof into clear numbered steps
- Use LaTeX for ALL math expressions: \\( ... \\) inline, \\[ ... \\] for display
- Present hints one step at a time
- If student is correct → "Bilkul sahi! Ekdum correct! ✅ Next step..."
- If student is wrong → "Arre, thoda sochna... Hint: ..."
- At the end, show the complete assembled proof beautifully formatted

Always include: {TEACHER_YOUTUBE}"""

DEBATE_PROMPT = f"""You are hosting Math Debate Club in MathSphere with Hinglish flair.
Start with: "Wah! Interesting argument hai yeh! Chalo isko mathematically check karte hain..."
Engage seriously with arguments. Use LaTeX for all math.
Challenge reasoning warmly. Guide toward mathematical truth.
Always include: {TEACHER_YOUTUBE}"""

CALCULATOR_PROMPT = f"""You are a precise step-by-step mathematical calculator for MathSphere.
You are a friendly Indian math teacher.

For EVERY problem:
1. "Dekho, yeh ek [type] problem hai. Method use karenge: [method]"
2. Show EVERY numbered step with full LaTeX formatting
3. Use \\[ ... \\] for major equations, \\( ... \\) for inline
4. Verify the final answer by substitution or alternative method
5. State: "✅ Final Answer: \\[ ... \\]"

ACCURACY IS PARAMOUNT — double check every calculation.
Always include: {TEACHER_YOUTUBE}"""

FORMULA_PROMPT = f"""You are a formula sheet generator for MathSphere students.
You are a friendly Indian math teacher — add brief Hinglish notes.

Generate a COMPLETE formula sheet with:
- All major formulas, theorems, definitions
- LaTeX formatting for EVERY formula: \\( ... \\) inline, \\[ ... \\] for display equations
- Clear sections with emoji headers
- Brief "Yaad rakho:" tips for important formulas

Always include: {TEACHER_YOUTUBE}"""

LATEX_PROMPT = f"""You are a LaTeX code generator for MathSphere.
Give: complete LaTeX code in code blocks, brief explanation, minimal working example.
Start with: "Haan! Yeh LaTeX code ready hai tumhare liye —"
Compile free at: https://overleaf.com | MathSphere: {TEACHER_YOUTUBE}"""

REVISION_PROMPT = f"""You are doing rapid revision for MathSphere students.
Start with: "Chalo! Quick revision karte hain — dhyan se padho!"
Give TOP 10 most important points for the topic.
Use LaTeX for all math expressions.
Be concise and exam-focused.
End with: "Exam Tips: [3 specific tips]"
Always include: {TEACHER_YOUTUBE}"""

CONCEPT_MAP_PROMPT = f"""You are creating a concept map for MathSphere students.
Start with: "Dekho, yeh topic kaafi connected hai — samjho poora picture!"
Show clearly:
🔙 PREREQUISITES: what you need to know first
🔗 CONNECTS TO: related topics
➡️ LEADS TO: advanced topics this unlocks
🌍 REAL WORLD: applications
Use LaTeX for all math.
Always include: {TEACHER_YOUTUBE}"""

COMPARE_PROMPT = f"""You are comparing mathematical concepts for MathSphere students.
Start with: "Bahut achha question! Log yeh dono confuse karte hain — aaj clear karte hain!"
Give:
- Definitions (with LaTeX)
- Key differences (table format)
- Similarities
- When to use which
- Common student mistakes ("Students aksar yeh galti karte hain...")
Always include: {TEACHER_YOUTUBE}"""

COUNTEREXAMPLE_PROMPT = f"""You are a mathematical claim verifier for MathSphere.
Start with: "Interesting claim hai! Dekho yeh sach hai ya jhooth —"
1. State the claim clearly with LaTeX
2. Either PROVE it rigorously OR find the simplest counterexample
3. Explain why counterexample works
4. State the correct version of the result
Use LaTeX for ALL math.
Always include: {TEACHER_YOUTUBE}"""

PROJECT_PROMPT = f"""You are a real-life math projects guide for MathSphere.
Start with: "Waah! Math ko real life mein apply karna — yeh toh bahut maza aayega!"
Generate 3 detailed project ideas. For each:
- Name and objective
- Mathematical concepts used (with LaTeX)
- Tools needed
- Step-by-step guide
- Expected outcome
Always include: {TEACHER_YOUTUBE}"""

RESEARCH_PROMPT = f"""You are a mathematics research assistant for MathSphere.
Be rigorous and academic. Use LaTeX for all math.
Help with research papers, formal proofs, topic ideas, peer review, citations.
Always include: {TEACHER_YOUTUBE}"""

# ── HELPERS ────────────────────────────────────────────────────

def detect_hindi(text):
    words = ["kya","hai","mujhe","samajh","batao","kaise","kyun","matlab",
             "nahi","haan","theek","accha","bhai","yaar","padh","sikho",
             "समझ","बताओ","कैसे","क्या","है","नहीं","हाँ","पढ़",
             "solve","bata","kar","dedo","chahiye","help"]
    return any(w in text.lower() for w in words)

# ── ROUTES ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","groq":GROQ_AVAILABLE,"gemini":GEMINI_AVAILABLE})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages   = data.get("messages", [])
        mode       = data.get("mode", "normal")
        image_b64  = data.get("image_b64")
        image_type = data.get("image_type", "image/jpeg")

        if not messages:
            return jsonify({"error":"messages required"}), 400

        if image_b64 and GEMINI_AVAILABLE:
            result = solve_image_with_gemini(image_b64, image_type)
            if result:
                return jsonify({"answer": result, "source":"gemini-vision"})

        clean = [{"role":m["role"],"content":str(m["content"])}
                 for m in messages if m.get("role") in ("user","assistant") and m.get("content")]

        # ✅ FIX #3.2: AGGRESSIVE HISTORY LIMITING
        # Keep only 6 last messages (3 exchanges) instead of 20+
        if len(clean) > 8:
            clean = clean[-6:]

        # ✅ FIX #3.3: ENHANCED SYSTEM PROMPT DIRECTIVE
        # Add critical instruction to focus only on current question
        enhanced_system = SYSTEM_PROMPT + """

═══════════════════════════════════════
⚠️ CRITICAL INSTRUCTION FOR THIS CHAT:
═══════════════════════════════════════
- Answer ONLY the current/latest question being asked
- Do NOT repeat or summarize answers from previous messages
- Do NOT reference or re-explain earlier topics unless asked again
- Focus completely on the NEW question only
- Be concise and avoid redundancy"""

        return jsonify({"answer": ask_ai(clean, system=enhanced_system)})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    try:
        d = request.get_json()
        prompt = QUIZ_PROMPT.format(
            topic=d.get("topic","Calculus"), difficulty=d.get("difficulty","medium"),
            q_num=d.get("q_num",1), total=d.get("total",5)
        )
        raw = ask_simple(prompt)
        lines = raw.strip().split('\n')
        ans_line  = next((l for l in lines if l.strip().startswith("ANSWER:")), "ANSWER: A")
        expl_line = next((l for l in lines if l.strip().startswith("EXPLANATION:")), "")
        correct = ans_line.replace("ANSWER:","").strip()[:1].upper()
        explanation = expl_line.replace("EXPLANATION:","").strip()
        question = '\n'.join(l for l in lines if not l.strip().startswith(("ANSWER:","EXPLANATION:")))
        return jsonify({"question":question.strip(), "answer":correct, "explanation":explanation})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/challenge")
def challenge():
    return jsonify({"challenge": random.choice(DAILY_CHALLENGES)})

@app.route("/api/mathematician")
def mathematician():
    return jsonify(random.choice(MATHEMATICIANS))

@app.route("/api/realworld")
def realworld():
    return jsonify(random.choice(REAL_WORLD_APPS))

@app.route("/api/paradox")
def paradox():
    name = request.args.get("name")
    if name:
        p = next((x for x in PARADOXES if x["name"]==name), None)
        return jsonify(p or random.choice(PARADOXES))
    return jsonify(random.choice(PARADOXES))

@app.route("/api/paradoxes")
def all_paradoxes():
    return jsonify(PARADOXES)

@app.route("/api/pyq")
def pyq():
    exam = request.args.get("exam","JAM").upper()
    qs = PYQ_BANK.get(exam, [])
    return jsonify(random.choice(qs)) if qs else jsonify({"error":"Not found"}), 404

@app.route("/api/exam/<exam>")
def exam_info(exam):
    info = EXAM_INFO.get(exam.upper())
    return jsonify(info) if info else (jsonify({"error":"Not found"}), 404)

@app.route("/api/formula", methods=["POST"])
def formula():
    topic = request.get_json().get("topic","")
    return jsonify({"answer": ask_simple(f"Generate a complete formula sheet for: {topic}", system=FORMULA_PROMPT)})

@app.route("/api/calculator", methods=["POST"])
def calculator():
    problem = request.get_json().get("problem","")
    sympy_r = solve_with_sympy(problem)
    answer  = ask_simple(f"Solve this step by step: {problem}", system=CALCULATOR_PROMPT)
    if sympy_r:
        answer = f"{sympy_r}\n\n◆━━━━━━━━━━━━━━━━━━◆\nStep-by-Step Solution:\n\n{answer}"
    return jsonify({"answer": answer})

@app.route("/api/latex", methods=["POST"])
def latex():
    text = request.get_json().get("text","")
    return jsonify({"answer": ask_simple(f"Generate LaTeX code for: {text}", system=LATEX_PROMPT)})

@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic","")
    return jsonify({"answer": ask_simple(f"TOP 10 rapid revision points for: {topic}", system=REVISION_PROMPT)})

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic","")
    return jsonify({"answer": ask_simple(f"Concept map for: {topic}", system=CONCEPT_MAP_PROMPT)})

@app.route("/api/compare", methods=["POST"])
def compare():
    concepts = request.get_json().get("concepts","")
    return jsonify({"answer": ask_simple(f"Compare: {concepts}", system=COMPARE_PROMPT)})

@app.route("/api/verify", methods=["POST"])
def verify():
    claim = request.get_json().get("claim","")
    return jsonify({"answer": ask_simple(f"Verify or find counterexample: {claim}", system=COUNTEREXAMPLE_PROMPT)})

@app.route("/api/projects", methods=["POST"])
def projects():
    domain = request.get_json().get("domain","")
    return jsonify({"answer": ask_simple(f"3 real-life math projects for: {domain}", system=PROJECT_PROMPT)})

@app.route("/api/proof", methods=["POST"])
def proof():
    d = request.get_json()
    msgs = d.get("history",[]) + [{"role":"user","content":d.get("theorem","")}]
    return jsonify({"answer": ask_ai(msgs, system=PROOF_PROMPT)})

@app.route("/api/debate", methods=["POST"])
def debate():
    d = request.get_json()
    msgs = d.get("history",[]) + [{"role":"user","content":d.get("argument","")}]
    return jsonify({"answer": ask_ai(msgs, system=SYSTEM_PROMPT+"\n\n"+DEBATE_PROMPT)})

@app.route("/api/research", methods=["POST"])
def research():
    d = request.get_json()
    return jsonify({"answer": ask_simple(d.get("question",""), system=RESEARCH_PROMPT)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🧮 MathSphere starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)