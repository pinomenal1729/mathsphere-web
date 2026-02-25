"""
MathSphere Web — Complete Backend with ALL 4 FEATURES
Features:
✅ PYQ Mock Tests
✅ Concept Checker
✅ Progress Tracker
✅ Graph Visualization

By Anupam Nigam | youtube.com/@pi_nomenal1729
"""

import os
import json
import random
import base64
import numpy as np
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

# ── VERIFIED DEFINITIONS DATABASE ────────────────────────────

VERIFIED_DEFINITIONS = {
    "Limit": {
        "definition": "A function f(x) approaches limit L as x approaches a if: for every ε > 0, there exists δ > 0 such that |x - a| < δ implies |f(x) - L| < ε",
        "latex": "\\lim_{x \\to a} f(x) = L \\iff \\forall \\varepsilon > 0, \\exists \\delta > 0 : |x-a| < \\delta \\Rightarrow |f(x)-L| < \\varepsilon",
        "source": "Rudin - Principles of Mathematical Analysis",
        "page": "47",
        "edition": "3rd Edition, 1976",
        "verified": True,
        "confidence": "100%",
        "exams": ["JAM", "NET", "GATE", "BOARDS"]
    },
    "Eigenvalue": {
        "definition": "A scalar λ is an eigenvalue of matrix A if there exists a non-zero vector v (eigenvector) such that Av = λv",
        "latex": "A\\mathbf{v} = \\lambda\\mathbf{v}, \\text{ where } \\mathbf{v} \\neq \\mathbf{0}",
        "source": "Gilbert Strang - Linear Algebra and Its Applications",
        "page": "228",
        "edition": "5th Edition, 2016",
        "verified": True,
        "confidence": "100%",
        "exams": ["JAM", "GATE", "NET"]
    },
    "Continuous Function": {
        "definition": "A function f is continuous at point a if lim(x→a) f(x) = f(a). A function is continuous on an interval if it is continuous at every point in that interval.",
        "latex": "f \\text{ is continuous at } a \\iff \\lim_{x \\to a} f(x) = f(a)",
        "source": "Rudin - Principles of Mathematical Analysis",
        "page": "84",
        "edition": "3rd Edition, 1976",
        "verified": True,
        "confidence": "100%",
        "exams": ["JAM", "NET", "BOARDS"]
    },
    "Derivative": {
        "definition": "The derivative of f at x is defined as: f'(x) = lim(h→0) [f(x+h) - f(x)] / h, if this limit exists",
        "latex": "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
        "source": "Rudin - Principles of Mathematical Analysis",
        "page": "103",
        "edition": "3rd Edition, 1976",
        "verified": True,
        "confidence": "100%",
        "exams": ["JAM", "GATE", "NET", "BOARDS"]
    },
    "Linear Independence": {
        "definition": "Vectors v₁, v₂, ..., vₙ are linearly independent if the only solution to c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 is c₁ = c₂ = ... = cₙ = 0",
        "latex": "c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_n\\mathbf{v}_n = \\mathbf{0} \\implies c_1 = c_2 = \\cdots = c_n = 0",
        "source": "Strang - Linear Algebra and Its Applications",
        "page": "89",
        "edition": "5th Edition, 2016",
        "verified": True,
        "confidence": "100%",
        "exams": ["JAM", "GATE", "NET"]
    }
}

# ── STUDY MATERIALS DATABASE ────────────────────────────

STUDY_MATERIALS = {
    "JAM": {
        "Real Analysis": {
            "Limits and Continuity": {
                "theory": "The concept of limit is foundational to calculus. It describes the behavior of a function as the input approaches some value.",
                "key_theorems": [
                    "Epsilon-Delta Definition of Limit",
                    "Squeeze Theorem",
                    "Continuity Theorem",
                    "Intermediate Value Theorem"
                ],
                "practice_problems": [
                    {
                        "id": "RA-LC-001",
                        "question": "Prove using epsilon-delta definition that lim(x→2) (3x + 1) = 7",
                        "difficulty": "medium",
                        "source": "JAM 2022"
                    },
                    {
                        "id": "RA-LC-002",
                        "question": "Find the limit: lim(x→0) sin(x)/x",
                        "difficulty": "easy",
                        "source": "JAM 2021"
                    }
                ],
                "estimated_hours": 8
            },
            "Derivatives": {
                "theory": "Derivative measures the rate of change of a function.",
                "key_theorems": ["Product Rule", "Chain Rule", "Mean Value Theorem"],
                "practice_problems": [],
                "estimated_hours": 6
            }
        },
        "Linear Algebra": {
            "Vector Spaces": {
                "theory": "A vector space is a set of vectors with operations of addition and scalar multiplication.",
                "key_theorems": ["Basis Theorem", "Dimension Theorem"],
                "practice_problems": [],
                "estimated_hours": 7
            }
        }
    },
    "GATE": {
        "Calculus": {
            "Integration": {
                "theory": "Integration is the reverse of differentiation.",
                "key_theorems": ["Fundamental Theorem of Calculus"],
                "practice_problems": [],
                "estimated_hours": 6
            }
        }
    },
    "NET": {
        "Analysis": {
            "Convergence": {
                "theory": "Understanding convergence is key to real analysis.",
                "key_theorems": ["Monotone Convergence Theorem"],
                "practice_problems": [],
                "estimated_hours": 8
            }
        }
    }
}

# ── CONCEPT CHECKER QUESTIONS ────────────────────────────

CONCEPT_QUESTIONS = {
    "JAM": {
        "Limits and Continuity": [
            {
                "question": "What is the epsilon-delta definition of limit?",
                "options": {
                    "A": "For every ε > 0, there exists δ > 0 such that |x-a| < δ implies |f(x)-L| < ε",
                    "B": "For every δ > 0, there exists ε > 0 such that |f(x)-L| < δ implies |x-a| < ε",
                    "C": "f(x) approaches L as x gets very large",
                    "D": "f(x) is continuous at point a"
                },
                "correct": "A"
            },
            {
                "question": "A function is continuous at x=a if:",
                "options": {
                    "A": "lim(x→a) f(x) = f(a)",
                    "B": "f(a) is defined",
                    "C": "f(x) is differentiable at a",
                    "D": "f(x) > 0 for all x near a"
                },
                "correct": "A"
            },
            {
                "question": "What does the Intermediate Value Theorem state?",
                "options": {
                    "A": "If f is continuous on [a,b] and k is between f(a) and f(b), then f(c)=k for some c in [a,b]",
                    "B": "Every function has a limit at every point",
                    "C": "Every continuous function is differentiable",
                    "D": "The derivative exists at every point"
                },
                "correct": "A"
            },
            {
                "question": "What is the Squeeze Theorem?",
                "options": {
                    "A": "If g(x) ≤ f(x) ≤ h(x) and lim g(x) = lim h(x) = L, then lim f(x) = L",
                    "B": "If f is continuous, then f is bounded",
                    "C": "If f(x) → L, then f is differentiable",
                    "D": "Every function has a minimum value"
                },
                "correct": "A"
            },
            {
                "question": "When is a function NOT continuous at a point?",
                "options": {
                    "A": "When the limit does not equal the function value",
                    "B": "When the derivative exists",
                    "C": "When the function is increasing",
                    "D": "When the function is positive"
                },
                "correct": "A"
            }
        ],
        "Derivatives": [
            {
                "question": "What is the formal definition of derivative?",
                "options": {
                    "A": "f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
                    "B": "f'(x) = f(x+1) - f(x)",
                    "C": "f'(x) = [f(b) - f(a)] / (b - a)",
                    "D": "f'(x) = f(x) * 2"
                },
                "correct": "A"
            }
        ]
    },
    "GATE": {
        "Linear Algebra": [
            {
                "question": "What is an eigenvalue?",
                "options": {
                    "A": "A scalar λ such that Av = λv for non-zero vector v",
                    "B": "A vector that is perpendicular to another vector",
                    "C": "The determinant of a matrix",
                    "D": "A number that is always positive"
                },
                "correct": "A"
            },
            {
                "question": "What do linearly independent vectors mean?",
                "options": {
                    "A": "No vector is a linear combination of others",
                    "B": "All vectors are in the same direction",
                    "C": "All vectors have the same magnitude",
                    "D": "The vectors are orthogonal to each other"
                },
                "correct": "A"
            }
        ]
    }
}

# ── TRACKER DATA ────────────────────────────────

TRACKER_DATA = {
    "default": {
        "JAM": {
            "Real Analysis": {
                "Limits and Continuity": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                },
                "Derivatives": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                },
                "Integration": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                }
            },
            "Linear Algebra": {
                "Vector Spaces": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                },
                "Eigenvalues": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                }
            }
        },
        "GATE": {
            "Calculus": {
                "Integration": {
                    "definition_learned": False,
                    "concept_check_passed": False,
                    "practice_done": False,
                    "pyq_attempted": False
                }
            }
        }
    }
}

# ── SYSTEM PROMPT ────────────────────────────────────────────

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

        if len(clean) > 8:
            clean = clean[-6:]

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

# ── FEATURE 1: PYQ MOCK TEST ────────────────────────────

@app.route("/api/pyq-mock/<exam>")
def pyq_mock_test(exam):
    """Get random PYQ questions for mock test"""
    exam_upper = exam.upper()
    
    if exam_upper not in PYQ_BANK:
        return jsonify({"error": "Exam not found"}), 404
    
    questions = PYQ_BANK[exam_upper]
    
    if not questions:
        return jsonify({"error": "No questions found"}), 404
    
    test_questions = questions[:min(10, len(questions))]
    
    mock_test = {
        "test_id": f"{exam_upper}-MOCK-{datetime.now().timestamp()}",
        "exam": exam_upper,
        "duration": 60,
        "total_questions": len(test_questions),
        "questions": [
            {
                "id": f"Q{i+1}",
                "question": q.get("q", "Question not available"),
                "topic": q.get("topic", "General"),
                "year": q.get("year", "Unknown"),
                "source": "PYQ"
            }
            for i, q in enumerate(test_questions)
        ]
    }
    
    return jsonify(mock_test)

@app.route("/api/pyq-submit/<exam>/<test_id>", methods=["POST"])
def submit_pyq_test(exam, test_id):
    """Grade mock test and save results"""
    try:
        data = request.get_json()
        user_answers = data.get("answers", {})
        
        exam_upper = exam.upper()
        questions = PYQ_BANK.get(exam_upper, [])
        
        if not questions:
            return jsonify({"error": "Exam not found"}), 404
        
        score = 0
        total = len(questions[:10])
        weak_areas = []
        strong_areas = []
        
        detailed_results = []
        
        for i, q in enumerate(questions[:10]):
            question_id = f"Q{i+1}"
            correct_answer = q.get("a", "").split()[0]
            user_answer = user_answers.get(question_id, "").upper()
            
            is_correct = user_answer == correct_answer
            
            if is_correct:
                score += 1
                if q.get("topic") not in strong_areas:
                    strong_areas.append(q.get("topic", "General"))
            else:
                if q.get("topic") not in weak_areas:
                    weak_areas.append(q.get("topic", "General"))
            
            detailed_results.append({
                "question_id": question_id,
                "question": q.get("q", ""),
                "topic": q.get("topic", ""),
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "solution": q.get("a", "No solution available")
            })
        
        percentage = (score / total) * 100
        
        result = {
            "test_id": test_id,
            "exam": exam_upper,
            "score": score,
            "total": total,
            "percentage": round(percentage, 2),
            "duration": data.get("duration", 0),
            "date": datetime.now().isoformat(),
            "weak_areas": list(set(weak_areas)),
            "strong_areas": list(set(strong_areas)),
            "detailed_results": detailed_results,
            "status": "PASSED" if percentage >= 70 else "NEEDS IMPROVEMENT",
            "feedback": generate_feedback(percentage, weak_areas)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error submitting test: {e}")
        return jsonify({"error": str(e)}), 500

def generate_feedback(percentage, weak_areas):
    """Generate personalized feedback"""
    if percentage >= 90:
        return f"🌟 Excellent! You scored {percentage}%. Ready for exam!"
    elif percentage >= 70:
        return f"✅ Good! Score {percentage}%. Focus on: {', '.join(weak_areas[:2])}"
    elif percentage >= 50:
        return f"⚠️ Need improvement. Score {percentage}%. Review: {', '.join(weak_areas[:3])}"
    else:
        return f"❌ Low score {percentage}%. Need serious revision. Start with: {weak_areas[0] if weak_areas else 'Basics'}"

# ── FEATURE 2: CONCEPT CHECKER ────────────────────────────

@app.route("/api/concept-check/<exam>/<topic>", methods=["POST"])
def concept_checker(exam, topic):
    """Generate concept verification quiz"""
    exam_upper = exam.upper()
    
    if exam_upper not in CONCEPT_QUESTIONS:
        return jsonify({"error": "Exam not found"}), 404
    
    if topic not in CONCEPT_QUESTIONS[exam_upper]:
        return jsonify({"error": "Topic not found"}), 404
    
    questions = CONCEPT_QUESTIONS[exam_upper][topic]
    
    formatted_questions = [
        {
            "id": f"Q{i+1}",
            "question": q["question"],
            "options": q["options"]
        }
        for i, q in enumerate(questions)
    ]
    
    return jsonify({
        "exam": exam_upper,
        "topic": topic,
        "total_questions": len(formatted_questions),
        "questions": formatted_questions,
        "pass_score": 80
    })

@app.route("/api/concept-check/submit/<exam>/<topic>", methods=["POST"])
def submit_concept_check(exam, topic):
    """Grade concept check"""
    try:
        data = request.get_json()
        user_answers = data.get("answers", {})
        
        exam_upper = exam.upper()
        
        if exam_upper not in CONCEPT_QUESTIONS or topic not in CONCEPT_QUESTIONS[exam_upper]:
            return jsonify({"error": "Invalid exam or topic"}), 404
        
        questions = CONCEPT_QUESTIONS[exam_upper][topic]
        
        score = 0
        total = len(questions)
        feedback_list = []
        
        for i, q in enumerate(questions):
            question_id = f"Q{i+1}"
            user_answer = user_answers.get(question_id, "").upper()
            correct_answer = q["correct"]
            
            is_correct = user_answer == correct_answer
            
            if is_correct:
                score += 1
                feedback_list.append({
                    "question": q["question"],
                    "status": "✅ CORRECT",
                    "your_answer": user_answer,
                    "correct_answer": correct_answer
                })
            else:
                feedback_list.append({
                    "question": q["question"],
                    "status": "❌ INCORRECT",
                    "your_answer": user_answer,
                    "correct_answer": correct_answer,
                    "explanation": f"Correct answer is {correct_answer}"
                })
        
        percentage = (score / total) * 100
        
        result = {
            "exam": exam_upper,
            "topic": topic,
            "score": score,
            "total": total,
            "percentage": round(percentage, 2),
            "date": datetime.now().isoformat(),
            "status": "✅ CONCEPT CLEAR!" if percentage >= 80 else "⚠️ REVIEW NEEDED",
            "feedback": feedback_list,
            "message": generate_concept_feedback(percentage)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_concept_feedback(percentage):
    """Generate concept check feedback"""
    if percentage >= 100:
        return "🌟 Perfect! You have mastered this concept!"
    elif percentage >= 80:
        return "✅ Concept is clear! You understand the fundamentals."
    elif percentage >= 60:
        return "⚠️ Partial understanding. Review the material before proceeding."
    else:
        return "❌ Concept is not clear. Study the definitions and examples again."

# ── FEATURE 3: PROGRESS TRACKER ────────────────────────────

@app.route("/api/tracker/<exam>")
def get_tracker(exam):
    """Get complete progress tracker for exam"""
    exam_upper = exam.upper()
    
    tracker = TRACKER_DATA["default"].get(exam_upper, {})
    
    if not tracker:
        return jsonify({"error": "Exam not found"}), 404
    
    total_items = 0
    completed_items = 0
    
    structure = {}
    
    for topic, subtopics in tracker.items():
        structure[topic] = {}
        for subtopic, status in subtopics.items():
            total_items += 4
            completed_items += sum(1 for v in status.values() if v)
            
            structure[topic][subtopic] = {
                "definition_learned": status.get("definition_learned", False),
                "concept_check_passed": status.get("concept_check_passed", False),
                "practice_done": status.get("practice_done", False),
                "pyq_attempted": status.get("pyq_attempted", False),
                "completion_percentage": (sum(1 for v in status.values() if v) / 4) * 100
            }
    
    overall_progress = (completed_items / total_items * 100) if total_items > 0 else 0
    
    return jsonify({
        "exam": exam_upper,
        "topics": structure,
        "overall_progress": round(overall_progress, 2),
        "completed_items": completed_items,
        "total_items": total_items
    })

@app.route("/api/tracker/update", methods=["POST"])
def update_tracker():
    """Update tracker status"""
    try:
        data = request.get_json()
        exam = data.get("exam", "").upper()
        topic = data.get("topic", "")
        subtopic = data.get("subtopic", "")
        item = data.get("item", "")
        status = data.get("status", False)
        
        if exam not in TRACKER_DATA["default"]:
            return jsonify({"error": "Exam not found"}), 404
        
        if topic not in TRACKER_DATA["default"][exam]:
            return jsonify({"error": "Topic not found"}), 404
        
        if subtopic not in TRACKER_DATA["default"][exam][topic]:
            return jsonify({"error": "Subtopic not found"}), 404
        
        TRACKER_DATA["default"][exam][topic][subtopic][item] = status
        
        tracker = TRACKER_DATA["default"][exam]
        total_items = 0
        completed_items = 0
        
        for t, subtopics in tracker.items():
            for st, st_status in subtopics.items():
                total_items += 4
                completed_items += sum(1 for v in st_status.values() if v)
        
        overall_progress = (completed_items / total_items * 100) if total_items > 0 else 0
        
        return jsonify({
            "status": "success",
            "message": f"{item.replace('_', ' ').title()} updated",
            "overall_progress": round(overall_progress, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── FEATURE 4: GRAPH VISUALIZATION ────────────────────────────

@app.route("/api/plot-2d", methods=["POST"])
def plot_2d():
    """Plot 2D function and analyze it"""
    try:
        data = request.get_json()
        equation = data.get("equation", "x**2")
        x_min = data.get("x_min", -10)
        x_max = data.get("x_max", 10)
        
        x = np.linspace(x_min, x_max, 1000)
        
        try:
            y = eval(equation.replace('^', '**'), {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log})
        except:
            return jsonify({"error": "Invalid equation"}), 400
        
        roots = []
        for i in range(len(y)-1):
            if (y[i] * y[i+1]) < 0:
                roots.append(float(x[i]))
        
        critical_points = []
        dy = np.gradient(y)
        for i in range(1, len(dy)-1):
            if (dy[i-1] * dy[i+1]) < 0:
                critical_points.append({
                    "x": float(x[i]),
                    "y": float(y[i]),
                    "type": "maximum" if dy[i-1] > 0 else "minimum"
                })
        
        plot_data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "equation": equation,
            "roots": [round(r, 4) for r in roots],
            "critical_points": critical_points,
            "analysis": {
                "domain": f"[{x_min}, {x_max}]",
                "number_of_roots": len(roots),
                "has_critical_points": len(critical_points) > 0,
                "y_intercept": float(eval(equation.replace('x', '0').replace('^', '**'), {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log})) if 'x' in equation else None
            }
        }
        
        return jsonify(plot_data)
    
    except Exception as e:
        print(f"Error in plot_2d: {e}")
        return jsonify({"error": str(e)}), 500

# ── EXISTING ROUTES (KEEP THESE) ────────────────────────────

@app.route("/api/definition/<concept>")
def get_definition(concept):
    """Get verified definition from authenticated sources"""
    concept_lower = concept.lower()
    
    for key, value in VERIFIED_DEFINITIONS.items():
        if key.lower() == concept_lower:
            return jsonify({
                "concept": key,
                "definition": value["definition"],
                "latex": value["latex"],
                "source": value["source"],
                "page": value["page"],
                "edition": value["edition"],
                "verified": value["verified"],
                "confidence": value["confidence"],
                "exams": value["exams"]
            })
    
    return jsonify({"error": "Definition not found in verified database"}), 404

@app.route("/api/materials/<exam>")
def get_materials(exam):
    """Get study materials for an exam"""
    exam_upper = exam.upper()
    materials = STUDY_MATERIALS.get(exam_upper, {})
    
    if not materials:
        return jsonify({"error": f"Materials not found for {exam}"}), 404
    
    structure = {}
    for topic, subtopics in materials.items():
        structure[topic] = list(subtopics.keys())
    
    return jsonify({"exam": exam_upper, "topics": structure})

@app.route("/api/materials/<exam>/<topic>/<subtopic>")
def get_material_detail(exam, topic, subtopic):
    """Get specific material"""
    exam_upper = exam.upper()
    material = STUDY_MATERIALS.get(exam_upper, {}).get(topic, {}).get(subtopic, {})
    
    if not material:
        return jsonify({"error": "Material not found"}), 404
    
    return jsonify(material)

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    try:
        d = request.get_json()
        prompt = f"""You are a mathematics quiz generator for graduation level students.
Generate ONE multiple choice question.
Topic: {d.get("topic","Calculus")}
Difficulty: {d.get("difficulty","medium")}
Question {d.get("q_num",1)} of {d.get("total",5)}

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
    return jsonify({"answer": ask_simple(f"Generate a complete formula sheet for: {topic}", system=SYSTEM_PROMPT)})

@app.route("/api/calculator", methods=["POST"])
def calculator():
    problem = request.get_json().get("problem","")
    sympy_r = solve_with_sympy(problem)
    answer  = ask_simple(f"Solve this step by step: {problem}")
    if sympy_r:
        answer = f"{sympy_r}\n\n◆━━━━━━━━━━━━━━━━━━◆\nStep-by-Step Solution:\n\n{answer}"
    return jsonify({"answer": answer})

@app.route("/api/latex", methods=["POST"])
def latex():
    text = request.get_json().get("text","")
    return jsonify({"answer": ask_simple(f"Generate LaTeX code for: {text}", system=SYSTEM_PROMPT)})

@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic","")
    return jsonify({"answer": ask_simple(f"TOP 10 rapid revision points for: {topic}", system=SYSTEM_PROMPT)})

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic","")
    return jsonify({"answer": ask_simple(f"Concept map for: {topic}", system=SYSTEM_PROMPT)})

@app.route("/api/compare", methods=["POST"])
def compare():
    concepts = request.get_json().get("concepts","")
    return jsonify({"answer": ask_simple(f"Compare: {concepts}", system=SYSTEM_PROMPT)})

@app.route("/api/verify", methods=["POST"])
def verify():
    claim = request.get_json().get("claim","")
    return jsonify({"answer": ask_simple(f"Verify or find counterexample: {claim}", system=SYSTEM_PROMPT)})

@app.route("/api/projects", methods=["POST"])
def projects():
    domain = request.get_json().get("domain","")
    return jsonify({"answer": ask_simple(f"3 real-life math projects for: {domain}", system=SYSTEM_PROMPT)})

@app.route("/api/proof", methods=["POST"])
def proof():
    d = request.get_json()
    msgs = d.get("history",[]) + [{"role":"user","content":d.get("theorem","")}]
    return jsonify({"answer": ask_ai(msgs, system=SYSTEM_PROMPT)})

@app.route("/api/debate", methods=["POST"])
def debate():
    d = request.get_json()
    msgs = d.get("history",[]) + [{"role":"user","content":d.get("argument","")}]
    return jsonify({"answer": ask_ai(msgs, system=SYSTEM_PROMPT)})

@app.route("/api/research", methods=["POST"])
def research():
    d = request.get_json()
    return jsonify({"answer": ask_simple(d.get("question",""), system=SYSTEM_PROMPT)})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🧮 MathSphere starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)