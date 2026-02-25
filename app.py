"""
MathSphere Web — Complete Backend v3.0
Fixes in this version:
✅ Mock test auto-generates 30 questions as a full list instantly
✅ New /api/pyq-solution/<exam>/<q_num> endpoint for per-question detailed solutions
✅ Aggressive post-processing strips ALL * and ** from every AI response
✅ Stronger system prompt with chain-of-thought for better accuracy
✅ Better error handling and fallbacks
✅ All other original features preserved

By Anupam Nigam | youtube.com/@pi_nomenal1729
"""

import os
import re
import json
import random
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

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

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

TEACHER_YOUTUBE   = "https://youtube.com/@pi_nomenal1729"
TEACHER_INSTAGRAM = "https://instagram.com/pi_nomenal1729"
TEACHER_WEBSITE   = "https://www.anupamnigam.com"


# ═══════════════════════════════════════════════════════════
# ── CRITICAL: STRIP ALL ASTERISKS FROM EVERY AI RESPONSE ──
# ═══════════════════════════════════════════════════════════

def clean_response(text: str) -> str:
    """
    Aggressively removes ALL markdown asterisk formatting from AI responses.
    This runs on EVERY response before it is returned to the frontend.
    """
    if not text:
        return text

    # Remove bold+italic: ***text*** or ___text___
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{3}(.+?)_{3}',   r'\1', text, flags=re.DOTALL)

    # Remove bold: **text** or __text__
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{2}(.+?)_{2}',   r'\1', text, flags=re.DOTALL)

    # Remove italic: *text* or _text_  (but NOT inside LaTeX \( ... \) or \[ ... \])
    # We protect LaTeX first, then strip, then restore
    latex_inline  = re.findall(r'\\\(.*?\\\)', text, flags=re.DOTALL)
    latex_display = re.findall(r'\\\[.*?\\\]',  text, flags=re.DOTALL)

    placeholder_map = {}
    for i, l in enumerate(latex_inline):
        ph = f"LATEXINLINE{i}PLACEHOLDER"
        placeholder_map[ph] = l
        text = text.replace(l, ph, 1)
    for i, l in enumerate(latex_display):
        ph = f"LATEXDISP{i}PLACEHOLDER"
        placeholder_map[ph] = l
        text = text.replace(l, ph, 1)

    # Now safe to remove stray single asterisks used as italic
    text = re.sub(r'(?<!\S)\*(.+?)\*(?!\S)', r'\1', text)

    # Remove any remaining lone asterisks that are NOT inside LaTeX placeholders
    # (be careful: multiplication sign in plain text like "3 * 4" — convert to ×)
    text = re.sub(r'(?<=\d)\s*\*\s*(?=\d)', ' × ', text)  # turn 3*4 → 3 × 4
    text = re.sub(r'\*', '', text)  # remove any remaining lone asterisks

    # Restore LaTeX
    for ph, orig in placeholder_map.items():
        text = text.replace(ph, orig)

    # Clean up multiple blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


# ═══════════════════════════════════════════════════════════
# ── AI CORE ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

def ask_ai(messages, system=None):
    if GROQ_AVAILABLE:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        if len(full) > 15:
            full = [full[0]] + full[-13:]
        for model in GROQ_MODELS:
            try:
                resp = groq_client.chat.completions.create(
                    model=model, messages=full, max_tokens=4000, temperature=0.3
                )
                return clean_response(resp.choices[0].message.content)
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
            return clean_response(resp.text)
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
        txt_part = gt.Part.from_text(text="""You are an expert mathematics teacher.
Solve this problem completely. Use LaTeX for all math expressions.
Inline math: \\( ... \\)   Display math: \\[ ... \\]
NEVER use * or ** for formatting. Use section headers like: 📌 Topic, 📐 Solution etc.
Start with: "Namaste! 🙏 Yeh problem dekh liya maine —"
End with: MathSphere: https://youtube.com/@pi_nomenal1729""")
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=[img_part, txt_part]
        )
        return clean_response(resp.text)
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
            return f"✅ SymPy Verified: \\( \\int = {sp.latex(result)} + C \\)"
        if "differentiate" in pl or "derivative" in pl:
            expr_str = pl.replace("differentiate","").replace("derivative","").replace("of","").strip()
            result = sp.diff(sp.sympify(expr_str), x)
            return f"✅ SymPy Verified: \\( \\frac{{d}}{{dx}} = {sp.latex(result)} \\)"
        if "solve" in pl and "=" in problem_text:
            eq_str = pl.replace("solve","").strip()
            lhs, rhs = eq_str.split("=",1)
            eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            result = sp.solve(eq, x)
            return f"✅ SymPy Verified: \\( x = {sp.latex(result)} \\)"
        if "simplify" in pl:
            expr_str = pl.replace("simplify","").strip()
            result = sp.simplify(sp.sympify(expr_str))
            return f"✅ SymPy Verified: \\( {sp.latex(result)} \\)"
        return None
    except:
        return None


# ═══════════════════════════════════════════════════════════
# ── SYSTEM PROMPT — STRICT NO-ASTERISKS + CHAIN OF THOUGHT
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are MathSphere — a warm, expert Mathematics teacher for graduation level students, created by Anupam Nigam (youtube.com/@pi_nomenal1729).

╔══════════════════════════════════════════╗
  ABSOLUTE FORMATTING RULE — NO EXCEPTIONS
╚══════════════════════════════════════════╝

NEVER use * or ** or *** for ANY reason whatsoever.
Not for bold. Not for italic. Not for emphasis. NEVER.
If you use *, the student sees raw asterisks — it breaks the display completely.

For emphasis: USE CAPS or emoji icons.
For bold text: Use section headers with emoji.
For math: ALWAYS use LaTeX — \\( inline \\) or \\[ display \\]

CORRECT: 📌 Important Concept
CORRECT: NOTE: This is critical
WRONG:   **Important Concept**  ← NEVER DO THIS
WRONG:   *Note*  ← NEVER DO THIS

═══════════════════════════════════════════
ACCURACY RULES — CHAIN OF THOUGHT:
═══════════════════════════════════════════
Before writing the final answer:
1. Identify the type of problem (calculus, algebra, analysis, etc.)
2. Recall the relevant theorem or technique
3. Work through each step carefully
4. Verify your answer by substitution or reverse calculation
5. Only then write the formatted solution

For exam questions (JAM/GATE/CSIR/NET):
- Eliminate wrong options by counterexample
- Show WHY each wrong option fails
- Prove the correct option rigorously

═══════════════════════════════════════════
LATEX RULES — MANDATORY:
═══════════════════════════════════════════
ALL math MUST be in LaTeX. Never write raw math like x^2.

Inline math: \\( x^2 + 3x + 2 = 0 \\)
Display math: \\[ \\int_0^1 x^2\\,dx = \\frac{{1}}{{3}} \\]
Fractions: \\frac{{a}}{{b}}
Square root: \\sqrt{{x}}
Matrices: \\begin{{pmatrix}} a & b \\\\ c & d \\end{{pmatrix}}
Sets: \\mathbb{{R}}, \\mathbb{{Z}}, \\mathbb{{Q}}, \\mathbb{{C}}
Limits: \\lim_{{x \\to 0}}
Integrals: \\int_{{a}}^{{b}}
Sums: \\sum_{{n=1}}^{{\\infty}}

═══════════════════════════════════════════
LANGUAGE & TONE:
═══════════════════════════════════════════
Mix Hinglish warmly: "Dekho...", "Samajh aaya?", "Yeh important hai!", "Bohot achha!"
Be like a friendly Indian teacher — warm, encouraging, clear.
For casual messages → brief friendly response.

═══════════════════════════════════════════
CONVERSATION MEMORY:
═══════════════════════════════════════════
Always use the FULL conversation history.
If student refers to "that problem" or "question 5" — refer back to it.
Build on previous explanations: "Jaise humne abhi dekha..."

═══════════════════════════════════════════
TEACHING STRUCTURE FOR MATH QUESTIONS:
═══════════════════════════════════════════

◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
📌 [Topic Name]
◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

💡 Real Life Analogy: [relatable Indian example]

📖 Definition: [precise definition with LaTeX]

📐 Step-by-Step Solution:
Step 1: [explanation with LaTeX]
Step 2: [explanation with LaTeX]
Step 3: [explanation with LaTeX]

✅ Verification: [verify the answer]

📝 Try This Yourself: [one practice problem]

📚 MathSphere: {TEACHER_YOUTUBE}
◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

Always end with the YouTube link: {TEACHER_YOUTUBE}"""


# ═══════════════════════════════════════════════════════════
# ── MOCK TEST GENERATOR PROMPT
# ═══════════════════════════════════════════════════════════

def get_mock_test_prompt(exam: str) -> str:
    exam_details = {
        "JAM": "IIT JAM Mathematics — topics: Real Analysis, Linear Algebra, Calculus, Abstract Algebra, ODE/PDE, Vector Calculus, Probability & Statistics. Level: BSc/MSc entrance.",
        "GATE": "GATE MA Mathematics — topics: Linear Algebra, Calculus, Real Analysis, Complex Analysis, ODE, PDE, Probability, Numerical Analysis, Combinatorics. Level: Engineering postgraduate.",
        "CSIR": "CSIR NET Mathematical Sciences — topics: Real Analysis, Complex Analysis, Functional Analysis, Abstract Algebra, Topology, ODE, PDE, Numerical Analysis. Level: MSc/PhD research.",
    }
    detail = exam_details.get(exam.upper(), f"{exam} Mathematics exam at postgraduate level")

    return f"""You are an expert mathematics professor creating a {exam.upper()} mock test paper.

Generate EXACTLY 30 multiple choice questions for: {detail}

STRICT FORMAT — follow exactly for each question:

Q1. [Question text with LaTeX using \\( inline \\) or \\[ display \\]]
(A) [option]
(B) [option]  
(C) [option]
(D) [option]
Answer: [A/B/C/D]
Topic: [topic name]

Q2. [Question text]
(A) [option]
...

Rules:
1. NEVER use * or ** anywhere — not even once
2. ALL math must use LaTeX: \\( ... \\) for inline, \\[ ... \\] for display
3. Questions must span ALL major topics of the exam
4. Each question must have exactly 4 options (A)(B)(C)(D)
5. The correct answer must be mathematically verified
6. Mix difficulty: 10 easy, 14 medium, 6 hard questions
7. Cover at least 8 different topics across 30 questions
8. No duplicate questions

Generate all 30 questions now."""


# ═══════════════════════════════════════════════════════════
# ── MOCK TEST PARSER
# ═══════════════════════════════════════════════════════════

def parse_mock_test(raw_text: str, exam: str):
    """Parse AI-generated mock test text into structured list of question dicts."""
    questions = []
    
    # Split by question numbers Q1, Q2, ... or 1., 2., etc.
    # Try both patterns
    blocks = re.split(r'\n(?=Q\d+\.|\d+\.\s)', raw_text.strip())
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Extract question number
        q_match = re.match(r'(?:Q)?(\d+)[\.\)]\s*(.+?)(?=\n\(A\)|\n\(a\)|\nA\)|\nA\.)', block, re.DOTALL)
        if not q_match:
            continue
        
        q_num = int(q_match.group(1))
        q_text = q_match.group(2).strip()
        
        # Extract options
        opts = {}
        for letter in ['A', 'B', 'C', 'D']:
            # Match (A) ... or A) ... or A. ...
            opt_match = re.search(
                rf'[\(\s]{letter}[\)\.\s]\s*(.+?)(?=\n[\(\s][BCDA][\)\.\s]|\nAnswer:|\nTopic:|\Z)',
                block, re.DOTALL | re.IGNORECASE
            )
            if opt_match:
                opts[letter] = opt_match.group(1).strip()
        
        # Extract answer
        ans_match = re.search(r'Answer:\s*([A-Da-d])', block, re.IGNORECASE)
        correct = ans_match.group(1).upper() if ans_match else 'A'
        
        # Extract topic
        topic_match = re.search(r'Topic:\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "Mathematics"
        
        if q_text and len(opts) >= 2:
            questions.append({
                "id": f"Q{q_num}",
                "num": q_num,
                "question": clean_response(q_text),
                "options": {k: clean_response(v) for k, v in opts.items()},
                "correct": correct,
                "topic": topic,
                "exam": exam.upper(),
            })
    
    return questions


# ═══════════════════════════════════════════════════════════
# ── MATHEMATICIANS DATABASE
# ═══════════════════════════════════════════════════════════

MATHEMATICIANS = [
    {
        "name": "Srinivasa Ramanujan",
        "period": "1887–1920",
        "country": "India",
        "fields": ["Number Theory", "Infinite Series", "Continued Fractions", "Mock Theta Functions"],
        "contribution": "One of the greatest mathematical geniuses in history. With almost no formal training, Ramanujan independently discovered thousands of results in number theory, infinite series, and continued fractions. His famous notebooks contain over 3,000 results, many still being proved today. He discovered the highly composite numbers, the Ramanujan prime, the Ramanujan-Soldner constant, and mock theta functions.",
        "key_results": "Ramanujan's tau function, Hardy-Ramanujan number 1729, Rogers-Ramanujan identities, Ramanujan conjecture",
        "application": "His work on partition functions is used in string theory and statistical mechanics. His mock theta functions appear in black hole physics.",
        "quote": "An equation for me has no meaning unless it expresses a thought of God.",
        "fun_fact": "1729 — the Hardy-Ramanujan number — is the smallest number expressible as sum of two cubes in two ways: \\( 1729 = 1^3 + 12^3 = 9^3 + 10^3 \\)"
    },
    {
        "name": "Leonhard Euler",
        "period": "1707–1783",
        "country": "Switzerland",
        "fields": ["Analysis", "Graph Theory", "Number Theory", "Topology"],
        "contribution": "The most prolific mathematician in history — wrote over 800 papers even after going completely blind. Euler created modern mathematical notation: f(x), e, π, i, Σ, and Δ. He proved Euler's identity \\( e^{i\\pi} + 1 = 0 \\), solved the Basel problem \\( \\sum 1/n^2 = π^2/6 \\), founded graph theory with the Königsberg bridge problem.",
        "key_results": "Euler's identity, Euler's formula, Euler characteristic, Basel problem, Euler's totient function",
        "application": "Euler's formula is foundational in electrical engineering, signal processing, and quantum mechanics. Graph theory powers the entire internet infrastructure.",
        "quote": "Mathematics is the queen of sciences.",
        "fun_fact": "Euler solved the Basel problem \\( \\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6} \\) in 1734 — a problem that had stumped mathematicians for 90 years!"
    },
    {
        "name": "Carl Friedrich Gauss",
        "period": "1777–1855",
        "country": "Germany",
        "fields": ["Number Theory", "Statistics", "Differential Geometry", "Algebra"],
        "contribution": "Called the Prince of Mathematics, Gauss proved the Fundamental Theorem of Algebra at age 21. He invented the method of least squares, the Gaussian distribution (bell curve), and modular arithmetic.",
        "key_results": "Fundamental Theorem of Algebra, Gaussian distribution, Gauss-Bonnet theorem, quadratic reciprocity",
        "application": "Gaussian distribution is the foundation of statistics and machine learning. His work on magnetic fields is used in MRI machines.",
        "quote": "Mathematics is the queen of the sciences and number theory is the queen of mathematics.",
        "fun_fact": "At age 10, Gauss instantly summed \\( 1 + 2 + \\cdots + 100 = 5050 \\) by noticing it equals \\( \\frac{100 \\times 101}{2} \\)!"
    },
    {
        "name": "Emmy Noether",
        "period": "1882–1935",
        "country": "Germany",
        "fields": ["Abstract Algebra", "Theoretical Physics", "Ring Theory"],
        "contribution": "Einstein called her the most significant creative mathematical genius yet produced. Noether revolutionized abstract algebra by introducing the concept of ideals, Noetherian rings, and chain conditions. Her Noether's theorem — connecting symmetry to conservation laws — is arguably the most important theorem in theoretical physics.",
        "key_results": "Noether's theorem, Noetherian rings, ascending chain condition, invariant theory",
        "application": "Noether's theorem underlies all of modern physics — conservation of energy, momentum, and charge all follow from it.",
        "quote": "My methods are really methods of working and thinking; this is why they have crept in everywhere anonymously.",
        "fun_fact": "Despite being acknowledged as brilliant by Einstein and Hilbert, Noether was denied a university position for years purely because she was a woman."
    },
    {
        "name": "Aryabhata",
        "period": "476–550 AD",
        "country": "India",
        "fields": ["Arithmetic", "Algebra", "Trigonometry", "Astronomy"],
        "contribution": "India's first major mathematician-astronomer. Aryabhata calculated π ≈ 3.1416 correct to 4 decimal places in 499 AD — over 1000 years before European mathematicians. He invented the place value system, introduced zero as a positional digit, developed sine and cosine tables.",
        "key_results": "Value of π, place value system, sine tables, rotation of Earth, eclipse calculations",
        "application": "His mathematical methods are foundational to modern astronomy and navigation.",
        "quote": "Just as a boat in water, the earth floats in space.",
        "fun_fact": "Aryabhata calculated the length of a year as 365.358 days — accurate to within minutes — using only naked-eye observations!"
    },
    {
        "name": "Terence Tao",
        "period": "1975–present",
        "country": "Australia",
        "fields": ["Harmonic Analysis", "Number Theory", "Partial Differential Equations"],
        "contribution": "Called the Mozart of Mathematics. Tao received the Fields Medal at age 31 and is widely considered the greatest living mathematician. He proved (with Ben Green) the Green-Tao theorem: prime numbers contain arbitrarily long arithmetic progressions.",
        "key_results": "Green-Tao theorem, Erdős discrepancy problem, compressed sensing, Navier-Stokes advances",
        "application": "His work on compressed sensing revolutionized medical imaging — MRI scans can now be done with far fewer measurements.",
        "quote": "What mathematics achieves: it gives us a language to describe patterns — and that is all we need.",
        "fun_fact": "Tao scored 760 on the math SAT at age 8, got his PhD at 20, and won the Fields Medal at 31!"
    },
    {
        "name": "Maryam Mirzakhani",
        "period": "1977–2017",
        "country": "Iran",
        "fields": ["Teichmüller Theory", "Hyperbolic Geometry", "Ergodic Theory"],
        "contribution": "First and only woman to win the Fields Medal (2014). Mirzakhani made groundbreaking contributions to the understanding of Riemann surfaces, their moduli spaces, and symplectic geometry.",
        "key_results": "Counting closed geodesics, Weil-Petersson volumes, moduli spaces of Riemann surfaces",
        "application": "Her work connects to quantum field theory, string theory, and the mathematical physics of 2D gravity.",
        "quote": "The beauty of mathematics only shows itself to more patient followers.",
        "fun_fact": "As a child in Tehran, Mirzakhani wanted to be a novelist. She became interested in mathematics only in high school!"
    },
    {
        "name": "Bhaskara II (Bhaskaracharya)",
        "period": "1114–1185",
        "country": "India",
        "fields": ["Algebra", "Calculus precursor", "Trigonometry", "Astronomy"],
        "contribution": "Greatest mathematician of medieval India. Bhaskara II discovered key concepts of differential calculus 500 years before Newton and Leibniz, including the idea of instantaneous velocity.",
        "key_results": "Differential calculus precursor, Pell equation, instantaneous velocity, Lilavati",
        "application": "His planetary motion calculations were remarkably accurate. His understanding of instantaneous motion foreshadowed calculus.",
        "quote": "A particle of turmeric or a grain of rice, cut in half, again in half, becomes a paramāṇu.",
        "fun_fact": "Bhaskara's daughter Lilavati is the subject of his famous mathematics book — a masterpiece of mathematical exposition!"
    },
    {
        "name": "Kurt Gödel",
        "period": "1906–1978",
        "country": "Austria-Hungary (later USA)",
        "fields": ["Mathematical Logic", "Set Theory", "Philosophy of Mathematics"],
        "contribution": "Proved the two Incompleteness Theorems in 1931. Any consistent formal system strong enough to include arithmetic contains true statements that cannot be proved within the system.",
        "key_results": "Gödel's Incompleteness Theorems, completeness theorem, constructible universe",
        "application": "Incompleteness theorems imply the undecidability of the halting problem and limitations of artificial intelligence.",
        "quote": "Either mathematics is too big for the human mind or the human mind is more than a machine.",
        "fun_fact": "Gödel found a logical loophole in the US Constitution that could allow a dictator to take over — he tried to explain this at his citizenship hearing!"
    },
    {
        "name": "Alan Turing",
        "period": "1912–1954",
        "country": "England",
        "fields": ["Computability Theory", "Cryptography", "Mathematical Biology", "AI"],
        "contribution": "Father of theoretical computer science. Turing created the mathematical model of computation (Turing machine), proved the halting problem is undecidable, and broke the Nazi Enigma code in World War II.",
        "key_results": "Turing machine, halting problem, Turing test, Enigma decryption",
        "application": "Every computer is a physical realization of a Turing machine.",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        "fun_fact": "Turing's team cracked over 84,000 Enigma messages per month — estimated to have saved 14 million lives!"
    },
]


# ═══════════════════════════════════════════════════════════
# ── REAL WORLD APPLICATIONS
# ═══════════════════════════════════════════════════════════

REAL_WORLD_APPS = [
    {"concept":"Fourier Transform","application":"MRI Machines",
     "explanation":"MRI machines use Fourier Transform to convert radio frequency signals from hydrogen atoms into detailed 3D images of your organs. Without this mathematics, modern medical imaging would not exist!"},
    {"concept":"Linear Algebra","application":"Google PageRank Algorithm",
     "explanation":"Google's PageRank uses eigenvectors of a massive matrix representing the web graph. A single Google search involves computing eigenvectors of a matrix with billions of rows and columns!"},
    {"concept":"Probability Theory","application":"Weather Forecasting",
     "explanation":"Weather predictions use Bayesian probability, Markov chains, and stochastic differential equations. Ensemble forecasting runs 50+ simulations with slight variations to get that '70% chance of rain'."},
    {"concept":"Differential Equations","application":"COVID-19 Pandemic Modelling",
     "explanation":"Governments used SIR/SEIR differential equations to model the pandemic. These equations determined lockdown policies, vaccine rollout strategies, and hospital capacity planning for the entire world."},
    {"concept":"Number Theory","application":"RSA Encryption and Internet Security",
     "explanation":"Every HTTPS website uses RSA encryption based on the difficulty of factoring large numbers. Your WhatsApp messages and banking transactions are protected by number theory that Fermat developed purely out of curiosity!"},
    {"concept":"Graph Theory","application":"Google Maps and Navigation",
     "explanation":"Dijkstra's shortest path algorithm finds your fastest route through a graph of roads. Google Maps maintains a graph of billions of nodes and edges with real-time weights. Founded by Euler in 1736!"},
    {"concept":"Calculus","application":"ISRO Rocket Trajectory Planning",
     "explanation":"ISRO engineers solve systems of nonlinear differential equations to plan Chandrayaan-3's trajectory. Every course correction burn is computed by solving differential equations in real time!"},
    {"concept":"Information Theory","application":"WhatsApp and Data Compression",
     "explanation":"Shannon's entropy formula determines the minimum bits needed to represent information. Every MP3, JPEG, and ZIP file uses Shannon's information theory. Without it, a 10-minute HD video would be 15GB!"},
]


# ═══════════════════════════════════════════════════════════
# ── PARADOXES
# ═══════════════════════════════════════════════════════════

PARADOXES = [
    {"name":"Zeno's Paradox","statement":"Achilles can NEVER overtake a tortoise with a head start. Every time Achilles reaches where the tortoise was, the tortoise has moved ahead. This creates an infinite sequence of steps — implying motion is impossible. Yet we observe motion!","teaser":"Motion itself is mathematically impossible according to this paradox. The resolution involves convergent infinite series."},
    {"name":"0.999... = 1","statement":"0.999... repeating forever is EXACTLY equal to 1. Not approximately — exactly! Proof: Let x = 0.999..., then 10x = 9.999..., so 10x - x = 9, giving 9x = 9, so x = 1.","teaser":"This looks wrong. It feels wrong. But three different proofs confirm it is absolutely right!"},
    {"name":"Russell's Paradox","statement":"Consider the set R = {all sets that do NOT contain themselves}. Does R contain itself? If R is in R, then R should NOT be in R. If R is not in R, then R SHOULD be in R. This contradiction destroyed naive set theory!","teaser":"One question broke all of mathematics and forced mathematicians to rebuild foundations from scratch!"},
    {"name":"Hilbert's Infinite Hotel","statement":"A hotel with infinitely many rooms is completely full. Accommodate a new guest by moving guest in room n to room n+1. Accommodate infinitely many new guests by moving room n to room 2n, freeing all odd-numbered rooms!","teaser":"Some infinities can fit inside themselves. Not all infinite quantities behave the same way!"},
    {"name":"Banach-Tarski Paradox","statement":"Mathematically, you can decompose a unit sphere into 5 pieces, then reassemble those pieces into TWO unit spheres of the same size. Volume doubles from nothing!","teaser":"Pure mathematics says you can duplicate a ball — but the proof requires the Axiom of Choice!"},
    {"name":"Cantor's Different Infinities","statement":"The infinity of real numbers is STRICTLY LARGER than the infinity of natural numbers. Cantor proved this with his diagonal argument: any list of real numbers must be incomplete.","teaser":"Georg Cantor proved this and was called insane. He was right. Some infinities are bigger than others."},
    {"name":"Birthday Paradox","statement":"In a group of just 23 people, there is a 50% chance two people share a birthday. With 70 people: 99.9% probability! The calculation involves counting PAIRS of people, not people vs days.","teaser":"How can 23 people out of 365 possible birthdays give a 50% collision probability?"},
    {"name":"Monty Hall Problem","statement":"You pick door 1 of 3. Host opens door 3 (a goat). Should you switch? YES — switching wins 2/3 of the time! The probability P(car behind door 1) = 1/3 does NOT change after the host reveals a goat. Door 2 inherits the remaining 2/3 probability.","teaser":"Even PhD mathematicians got this wrong when first published. Thousands wrote in to say the correct answer was wrong!"},
]


# ═══════════════════════════════════════════════════════════
# ── DAILY CHALLENGES
# ═══════════════════════════════════════════════════════════

DAILY_CHALLENGES = [
    "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
    "If \\(f(x) = x^3 - 3x + 2\\), find all critical points and classify them.",
    "Find eigenvalues and eigenvectors of \\(\\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}\\).",
    "Evaluate \\(\\int x^2 e^x \\, dx\\) using integration by parts.",
    "Find the radius of convergence of \\(\\sum_{n=0}^{\\infty} \\frac{x^n}{n!}\\).",
    "Solve: \\(\\frac{dy}{dx} + 2y = 4x\\) with \\(y(0) = 1\\).",
    "Prove AM \\(\\geq\\) GM for positive reals \\(a, b\\).",
    "Find the Fourier series of \\(f(x) = x\\) on \\([-\\pi, \\pi]\\).",
    "Show that every finite integral domain is a field.",
    "Find all solutions of \\(z^4 = 1\\) in \\(\\mathbb{C}\\).",
    "Prove the continuous image of a compact set is compact.",
    "Evaluate \\(\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n\\).",
    "Show the p-series converges iff \\(p > 1\\).",
    "Prove every subgroup of a cyclic group is cyclic.",
    "Find all ideals of \\(\\mathbb{Z}/12\\mathbb{Z}\\).",
]


# ═══════════════════════════════════════════════════════════
# ── PYQ BANK (kept from original — used as fallback)
# ═══════════════════════════════════════════════════════════

PYQ_BANK = {
    "JAM": [
        {"q":"Let \\(f(x) = x^2 \\sin(1/x)\\) for \\(x \\neq 0\\) and \\(f(0) = 0\\). Is \\(f\\) differentiable at \\(x = 0\\)?",
         "opts":{"A":"Yes, \\(f'(0) = 0\\)","B":"No, the limit does not exist","C":"Yes, \\(f'(0) = 1\\)","D":"Yes, \\(f'(0) = \\infty\\)"},
         "correct":"A",
         "a":"ANSWER: A — \\(f'(0) = \\lim_{h \\to 0} \\frac{h^2 \\sin(1/h)}{h} = \\lim_{h \\to 0} h\\sin(1/h) = 0\\) since \\(|h\\sin(1/h)| \\leq |h| \\to 0\\) by the Squeeze Theorem.",
         "topic":"Real Analysis","year":"2023"},
        {"q":"The number of group homomorphisms from \\(\\mathbb{Z}_{12}\\) to \\(\\mathbb{Z}_8\\) is:",
         "opts":{"A":"2","B":"4","C":"6","D":"8"},
         "correct":"B",
         "a":"ANSWER: B — Homomorphisms \\(\\mathbb{Z}_m \\to \\mathbb{Z}_n\\) correspond to divisors of \\(\\gcd(m,n) = \\gcd(12,8) = 4\\). Number of divisors of 4 is 4.",
         "topic":"Algebra","year":"2023"},
        {"q":"The value of \\(\\int_0^{\\infty} e^{-x^2} \\, dx\\) is:",
         "opts":{"A":"\\(\\frac{\\sqrt{\\pi}}{2}\\)","B":"\\(\\sqrt{\\pi}\\)","C":"\\(\\frac{\\pi}{2}\\)","D":"\\(1\\)"},
         "correct":"A",
         "a":"ANSWER: A — Gaussian integral: \\( I = \\int_0^\\infty e^{-x^2}dx \\). Then \\(I^2 = \\int_0^\\infty\\int_0^\\infty e^{-(x^2+y^2)}dxdy = \\int_0^{\\pi/2}\\int_0^\\infty e^{-r^2}r\\,dr\\,d\\theta = \\frac{\\pi}{4} \\). So \\( I = \\frac{\\sqrt{\\pi}}{2} \\).",
         "topic":"Calculus","year":"2022"},
        {"q":"Eigenvalues of \\(A = \\begin{pmatrix} 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ 1 & -3 & 3 \\end{pmatrix}\\) are:",
         "opts":{"A":"\\(0, 1, 2\\)","B":"\\(1, 1, 1\\)","C":"\\(-1, 1, 3\\)","D":"\\(0, 0, 3\\)"},
         "correct":"B",
         "a":"ANSWER: B — Characteristic polynomial: \\(\\det(A - \\lambda I) = -(\\lambda-1)^3 = 0\\). So \\(\\lambda = 1\\) with multiplicity 3.",
         "topic":"Linear Algebra","year":"2022"},
        {"q":"The series \\(\\sum_{n=1}^{\\infty} \\frac{n^2+1}{n^3+n+1}\\) is:",
         "opts":{"A":"Convergent","B":"Divergent","C":"Conditionally convergent","D":"Cannot be determined"},
         "correct":"B",
         "a":"ANSWER: B — Compare with \\(1/n\\): \\(\\lim_{n\\to\\infty} \\frac{(n^2+1)/(n^3+n+1)}{1/n} = 1 \\neq 0\\). Since \\(\\sum 1/n\\) diverges, the original series diverges by Limit Comparison Test.",
         "topic":"Real Analysis","year":"2021"},
        {"q":"The ODE \\(y'' + 4y = 0\\) with \\(y(0)=1, y'(0)=2\\) has solution:",
         "opts":{"A":"\\(\\cos 2x + \\sin 2x\\)","B":"\\(\\cos 2x + 2\\sin 2x\\)","C":"\\(e^{2x}\\)","D":"\\(\\cosh 2x\\)"},
         "correct":"A",
         "a":"ANSWER: A — Characteristic equation: \\(r^2+4=0\\), so \\(r = \\pm 2i\\). General solution: \\(y = A\\cos 2x + B\\sin 2x\\). Applying \\(y(0)=1\\Rightarrow A=1\\) and \\(y'(0)=2\\Rightarrow 2B=2\\Rightarrow B=1\\). So \\(y = \\cos 2x + \\sin 2x\\).",
         "topic":"ODE","year":"2019"},
        {"q":"The dimension of the null space of \\(A = \\begin{pmatrix} 1 & 2 & 3 \\\\ 2 & 4 & 6 \\\\ 1 & 2 & 3 \\end{pmatrix}\\) is:",
         "opts":{"A":"0","B":"1","C":"2","D":"3"},
         "correct":"C",
         "a":"ANSWER: C — Row reduce: all rows reduce to \\([1, 2, 3]\\), so rank = 1. By rank-nullity theorem: nullity = 3 − 1 = 2.",
         "topic":"Linear Algebra","year":"2020"},
        {"q":"The number of elements of order 4 in \\(\\mathbb{Z}_2 \\times \\mathbb{Z}_4\\) is:",
         "opts":{"A":"2","B":"4","C":"6","D":"8"},
         "correct":"B",
         "a":"ANSWER: B — Element \\((a,b)\\) has order 4 iff \\(\\text{lcm}(\\text{ord}(a), \\text{ord}(b)) = 4\\). The elements \\((0,1),(0,3),(1,1),(1,3)\\) each have order 4. Count: 4.",
         "topic":"Algebra","year":"2020"},
        {"q":"The partial differential equation \\(u_{xx} - u_{tt} = 0\\) is:",
         "opts":{"A":"Parabolic","B":"Elliptic","C":"Hyperbolic","D":"Neither"},
         "correct":"C",
         "a":"ANSWER: C — With \\(A=1, B=0, C=-1\\): discriminant \\(B^2 - 4AC = 4 > 0\\), so this is HYPERBOLIC (the wave equation).",
         "topic":"PDE","year":"2023"},
        {"q":"The maximum value of \\(f(x,y) = x+y\\) subject to \\(x^2+y^2=1\\) is:",
         "opts":{"A":"1","B":"\\(\\sqrt{2}\\)","C":"2","D":"\\(\\frac{1}{\\sqrt{2}}\\)"},
         "correct":"B",
         "a":"ANSWER: B — By Cauchy-Schwarz: \\(x+y \\leq \\sqrt{2}\\sqrt{x^2+y^2} = \\sqrt{2}\\). Equality when \\(x=y=1/\\sqrt{2}\\). Maximum is \\(\\sqrt{2}\\).",
         "topic":"Calculus","year":"2021"},
    ],
    "GATE": [
        {"q":"The rank of \\(A = \\begin{pmatrix} 1 & 2 & 1 \\\\ 0 & 1 & 1 \\\\ 1 & 3 & 2 \\end{pmatrix}\\) is:",
         "opts":{"A":"1","B":"2","C":"3","D":"0"},
         "correct":"B",
         "a":"ANSWER: B — Row reduce: \\(R_3 \\to R_3 - R_1\\) gives \\([0,1,1]\\) same as \\(R_2\\). Rows 2 and 3 become identical — rank = 2.",
         "topic":"Linear Algebra","year":"2023"},
        {"q":"The PDE \\(u_{xx} + 4u_{xy} + 4u_{yy} = 0\\) is classified as:",
         "opts":{"A":"Elliptic","B":"Hyperbolic","C":"Parabolic","D":"None"},
         "correct":"C",
         "a":"ANSWER: C — With \\(A=1, B=4, C=4\\): discriminant \\(B^2 - 4AC = 16 - 16 = 0\\), so PARABOLIC.",
         "topic":"PDE","year":"2023"},
        {"q":"Number of onto functions from \\(\\{1,2,3,4\\}\\) to \\(\\{a,b,c\\}\\) is:",
         "opts":{"A":"18","B":"24","C":"36","D":"81"},
         "correct":"C",
         "a":"ANSWER: C — By inclusion-exclusion: \\(3^4 - \\binom{3}{1}2^4 + \\binom{3}{2}1^4 = 81 - 48 + 3 = 36\\).",
         "topic":"Combinatorics","year":"2022"},
        {"q":"\\(\\oint_{|z|=2} \\frac{dz}{z^2+1}\\) (counterclockwise) equals:",
         "opts":{"A":"\\(2\\pi i\\)","B":"\\(-2\\pi i\\)","C":"\\(0\\)","D":"\\(\\pi i\\)"},
         "correct":"C",
         "a":"ANSWER: C — Singularities at \\(z = \\pm i\\), both inside \\(|z|=2\\). Residue at \\(i\\) is \\(\\frac{1}{2i}\\), residue at \\(-i\\) is \\(\\frac{-1}{2i}\\). Sum = 0, so integral = 0.",
         "topic":"Complex Analysis","year":"2022"},
        {"q":"Laplace transform of \\(t\\sin(at)\\) is:",
         "opts":{"A":"\\(\\frac{a}{(s^2+a^2)^2}\\)","B":"\\(\\frac{2as}{(s^2+a^2)^2}\\)","C":"\\(\\frac{s^2-a^2}{(s^2+a^2)^2}\\)","D":"\\(\\frac{a}{s^2+a^2}\\)"},
         "correct":"B",
         "a":"ANSWER: B — Using \\(\\mathcal{L}\\{tf(t)\\} = -F'(s)\\): \\(-\\frac{d}{ds}\\frac{a}{s^2+a^2} = \\frac{2as}{(s^2+a^2)^2}\\).",
         "topic":"ODE","year":"2021"},
        {"q":"The value of \\(\\lim_{x \\to 0} \\frac{\\sin x - x}{x^3}\\) is:",
         "opts":{"A":"\\(-\\frac{1}{6}\\)","B":"\\(\\frac{1}{6}\\)","C":"0","D":"1"},
         "correct":"A",
         "a":"ANSWER: A — By Taylor series: \\(\\sin x = x - \\frac{x^3}{6} + O(x^5)\\). So \\(\\frac{\\sin x - x}{x^3} \\to -\\frac{1}{6}\\).",
         "topic":"Calculus","year":"2022"},
        {"q":"Sum \\(\\sum_{n=0}^{\\infty} \\frac{(-1)^n}{2n+1}\\) equals:",
         "opts":{"A":"\\(\\ln 2\\)","B":"\\(\\frac{\\pi}{4}\\)","C":"\\(\\frac{\\pi}{2}\\)","D":"1"},
         "correct":"B",
         "a":"ANSWER: B — Leibniz formula: \\(\\arctan(1) = \\frac{\\pi}{4} = \\sum_{n=0}^\\infty \\frac{(-1)^n}{2n+1}\\).",
         "topic":"Calculus","year":"2020"},
        {"q":"If \\(f(z) = u + iv\\) is analytic and \\(u = x^2 - y^2\\), then \\(v\\) is:",
         "opts":{"A":"\\(2xy + c\\)","B":"\\(xy + c\\)","C":"\\(x^2 - y^2 + c\\)","D":"\\(x^2 + y^2 + c\\)"},
         "correct":"A",
         "a":"ANSWER: A — Cauchy-Riemann: \\(\\partial v/\\partial y = \\partial u/\\partial x = 2x\\) and \\(\\partial v/\\partial x = -\\partial u/\\partial y = 2y\\). Integrating: \\(v = 2xy + c\\).",
         "topic":"Complex Analysis","year":"2021"},
    ],
    "CSIR": [
        {"q":"The number of non-isomorphic groups of order 8 is:",
         "opts":{"A":"3","B":"4","C":"5","D":"6"},
         "correct":"C",
         "a":"ANSWER: C — The 5 non-isomorphic groups of order 8 are: \\(\\mathbb{Z}_8\\), \\(\\mathbb{Z}_4 \\times \\mathbb{Z}_2\\), \\(\\mathbb{Z}_2^3\\), \\(D_4\\) (dihedral), \\(Q_8\\) (quaternion).",
         "topic":"Algebra","year":"2023"},
        {"q":"The closure of \\(\\mathbb{Q}\\) in \\(\\mathbb{R}\\) with the standard topology is:",
         "opts":{"A":"\\(\\mathbb{Q}\\)","B":"\\((0,1)\\)","C":"\\(\\mathbb{R}\\)","D":"\\(\\mathbb{Z}\\)"},
         "correct":"C",
         "a":"ANSWER: C — \\(\\mathbb{Q}\\) is dense in \\(\\mathbb{R}\\): every real number is the limit of a sequence of rationals. Therefore \\(\\overline{\\mathbb{Q}} = \\mathbb{R}\\).",
         "topic":"Topology","year":"2021"},
        {"q":"The fundamental group of the torus \\(T^2 = S^1 \\times S^1\\) is:",
         "opts":{"A":"\\(\\mathbb{Z}\\)","B":"\\(\\mathbb{Z} \\times \\mathbb{Z}\\)","C":"Trivial","D":"\\(\\mathbb{Z}_2\\)"},
         "correct":"B",
         "a":"ANSWER: B — \\(\\pi_1(T^2) = \\pi_1(S^1) \\times \\pi_1(S^1) = \\mathbb{Z} \\times \\mathbb{Z}\\). The two generators correspond to loops around the two holes of the torus.",
         "topic":"Topology","year":"2022"},
        {"q":"A normed space is Banach iff every absolutely convergent series is convergent. This is:",
         "opts":{"A":"False","B":"True","C":"True only for Hilbert spaces","D":"True only finite-dimensional"},
         "correct":"B",
         "a":"ANSWER: B — Standard characterization: a normed space \\(X\\) is complete iff every series with \\(\\sum ||x_n|| < \\infty\\) converges in \\(X\\).",
         "topic":"Functional Analysis","year":"2022"},
        {"q":"Which function is uniformly continuous on \\((0, 1)\\)?",
         "opts":{"A":"\\(f(x) = 1/x\\)","B":"\\(f(x) = \\sin(1/x)\\)","C":"\\(f(x) = \\sqrt{x}\\)","D":"\\(f(x) = x^2 \\sin(1/x^2)\\)"},
         "correct":"C",
         "a":"ANSWER: C — \\(f(x) = \\sqrt{x}\\) extends continuously to \\([0,1]\\) (compact), so it is uniformly continuous on \\((0,1)\\). The others fail uniform continuity near 0.",
         "topic":"Real Analysis","year":"2021"},
    ]
}


# ═══════════════════════════════════════════════════════════
# ── EXAM INFO
# ═══════════════════════════════════════════════════════════

EXAM_INFO = {
    "JAM": {
        "full_name": "Joint Admission Test for Masters (JAM) — Mathematics",
        "conducting_body": "IITs on rotation (IIT Delhi, Bombay, Madras, etc.)",
        "eligibility": "Bachelor's degree with Mathematics as a subject with minimum 55% marks (50% for SC/ST/PwD)",
        "pattern": "3 hours total, 60 questions, 100 marks. Section A: 30 MCQ (negative marking). Section B: 10 MSQ (no negative). Section C: 20 NAT (no negative).",
        "syllabus": "Real Analysis, Linear Algebra, Calculus, Differential Equations, Vector Calculus, Probability & Statistics, Group Theory",
        "weightage": "Real Analysis: 25-30%, Linear Algebra: 20-25%, Calculus: 15-20%, ODE: 10-15%, Algebra: 10-12%, Stats: 8-10%",
        "books": "Rudin (Real Analysis), Gilbert Strang (Linear Algebra), Apostol (Calculus), Herstein (Algebra), Arora & Sharma (JAM guide)",
        "website": "https://jam.iitd.ac.in",
        "career": "MSc at IITs/IISc, leading to PhD in mathematics, data science, finance, and academia"
    },
    "GATE": {
        "full_name": "Graduate Aptitude Test in Engineering — Mathematics (GATE MA)",
        "conducting_body": "IITs and IISc on rotation",
        "eligibility": "Bachelor's degree in Mathematics or related field",
        "pattern": "3 hours, 65 questions, 100 marks. General Aptitude: 15 marks. Mathematics: 85 marks. Mix of MCQ and NAT.",
        "syllabus": "Calculus, Linear Algebra, Real Analysis, Complex Analysis, Abstract Algebra, ODE, PDE, Probability & Statistics, Numerical Analysis, Combinatorics",
        "weightage": "Calculus + Linear Algebra: 35-40%, Real + Complex Analysis: 20-25%, ODE + PDE: 15-18%, Algebra: 10-12%",
        "books": "Kreyszig (Advanced Engineering Mathematics), Rudin, Herstein, Churchill (Complex Variables), S.L. Ross (ODE)",
        "website": "https://gate2024.iisc.ac.in",
        "career": "PSU recruitment (BARC, DRDO, ISRO), NITs/IITs research, Central Government jobs, PhD admissions"
    },
    "CSIR": {
        "full_name": "CSIR UGC NET Mathematical Sciences",
        "conducting_body": "National Testing Agency (NTA) on behalf of CSIR",
        "eligibility": "MSc Mathematics with minimum 55% marks (50% for SC/ST/OBC). Final year MSc students can also apply.",
        "pattern": "3 hours, 3 parts. Part A: General Aptitude (30 marks). Part B: Core Math (75 marks). Part C: Advanced Math (60 marks).",
        "syllabus": "Real Analysis, Complex Analysis, Functional Analysis, Abstract Algebra, Topology, ODE, PDE, Numerical Analysis, Statistics",
        "weightage": "Analysis: 30-35%, Algebra + Linear Algebra: 25-30%, Topology: 12-15%, ODE/PDE: 10-12%",
        "books": "Rudin, Royden (Lebesgue measure), Ahlfors (Complex Analysis), Dummit & Foote (Algebra), Munkres (Topology)",
        "website": "https://csirnet.nta.nic.in",
        "career": "JRF with Rs 31,000/month stipend, Lectureship eligibility, PhD stipend at top research institutes, NBHM scholarship"
    }
}


# ═══════════════════════════════════════════════════════════
# ── IN-MEMORY MOCK TEST STORE (session-based)
# ═══════════════════════════════════════════════════════════
# Stores generated mock tests by test_id so solutions can be retrieved later
_mock_test_store = {}


# ═══════════════════════════════════════════════════════════
# ── ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "groq": GROQ_AVAILABLE, "gemini": GEMINI_AVAILABLE})


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages   = data.get("messages", [])
        image_b64  = data.get("image_b64")
        image_type = data.get("image_type", "image/jpeg")

        if not messages:
            return jsonify({"error": "messages required"}), 400

        if image_b64 and GEMINI_AVAILABLE:
            result = solve_image_with_gemini(image_b64, image_type)
            if result:
                return jsonify({"answer": result, "source": "gemini-vision"})

        clean = [{"role": m["role"], "content": str(m["content"])}
                 for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]

        if len(clean) > 16:
            clean = clean[-14:]

        enhanced_system = SYSTEM_PROMPT + """

ADDITIONAL ABSOLUTE RULE:
NEVER output * or ** for any reason. This is a hard constraint.
If you feel the urge to write **word**, write it as: NOTE: word or use CAPS.
Remember the FULL conversation above and refer back naturally.
Work through problems step by step before writing the answer."""

        return jsonify({"answer": ask_ai(clean, system=enhanced_system)})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


# ── MOCK TEST — AUTO-GENERATE 30 QUESTIONS ────────────────────

@app.route("/api/pyq-mock/<exam>")
def pyq_mock_test(exam):
    """
    Generates a full 30-question mock test via AI.
    Returns the complete list of questions immediately.
    Also stores the test in memory for solution retrieval.
    """
    exam_upper = exam.upper()
    if exam_upper not in ["JAM", "GATE", "CSIR", "NET"]:
        # Fall back to PYQ bank for unknown exams
        if exam_upper not in PYQ_BANK:
            return jsonify({"error": f"Exam '{exam_upper}' not found"}), 404

    test_id = f"{exam_upper}-MOCK-{int(datetime.now().timestamp())}"

    # Try to generate 30 questions via AI
    prompt = get_mock_test_prompt(exam_upper)
    
    try:
        raw_ai = ask_simple(prompt)
        questions = parse_mock_test(raw_ai, exam_upper)
    except Exception as e:
        print(f"AI mock test generation error: {e}")
        questions = []

    # If AI generation fails or returns too few, fill from PYQ bank
    if len(questions) < 10:
        bank_qs = PYQ_BANK.get(exam_upper, [])
        sampled = random.sample(bank_qs, min(len(bank_qs), 30))
        questions = [
            {
                "id": f"Q{i+1}",
                "num": i+1,
                "question": q.get("q", ""),
                "options": q.get("opts", {"A":"Option A","B":"Option B","C":"Option C","D":"Option D"}),
                "correct": q.get("correct", "A"),
                "topic": q.get("topic", "Mathematics"),
                "exam": exam_upper,
                "_solution": q.get("a", ""),
            }
            for i, q in enumerate(sampled)
        ]

    # Ensure questions are numbered correctly
    for i, q in enumerate(questions):
        q["id"] = f"Q{i+1}"
        q["num"] = i+1

    # Store with solutions for later retrieval
    _mock_test_store[test_id] = {
        "exam": exam_upper,
        "questions": questions,
        "created_at": datetime.now().isoformat(),
    }

    # Return questions WITHOUT answers (for the test UI)
    public_questions = [
        {
            "id": q["id"],
            "num": q["num"],
            "question": q["question"],
            "options": q["options"],
            "topic": q.get("topic", "Mathematics"),
            "exam": exam_upper,
        }
        for q in questions
    ]

    return jsonify({
        "test_id": test_id,
        "exam": exam_upper,
        "total_questions": len(public_questions),
        "duration_minutes": 60,
        "questions": public_questions,
        "message": f"{exam_upper} Mock Test ready — {len(public_questions)} questions. All the best! 🎯"
    })


# ── GET SOLUTION FOR A SPECIFIC QUESTION ─────────────────────

@app.route("/api/pyq-solution/<exam>/<q_ref>", methods=["GET", "POST"])
def get_question_solution(exam, q_ref):
    """
    Get a detailed step-by-step solution for a specific question.
    q_ref can be:
      - A question number like "5" or "Q5"
      - A test_id can be passed in query params: ?test_id=JAM-MOCK-1234
      - Or the question text can be posted in the body
    """
    exam_upper = exam.upper()
    test_id = request.args.get("test_id", "")

    # Normalize question number
    q_num_str = q_ref.upper().replace("Q", "").strip()
    try:
        q_num = int(q_num_str)
    except ValueError:
        q_num = None

    question_text = ""
    options_text = ""

    # Try to find question from stored test
    if test_id and test_id in _mock_test_store:
        stored = _mock_test_store[test_id]
        if q_num:
            for q in stored["questions"]:
                if q["num"] == q_num:
                    question_text = q["question"]
                    opts = q.get("options", {})
                    options_text = "\n".join([f"({k}) {v}" for k, v in opts.items()])
                    correct = q.get("correct", "")
                    # If we have a stored solution (from PYQ bank), use it
                    stored_sol = q.get("_solution", "")
                    if stored_sol:
                        return jsonify({
                            "question_num": q_num,
                            "exam": exam_upper,
                            "question": question_text,
                            "solution": clean_response(stored_sol),
                            "correct_answer": correct,
                        })
                    break

    # If question text was posted in body
    if request.method == "POST":
        body = request.get_json() or {}
        question_text = question_text or body.get("question", "")
        options_text  = options_text  or body.get("options", "")

    # If we still don't have question text, try PYQ bank
    if not question_text and q_num:
        bank = PYQ_BANK.get(exam_upper, [])
        if q_num <= len(bank):
            q = bank[q_num - 1]
            question_text = q.get("q", "")
            opts = q.get("opts", {})
            options_text = "\n".join([f"({k}) {v}" for k, v in opts.items()])
            stored_sol = q.get("a", "")
            if stored_sol:
                return jsonify({
                    "question_num": q_num,
                    "exam": exam_upper,
                    "question": question_text,
                    "solution": clean_response(stored_sol),
                    "correct_answer": q.get("correct", ""),
                })

    if not question_text:
        return jsonify({"error": f"Question {q_ref} not found. Please provide the question text."}), 404

    # Generate detailed AI solution
    solution_prompt = f"""You are an expert {exam_upper} mathematics examiner. Provide a complete, rigorous solution.

QUESTION {q_num or q_ref} ({exam_upper}):
{question_text}

OPTIONS:
{options_text}

SOLUTION FORMAT (follow exactly — NO * or ** anywhere):

◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
📌 Question {q_num or q_ref} — {exam_upper} Solution
◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

📖 Topic: [identify the mathematical topic]
📊 Concept Required: [state the key theorem/technique needed]

🔍 Analysis of Each Option:
Option A: [explain why correct or incorrect with proof/counterexample]
Option B: [explain why correct or incorrect with proof/counterexample]
Option C: [explain why correct or incorrect with proof/counterexample]
Option D: [explain why correct or incorrect with proof/counterexample]

📐 Complete Step-by-Step Solution:
Step 1: [with LaTeX]
Step 2: [with LaTeX]
Step 3: [with LaTeX]
...

✅ CORRECT ANSWER: [letter] — [one line summary of why]

💡 Key Insight: [the most important concept to remember]

📝 Similar Questions to Practice: [suggest 1-2 related problems]

📚 MathSphere: {TEACHER_YOUTUBE}
◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

CRITICAL: Use LaTeX for ALL math. NEVER use * or ** for anything."""

    solution = ask_simple(solution_prompt, system=SYSTEM_PROMPT)

    return jsonify({
        "question_num": q_num or q_ref,
        "exam": exam_upper,
        "question": question_text,
        "solution": solution,
    })


# ── SUBMIT MOCK TEST ──────────────────────────────────────────

@app.route("/api/pyq-submit/<exam>/<test_id>", methods=["POST"])
def submit_pyq_test(exam, test_id):
    try:
        data = request.get_json()
        user_answers    = data.get("answers", {})
        correct_answers = data.get("correct_answers", {})
        solutions       = data.get("solutions", {})
        exam_upper = exam.upper()

        # Try to get correct answers from stored test
        if test_id in _mock_test_store:
            stored = _mock_test_store[test_id]
            for q in stored["questions"]:
                qid = q["id"]
                if qid not in correct_answers:
                    correct_answers[qid] = q.get("correct", "A")

        score = 0
        total = len(user_answers) or len(correct_answers) or 10
        detailed_results = []

        for qid, user_ans in user_answers.items():
            correct_ans = correct_answers.get(qid, "A")
            solution_text = solutions.get(qid, "")
            is_correct = str(user_ans).upper() == str(correct_ans).upper()
            if is_correct:
                score += 1
            detailed_results.append({
                "question_id": qid,
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "solution": clean_response(solution_text),
            })

        percentage = (score / total * 100) if total > 0 else 0

        return jsonify({
            "test_id": test_id,
            "exam": exam_upper,
            "score": score,
            "total": total,
            "percentage": round(percentage, 2),
            "detailed_results": detailed_results,
            "status": "PASSED" if percentage >= 70 else "NEEDS IMPROVEMENT",
            "feedback": _generate_feedback(percentage),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _generate_feedback(percentage):
    if percentage >= 90:
        return f"Excellent! {percentage:.1f}% — Exam ready! Bahut achha kiya! 🎉"
    elif percentage >= 70:
        return f"Good work! {percentage:.1f}% — Keep practicing. Aur thoda mehnat karo! 💪"
    elif percentage >= 50:
        return f"Needs improvement. {percentage:.1f}% — Focus on weak topics carefully."
    else:
        return f"Low score {percentage:.1f}% — Revise fundamentals. MathSphere ke saath dobara practice karo!"


# ── OTHER ROUTES ──────────────────────────────────────────────

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    try:
        d = request.get_json()
        prompt = f"""Generate ONE rigorous multiple-choice question for graduation-level mathematics.
Topic: {d.get("topic","Calculus")}
Difficulty: {d.get("difficulty","medium")}
Question {d.get("q_num",1)} of {d.get("total",5)}

RULES:
- NEVER use * or ** anywhere
- ALL math must use LaTeX: \\( inline \\) and \\[ display \\]
- 4 plausible options at BSc/MSc level
- Double-check the correct answer is mathematically verified
- Show clear chain-of-thought in explanation

FORMAT:
Q: [question with LaTeX]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [A/B/C/D]
EXPLANATION: [step-by-step proof of why this answer is correct]"""

        raw = ask_simple(prompt)
        lines = raw.strip().split('\n')
        ans_line  = next((l for l in lines if l.strip().startswith("ANSWER:")), "ANSWER: A")
        expl_line = next((l for l in lines if l.strip().startswith("EXPLANATION:")), "")
        correct     = ans_line.replace("ANSWER:", "").strip()[:1].upper()
        explanation = expl_line.replace("EXPLANATION:", "").strip()
        question    = '\n'.join(l for l in lines if not l.strip().startswith(("ANSWER:", "EXPLANATION:")))
        return jsonify({"question": question.strip(), "answer": correct, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        p = next((x for x in PARADOXES if x["name"] == name), None)
        return jsonify(p or random.choice(PARADOXES))
    return jsonify(random.choice(PARADOXES))


@app.route("/api/paradoxes")
def all_paradoxes():
    return jsonify(PARADOXES)


@app.route("/api/pyq")
def pyq():
    exam = request.args.get("exam", "JAM").upper()
    qs = PYQ_BANK.get(exam, [])
    if not qs:
        return jsonify({"error": "Not found"}), 404
    q = random.choice(qs)
    return jsonify({
        "q": q.get("q", ""),
        "opts": q.get("opts", {}),
        "correct": q.get("correct", ""),
        "a": clean_response(q.get("a", "")),
        "topic": q.get("topic", ""),
        "year": q.get("year", ""),
    })


@app.route("/api/exam/<exam>")
def exam_info(exam):
    info = EXAM_INFO.get(exam.upper())
    return jsonify(info) if info else (jsonify({"error": "Not found"}), 404)


@app.route("/api/materials/<exam>")
def get_materials(exam):
    return jsonify({"exam": exam.upper(), "topics": {}})


@app.route("/api/formula", methods=["POST"])
def formula():
    topic = request.get_json().get("topic", "")
    prompt = f"""Generate a complete, well-organized formula sheet for: {topic}

Format each formula clearly:
Section Name (use emoji like 📌 📐):
Formula name: \\[ formula in LaTeX \\]
Brief note on when to use it.

NEVER use * or ** for formatting. Include at least 10-15 important formulas."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/calculator", methods=["POST"])
def calculator():
    problem = request.get_json().get("problem", "")
    sympy_r = solve_with_sympy(problem)
    prompt = f"""Solve this step by step: {problem}

Show EVERY step. Use LaTeX for all math. NEVER use * or **.

📌 Problem Type: [identify]
📐 Method: [state the technique]
Step 1: ...
Step 2: ...
✅ Final Answer: [box the answer]"""
    answer = ask_simple(prompt, system=SYSTEM_PROMPT)
    if sympy_r:
        answer = f"{sympy_r}\n\n◆━━━━━━━━━━━━━━━━━━◆\n\n{answer}"
    return jsonify({"answer": answer})


@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic", "")
    prompt = f"""Give TOP 10 rapid revision points for: {topic}

Number each point. For each:
Number. Topic Name: Key fact/formula/theorem in LaTeX.

NEVER use * or **. Focus on exam-critical points for JAM/GATE/CSIR."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic", "")
    prompt = f"""Create a detailed concept map for: {topic}

Include:
1. Core concept and definition
2. Sub-topics and connections
3. Prerequisites
4. Topics this leads to
5. Key theorems
6. Real world applications

NEVER use * or **. Use emoji headers: 📌 📐 💡 ✅"""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/compare", methods=["POST"])
def compare():
    concepts = request.get_json().get("concepts", "")
    prompt = f"""Compare and contrast: {concepts}

Include:
1. Precise definitions of each
2. Key differences (use plain numbered list, NO markdown)
3. Examples where each applies
4. Common confusions and how to avoid them

NEVER use * or **."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/verify", methods=["POST"])
def verify():
    claim = request.get_json().get("claim", "")
    prompt = f"""Verify or find a counterexample for: {claim}

State clearly:
1. Whether TRUE, FALSE, or PARTIALLY TRUE
2. If TRUE: rigorous proof with LaTeX
3. If FALSE: specific counterexample
4. Related correct statements

NEVER use * or **."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/latex", methods=["POST"])
def latex():
    text = request.get_json().get("text", "")
    return jsonify({"answer": ask_simple(
        f"Generate clean LaTeX for: {text}. Use \\[ \\] for display math or \\( \\) for inline.",
        system=SYSTEM_PROMPT
    )})


@app.route("/api/projects", methods=["POST"])
def projects():
    domain = request.get_json().get("domain", "")
    prompt = f"""Give 3 detailed real-life mathematics project ideas for: {domain}

For each project:
1. Project Title
2. Mathematical concepts used
3. Problem statement
4. Methodology
5. Expected output and difficulty level

NEVER use * or **."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


@app.route("/api/proof", methods=["POST"])
def proof():
    d = request.get_json()
    msgs = d.get("history", []) + [{"role": "user", "content": d.get("theorem", "")}]
    return jsonify({"answer": ask_ai(msgs, system=SYSTEM_PROMPT)})


@app.route("/api/debate", methods=["POST"])
def debate():
    d = request.get_json()
    msgs = d.get("history", []) + [{"role": "user", "content": d.get("argument", "")}]
    return jsonify({"answer": ask_ai(msgs, system=SYSTEM_PROMPT)})


@app.route("/api/research", methods=["POST"])
def research():
    d = request.get_json()
    return jsonify({"answer": ask_simple(d.get("question", ""), system=SYSTEM_PROMPT)})


# ── GRAPH VISUALIZATION ───────────────────────────────────────

@app.route("/api/plot-2d", methods=["POST"])
def plot_2d():
    try:
        data = request.get_json()
        equation = data.get("equation", "x**2")
        x_min = data.get("x_min", -10)
        x_max = data.get("x_max", 10)
        x = np.linspace(x_min, x_max, 500)
        safe_env = {
            "x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
            "pi": np.pi, "e": np.e,
        }
        try:
            y = eval(equation.replace('^', '**'), {"__builtins__": {}}, safe_env)
        except Exception as e:
            return jsonify({"error": f"Invalid equation: {str(e)}"}), 400
        if not isinstance(y, np.ndarray):
            y = np.full_like(x, float(y))
        y_clean = np.where(np.isfinite(y), y, np.nan)
        roots = []
        for i in range(len(y_clean)-1):
            if (not np.isnan(y_clean[i])) and (not np.isnan(y_clean[i+1])):
                if y_clean[i] * y_clean[i+1] < 0:
                    roots.append(round(float(x[i]), 4))
        dy = np.gradient(y_clean)
        critical_points = []
        for i in range(1, len(dy)-1):
            if (not np.isnan(dy[i-1])) and (not np.isnan(dy[i+1])):
                if dy[i-1] * dy[i+1] < 0:
                    critical_points.append({
                        "x": round(float(x[i]), 4),
                        "y": round(float(y_clean[i]), 4),
                        "type": "maximum" if dy[i-1] > 0 else "minimum",
                    })
        return jsonify({
            "x": x.tolist(),
            "y": y_clean.tolist(),
            "equation": equation,
            "roots": roots,
            "critical_points": critical_points[:10],
            "analysis": {
                "domain": f"[{x_min}, {x_max}]",
                "number_of_roots": len(roots),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── TRACKER ───────────────────────────────────────────────────

TRACKER_DATA = {
    "default": {
        "JAM": {
            "Real Analysis": {
                "Limits and Continuity": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Derivatives":           {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Integration":           {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Sequences and Series":  {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Linear Algebra": {
                "Vector Spaces":         {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Eigenvalues":           {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
        },
        "GATE": {
            "Calculus": {
                "Multivariate Calculus": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Complex Analysis": {
                "Analytic Functions":    {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
        },
        "CSIR": {
            "Analysis": {
                "Metric Spaces":         {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Topology": {
                "Compactness":           {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
        }
    }
}


@app.route("/api/tracker/<exam>")
def get_tracker(exam):
    exam_upper = exam.upper()
    tracker = TRACKER_DATA["default"].get(exam_upper, {})
    if not tracker:
        return jsonify({"error": "Exam not found"}), 404
    total_items = completed_items = 0
    structure = {}
    for topic, subtopics in tracker.items():
        structure[topic] = {}
        for subtopic, status in subtopics.items():
            total_items += 4
            comp = sum(1 for v in status.values() if v)
            completed_items += comp
            structure[topic][subtopic] = {**status, "completion_percentage": comp / 4 * 100}
    overall = (completed_items / total_items * 100) if total_items > 0 else 0
    return jsonify({
        "exam": exam_upper,
        "topics": structure,
        "overall_progress": round(overall, 2),
        "completed_items": completed_items,
        "total_items": total_items,
    })


@app.route("/api/tracker/update", methods=["POST"])
def update_tracker():
    try:
        data     = request.get_json()
        exam     = data.get("exam", "").upper()
        topic    = data.get("topic", "")
        subtopic = data.get("subtopic", "")
        item     = data.get("item", "")
        status   = data.get("status", False)
        if exam not in TRACKER_DATA["default"]:
            return jsonify({"error": "Exam not found"}), 404
        if topic not in TRACKER_DATA["default"][exam]:
            return jsonify({"error": "Topic not found"}), 404
        if subtopic not in TRACKER_DATA["default"][exam][topic]:
            return jsonify({"error": "Subtopic not found"}), 404
        TRACKER_DATA["default"][exam][topic][subtopic][item] = status
        return jsonify({"status": "success", "message": f"{item} updated to {status}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── CONCEPT CHECKER ───────────────────────────────────────────

CONCEPT_QUESTIONS = {
    "JAM": {
        "Limits and Continuity": [
            {"question": "What is the epsilon-delta definition of limit?",
             "options": {"A": "For every ε > 0, there exists δ > 0 such that |x-a| < δ implies |f(x)-L| < ε",
                         "B": "For every δ > 0, there exists ε > 0 such that |f(x)-L| < δ implies |x-a| < ε",
                         "C": "f(x) approaches L as x gets very large",
                         "D": "f(x) is continuous at point a"},
             "correct": "A"},
            {"question": "A function is continuous at x=a if:",
             "options": {"A": "lim(x→a) f(x) = f(a)",
                         "B": "f(a) is defined",
                         "C": "f(x) is differentiable at a",
                         "D": "f(x) > 0 for all x near a"},
             "correct": "A"},
            {"question": "What does the Intermediate Value Theorem state?",
             "options": {"A": "If f is continuous on [a,b] and k is between f(a) and f(b), then f(c)=k for some c in (a,b)",
                         "B": "Every function has a limit at every point",
                         "C": "Every continuous function is differentiable",
                         "D": "The derivative exists at every point"},
             "correct": "A"},
        ],
        "Derivatives": [
            {"question": "The formal definition of derivative is:",
             "options": {"A": "f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
                         "B": "f'(x) = f(x+1) - f(x)",
                         "C": "f'(x) = [f(b) - f(a)] / (b - a)",
                         "D": "f'(x) = f(x) × 2"},
             "correct": "A"},
            {"question": "The Mean Value Theorem states:",
             "options": {"A": "If f is continuous on [a,b] and differentiable on (a,b), then f'(c) = (f(b)-f(a))/(b-a) for some c",
                         "B": "The average of f equals f at the midpoint",
                         "C": "f' = 0 at every interior extremum",
                         "D": "f is constant if f' = 0"},
             "correct": "A"},
        ]
    },
    "GATE": {
        "Linear Algebra": [
            {"question": "An eigenvalue is:",
             "options": {"A": "A scalar λ such that Av = λv for non-zero vector v",
                         "B": "A vector perpendicular to another",
                         "C": "The determinant of a matrix",
                         "D": "Always a positive number"},
             "correct": "A"},
            {"question": "Linearly independent vectors means:",
             "options": {"A": "No vector is a linear combination of the others",
                         "B": "All vectors point in the same direction",
                         "C": "All vectors have unit magnitude",
                         "D": "The vectors are orthogonal"},
             "correct": "A"},
        ]
    },
    "NET": {
        "Analysis": [
            {"question": "A Banach space is:",
             "options": {"A": "A complete normed vector space",
                         "B": "An inner product space",
                         "C": "A finite-dimensional space",
                         "D": "A compact metric space"},
             "correct": "A"},
        ]
    }
}


@app.route("/api/concept-check/<exam>/<topic>", methods=["POST"])
def concept_checker(exam, topic):
    exam_upper = exam.upper()
    if exam_upper not in CONCEPT_QUESTIONS:
        return jsonify({"error": "Exam not found"}), 404
    if topic not in CONCEPT_QUESTIONS[exam_upper]:
        return jsonify({"error": "Topic not found"}), 404
    questions = CONCEPT_QUESTIONS[exam_upper][topic]
    return jsonify({
        "exam": exam_upper, "topic": topic,
        "total_questions": len(questions),
        "questions": [{"id": f"Q{i+1}", "question": q["question"], "options": q["options"]} for i, q in enumerate(questions)],
        "pass_score": 80,
    })


@app.route("/api/concept-check/submit/<exam>/<topic>", methods=["POST"])
def submit_concept_check(exam, topic):
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
            qid = f"Q{i+1}"
            user_ans = user_answers.get(qid, "").upper()
            correct = q["correct"]
            is_correct = user_ans == correct
            if is_correct:
                score += 1
            feedback_list.append({
                "question": q["question"],
                "status": "Correct" if is_correct else "Incorrect",
                "your_answer": user_ans,
                "correct_answer": correct,
            })
        percentage = (score / total * 100) if total > 0 else 0
        return jsonify({
            "exam": exam_upper, "topic": topic, "score": score, "total": total,
            "percentage": round(percentage, 2),
            "status": "CONCEPT CLEAR!" if percentage >= 80 else "REVIEW NEEDED",
            "feedback": feedback_list,
            "message": _concept_feedback(percentage),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _concept_feedback(p):
    if p >= 100: return "Perfect! You have completely mastered this concept!"
    if p >= 80:  return "Concept is clear! Well done. Bahut achha! 🎉"
    if p >= 60:  return "Partial understanding — review the material once more."
    return "Concept needs more work — go back to definitions and examples."


# ═══════════════════════════════════════════════════════════
# ── MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"MathSphere v3.0 starting on port {port}")
    print(f"Fixes active: asterisk stripping, AI mock test generation, per-question solutions")
    app.run(host="0.0.0.0", port=port, debug=False)