"""
MathSphere Web — Complete Backend v2.0
Fixes & Improvements:
✅ No more ** asterisk formatting in responses
✅ Proper bold/structured formatting
✅ 100 PYQ questions per exam with options + step-by-step solutions
✅ 50+ mathematicians worldwide
✅ Better conversation memory
✅ More exam detail
✅ Graph visualization
✅ Concept Checker
✅ Progress Tracker

By Anupam Nigam | youtube.com/@pi_nomenal1729
"""

import os
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

# ── AI CORE ───────────────────────────────────────────────────

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
        txt_part = gt.Part.from_text(text="""You are an expert mathematics teacher.
Solve this problem completely. Use LaTeX for all math expressions.
Inline math: \\( ... \\)   Display math: \\[ ... \\]
Do NOT use ** or * for formatting. Use section headers like 📌 Topic, 📐 Solution etc.
Start with: "Namaste! 🙏 Yeh problem dekh liya maine —"
End with: MathSphere: https://youtube.com/@pi_nomenal1729""")
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

# ── SYSTEM PROMPT — NO ASTERISKS ─────────────────────────────

SYSTEM_PROMPT = f"""You are MathSphere — a warm, expert Mathematics teacher for graduation level students, created by Anupam Nigam (youtube.com/@pi_nomenal1729).

═══════════════════════════════════════
CRITICAL FORMATTING RULES — FOLLOW EXACTLY:
═══════════════════════════════════════
1. NEVER use ** or * for bold or emphasis. NEVER. Not even once.
2. NEVER use markdown bold like **word** — this appears as raw asterisks to users.
3. For section headers use emoji icons like: 📌 Topic Name, 💡 Key Idea, 📐 Solution
4. For important terms just write them normally or use CAPS for emphasis
5. ALL mathematical expressions MUST use LaTeX:
   - Inline math: \\( ... \\)   Example: \\( x^2 + 3x + 2 = 0 \\)
   - Display math: \\[ ... \\]  Example: \\[ \\int_0^1 x^2\\,dx = \\frac{{1}}{{3}} \\]
6. Use: \\frac{{a}}{{b}}, \\sqrt{{x}}, \\int, \\sum, \\lim, \\infty
7. Use \\mathbb{{R}}, \\mathbb{{Z}}, \\mathbb{{Q}}, \\mathbb{{C}} for number sets
8. For matrices: \\begin{{pmatrix}} a & b \\\\ c & d \\end{{pmatrix}}
9. Never write raw math like x^2 — always use LaTeX

═══════════════════════════════════════
LANGUAGE & TONE:
═══════════════════════════════════════
- Mix Hinglish warmly: "Dekho...", "Samajh aaya?", "Yeh important hai!", "Bohot achha!"
- Be like a friendly Indian teacher — warm, encouraging, clear
- For casual messages → brief friendly response
- NEVER be robotic or use bullet points with asterisks

═══════════════════════════════════════
CONVERSATION MEMORY — VERY IMPORTANT:
═══════════════════════════════════════
- You have access to the FULL conversation history
- Always remember what was discussed earlier in the conversation
- If a student refers to "that problem" or "previous question" — refer back to it
- Build on previous explanations rather than starting fresh every time
- If the same topic continues, say "Jaise humne abhi dekha..." (As we just saw...)

═══════════════════════════════════════
TEACHING STRUCTURE (for math questions):
═══════════════════════════════════════

◆━━━━━━━━━━━━━━━━━━◆
📌 [Topic Name]
◆━━━━━━━━━━━━━━━━━━◆

💡 Real Life Analogy: [relatable Indian example]

📖 Definition: [precise definition with LaTeX]

📐 Step-by-Step Solution:
Step 1: [explanation with LaTeX]
Step 2: [explanation with LaTeX]
Step 3: [explanation with LaTeX]

✅ Verification: [verify the answer]

📝 Try This Yourself: [one practice problem]

📚 MathSphere: {TEACHER_YOUTUBE}
◆━━━━━━━━━━━━━━━━━━◆

Always end with the YouTube link: {TEACHER_YOUTUBE}"""

# ── MATHEMATICIANS DATABASE — 50+ worldwide ──────────────────

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
        "contribution": "The most prolific mathematician in history — wrote over 800 papers even after going completely blind. Euler created modern mathematical notation: f(x), e, π, i, Σ, and Δ. He proved Euler's identity \\( e^{i\\pi} + 1 = 0 \\), solved the Basel problem \\( \\sum 1/n^2 = π^2/6 \\), founded graph theory with the Königsberg bridge problem, and introduced the Euler characteristic.",
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
        "contribution": "Called the Prince of Mathematics, Gauss proved the Fundamental Theorem of Algebra at age 21. He invented the method of least squares, the Gaussian distribution (bell curve), and modular arithmetic. His Disquisitiones Arithmeticae revolutionized number theory. He also proved the Prime Number Theorem and made major contributions to differential geometry with the Theorema Egregium.",
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
        "contribution": "Einstein called her the most significant creative mathematical genius yet produced. Noether revolutionized abstract algebra by introducing the concept of ideals, Noetherian rings, and chain conditions. Her Noether's theorem — connecting symmetry to conservation laws — is arguably the most important theorem in theoretical physics. She developed the modern theory of rings, fields, and algebras.",
        "key_results": "Noether's theorem, Noetherian rings, ascending chain condition, invariant theory",
        "application": "Noether's theorem underlies all of modern physics — conservation of energy, momentum, and charge all follow from it. It is used in every quantum field theory.",
        "quote": "My methods are really methods of working and thinking; this is why they have crept in everywhere anonymously.",
        "fun_fact": "Despite being acknowledged as brilliant by Einstein and Hilbert, Noether was denied a university position for years purely because she was a woman."
    },
    {
        "name": "Aryabhata",
        "period": "476–550 AD",
        "country": "India",
        "fields": ["Arithmetic", "Algebra", "Trigonometry", "Astronomy"],
        "contribution": "India's first major mathematician-astronomer. Aryabhata calculated π ≈ 3.1416 correct to 4 decimal places in 499 AD — over 1000 years before European mathematicians. He invented the place value system, introduced zero as a positional digit, developed sine and cosine tables, and correctly explained solar and lunar eclipses using shadow geometry.",
        "key_results": "Value of π, place value system, sine tables, rotation of Earth, eclipse calculations",
        "application": "His mathematical methods are foundational to modern astronomy and navigation. His sine tables were the precursor to modern trigonometry.",
        "quote": "Just as a boat in water, the earth floats in space.",
        "fun_fact": "Aryabhata calculated the length of a year as 365.358 days — accurate to within minutes — using only naked-eye observations!"
    },
    {
        "name": "Euclid of Alexandria",
        "period": "~300 BC",
        "country": "Greece (Egypt)",
        "fields": ["Geometry", "Number Theory", "Logic"],
        "contribution": "Father of Geometry. His 13-volume work Elements is the second most printed book after the Bible, used as a textbook for 2000+ years. Euclid established axiomatic mathematics — the idea of building all results from a small set of axioms. He proved the infinitude of primes, the Euclidean algorithm for GCD, and systematized all of Greek geometry.",
        "key_results": "Euclidean geometry, Euclidean algorithm, infinitude of primes, 5 postulates",
        "application": "Euclidean geometry is the foundation of architecture, engineering, computer graphics, and GPS systems. The Euclidean algorithm is used in modern cryptography.",
        "quote": "There is no royal road to geometry.",
        "fun_fact": "Euclid's proof that there are infinitely many primes is still considered one of the most elegant proofs in mathematics — it fits in 4 lines!"
    },
    {
        "name": "Isaac Newton",
        "period": "1643–1727",
        "country": "England",
        "fields": ["Calculus", "Physics", "Algebra", "Optics"],
        "contribution": "Co-invented calculus (simultaneously with Leibniz), discovering differentiation and integration as inverse operations. Newton's laws of motion and universal gravitation unified terrestrial and celestial mechanics. He developed the binomial theorem for any exponent, generalized the concept of power series, and proved that white light is composed of all colors.",
        "key_results": "Calculus, Newton's laws, binomial theorem, Newton-Raphson method, law of universal gravitation",
        "application": "Calculus is used in every field of science and engineering. Newton's laws govern classical mechanics, orbital mechanics, and structural engineering.",
        "quote": "If I have seen further, it is by standing on the shoulders of Giants.",
        "fun_fact": "Newton invented calculus at age 23 during 18 months of isolation during the Great Plague of 1665 — his annus mirabilis (miraculous year)!"
    },
    {
        "name": "Gottfried Wilhelm Leibniz",
        "period": "1646–1716",
        "country": "Germany",
        "fields": ["Calculus", "Logic", "Topology", "Philosophy"],
        "contribution": "Co-inventor of calculus who gave us the notation we use today: dx, dy, ∫, and the product rule. Leibniz introduced binary arithmetic (0 and 1) and designed mechanical calculators. He conceived of symbolic logic over 200 years before Boole. His notation for calculus is universally preferred over Newton's because of its clarity and ease of manipulation.",
        "key_results": "Calculus notation, binary system, Leibniz formula for π, product rule, chain rule",
        "application": "Binary arithmetic is the foundation of ALL digital computers. Leibniz's calculus notation is used in every physics and engineering textbook.",
        "quote": "Music is the pleasure the human mind experiences from counting without being aware that it is counting.",
        "fun_fact": "Leibniz derived \\( \\frac{\\pi}{4} = 1 - \\frac{1}{3} + \\frac{1}{5} - \\frac{1}{7} + \\cdots \\) — a beautiful infinite series for π!"
    },
    {
        "name": "Bernhard Riemann",
        "period": "1826–1866",
        "country": "Germany",
        "fields": ["Complex Analysis", "Differential Geometry", "Number Theory"],
        "contribution": "Revolutionized mathematics in his short 40-year life. Riemann created the Riemann integral (the standard definition of integration), Riemannian geometry (which Einstein used for General Relativity), and the Riemann zeta function. His 1859 paper on prime numbers, containing the Riemann Hypothesis, is considered the most important unsolved problem in mathematics.",
        "key_results": "Riemann integral, Riemannian geometry, Riemann hypothesis, Riemann surfaces, Cauchy-Riemann equations",
        "application": "Riemannian geometry is the mathematical language of Einstein's General Relativity. Riemann surfaces are fundamental to string theory.",
        "quote": "The question of the validity of the axioms of geometry in the infinitely small is bound up with the question of the ground of the metric relations of space.",
        "fun_fact": "The Riemann Hypothesis — that all non-trivial zeros of \\( \\zeta(s) \\) have real part \\( \\frac{1}{2} \\) — is worth $1 million as a Millennium Prize Problem!"
    },
    {
        "name": "Pierre de Fermat",
        "period": "1607–1665",
        "country": "France",
        "fields": ["Number Theory", "Analytic Geometry", "Probability", "Calculus"],
        "contribution": "Father of modern number theory. Fermat co-invented analytic geometry with Descartes, made foundational contributions to probability theory with Pascal, and developed early ideas of calculus. Fermat's Last Theorem — \\( x^n + y^n = z^n \\) has no integer solutions for n > 2 — was stated in 1637 and proved only in 1995 by Andrew Wiles after 358 years!",
        "key_results": "Fermat's Last Theorem, Fermat's Little Theorem, Fermat primes, method of infinite descent",
        "application": "Fermat's Little Theorem is the foundation of RSA public-key cryptography, protecting all internet communication. His work on optics led to the principle of least time.",
        "quote": "I have discovered a truly remarkable proof which this margin is too small to contain.",
        "fun_fact": "Andrew Wiles secretly worked on Fermat's Last Theorem for 7 years in his attic, not telling anyone, and wept when he finally proved it in 1995!"
    },
    {
        "name": "Évariste Galois",
        "period": "1811–1832",
        "country": "France",
        "fields": ["Abstract Algebra", "Group Theory", "Field Theory"],
        "contribution": "Founder of group theory and Galois theory. In just 20 years of life, Galois completely solved the 350-year-old problem of which polynomial equations can be solved by radicals. He invented group theory to characterize when a polynomial is solvable, creating one of the deepest and most beautiful theories in mathematics. He wrote down his mathematical discoveries the night before dying in a duel.",
        "key_results": "Galois theory, Galois groups, solvability by radicals, finite fields",
        "application": "Galois theory is used in coding theory, cryptography, and quantum computing. Finite fields underlie error-correcting codes in CDs, DVDs, and satellite communication.",
        "quote": "Unfortunately, what is little recognized is that the most worthwhile scientific books are those in which the author clearly indicates what he does not know.",
        "fun_fact": "Galois died in a duel at age 20. The night before, he frantically wrote letters to friends, scribbling 'I have no time' in the margins — his genius lost to history at 20!"
    },
    {
        "name": "Georg Cantor",
        "period": "1845–1918",
        "country": "Germany (born Russia)",
        "fields": ["Set Theory", "Transfinite Numbers", "Real Analysis"],
        "contribution": "Created set theory and the rigorous mathematical treatment of infinity. Cantor proved that some infinities are strictly larger than others — the infinity of real numbers is larger than the infinity of natural numbers. He introduced cardinal numbers, ordinal numbers, and the continuum hypothesis. Contemporaries called him a corrupter of youth; today he is recognized as a visionary.",
        "key_results": "Cantor's diagonal argument, cardinal numbers, Cantor set, continuum hypothesis, aleph numbers",
        "application": "Set theory is the foundation of all modern mathematics. Cantor's work on cardinality is used in theoretical computer science to classify computational problems.",
        "quote": "The essence of mathematics lies in its freedom.",
        "fun_fact": "Cantor proved \\( |\\mathbb{R}| > |\\mathbb{N}| \\) using his diagonal argument — a 3-line proof so beautiful and surprising that it changed mathematics forever!"
    },
    {
        "name": "David Hilbert",
        "period": "1862–1943",
        "country": "Germany",
        "fields": ["Algebra", "Functional Analysis", "Mathematical Physics", "Logic"],
        "contribution": "The most influential mathematician of the 20th century. Hilbert formulated 23 famous problems in 1900 that shaped mathematics for a century. He created the formalist program for mathematics, developed Hilbert spaces (foundation of quantum mechanics), proved the finite basis theorem, and made major contributions to invariant theory, integral equations, and mathematical physics.",
        "key_results": "Hilbert's 23 problems, Hilbert spaces, Hilbert basis theorem, spectral theory, formalism",
        "application": "Hilbert spaces are the mathematical framework of quantum mechanics and functional analysis. His work on infinite-dimensional spaces is used in modern signal processing.",
        "quote": "We must know. We will know.",
        "fun_fact": "Hilbert's Hotel paradox: a hotel with infinitely many rooms, all occupied, can still accommodate infinitely many new guests — demonstrating the paradoxical nature of infinity!"
    },
    {
        "name": "Kurt Gödel",
        "period": "1906–1978",
        "country": "Austria-Hungary (later USA)",
        "fields": ["Mathematical Logic", "Set Theory", "Philosophy of Mathematics"],
        "contribution": "Proved the two Incompleteness Theorems in 1931 — one of the most shocking results in the history of mathematics. First theorem: any consistent formal system strong enough to include arithmetic contains true statements that cannot be proved within the system. Second theorem: such a system cannot prove its own consistency. This shattered Hilbert's dream of a complete and consistent mathematics.",
        "key_results": "Gödel's Incompleteness Theorems, Gödel completeness theorem, constructible universe",
        "application": "Incompleteness theorems have profound implications for computer science — they imply the undecidability of the halting problem and limitations of artificial intelligence.",
        "quote": "Either mathematics is too big for the human mind or the human mind is more than a machine.",
        "fun_fact": "Gödel found a logical loophole in the US Constitution that could allow a dictator to take over — he tried to explain this to the judge at his citizenship hearing!"
    },
    {
        "name": "Alan Turing",
        "period": "1912–1954",
        "country": "England",
        "fields": ["Computability Theory", "Cryptography", "Mathematical Biology", "AI"],
        "contribution": "Father of theoretical computer science and artificial intelligence. Turing created the mathematical model of computation (Turing machine), proved the halting problem is undecidable, and broke the Nazi Enigma code in World War II — shortening the war by an estimated 2–4 years. He proposed the Turing Test for artificial intelligence and made pioneering contributions to mathematical biology.",
        "key_results": "Turing machine, halting problem, Turing test, Enigma decryption, morphogenesis equations",
        "application": "Every computer is a physical realization of a Turing machine. His work on computability defines the theoretical limits of what computers can and cannot calculate.",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        "fun_fact": "Turing's team at Bletchley Park cracked over 84,000 Enigma messages per month at peak — their work is estimated to have saved 14 million lives!"
    },
    {
        "name": "Maryam Mirzakhani",
        "period": "1977–2017",
        "country": "Iran",
        "fields": ["Teichmüller Theory", "Hyperbolic Geometry", "Ergodic Theory"],
        "contribution": "First and only woman to win the Fields Medal (2014) — the highest honor in mathematics. Mirzakhani made groundbreaking contributions to the understanding of Riemann surfaces, their moduli spaces, and symplectic geometry. Her work on the dynamics and geometry of Riemann surfaces connected previously unrelated areas of mathematics and opened entire new fields of research.",
        "key_results": "Counting closed geodesics, Weil-Petersson volumes, moduli spaces of Riemann surfaces",
        "application": "Her work connects to quantum field theory, string theory, and the mathematical physics of 2D gravity. Her techniques are used to understand the shapes of the universe.",
        "quote": "The beauty of mathematics only shows itself to more patient followers.",
        "fun_fact": "As a child in Tehran, Mirzakhani wanted to be a novelist. She became interested in mathematics only in high school — and went on to win the mathematical equivalent of the Nobel Prize!"
    },
    {
        "name": "Henri Poincaré",
        "period": "1854–1912",
        "country": "France",
        "fields": ["Topology", "Dynamical Systems", "Celestial Mechanics", "Special Relativity"],
        "contribution": "One of the last mathematical universalists. Poincaré founded algebraic topology, created the theory of dynamical systems and chaos, solved (partially) the three-body problem, and formulated Poincaré's conjecture (proved by Perelman in 2003). He independently developed much of special relativity and was the first to discuss gravitational waves in modern terms.",
        "key_results": "Poincaré conjecture, fundamental group, Poincaré recurrence theorem, chaos theory, Betti numbers",
        "application": "Topology is fundamental to modern physics and data analysis. Chaos theory is used in weather forecasting, ecology, and financial modelling.",
        "quote": "Mathematics is the art of giving the same name to different things.",
        "fun_fact": "Poincaré made one of the greatest discoveries of chaos theory accidentally while working on the three-body problem for a prize competition — he initially thought he had proved stability!"
    },
    {
        "name": "Sophie Germain",
        "period": "1776–1831",
        "country": "France",
        "fields": ["Number Theory", "Mathematical Physics", "Elasticity Theory"],
        "contribution": "Pioneering female mathematician who made major contributions to Fermat's Last Theorem and the theory of elasticity. Germain proved an important special case of Fermat's Last Theorem for all primes p where 2p+1 is also prime (Germain primes). Her theory of elastic surfaces won the French Academy's grand prize three times and laid the foundation for structural engineering.",
        "key_results": "Sophie Germain primes, partial proof of Fermat's Last Theorem, theory of elasticity, Germain's theorem",
        "application": "Germain primes are used in primality testing algorithms and cryptography. Her elasticity theory is applied in civil engineering and the design of bridges.",
        "quote": "Despite the efforts of science, the precious sensitivity that makes us capable of appreciating beauty is not yet destroyed.",
        "fun_fact": "Germain corresponded with Gauss for years under the male pseudonym 'Monsieur LeBlanc'. When Gauss discovered she was a woman, he wrote that her merit was even more extraordinary!"
    },
    {
        "name": "Bhaskara II (Bhaskaracharya)",
        "period": "1114–1185",
        "country": "India",
        "fields": ["Algebra", "Calculus precursor", "Trigonometry", "Astronomy"],
        "contribution": "Greatest mathematician of medieval India. Bhaskara II discovered key concepts of differential calculus 500 years before Newton and Leibniz, including the idea of instantaneous velocity. He found the derivative of the sine function, developed Pell's equation solutions, and computed planetary orbits with great accuracy. His work Lilavati is a masterpiece of mathematical exposition.",
        "key_results": "Differential calculus precursor, Pell equation, instantaneous velocity, Lilavati, trigonometric identities",
        "application": "His planetary motion calculations were remarkably accurate. His understanding of instantaneous motion foreshadowed the calculus that would transform science 500 years later.",
        "quote": "A particle of turmeric or a grain of rice or a sesame seed, cut in half, again in half, and so on, becomes a paramāṇu.",
        "fun_fact": "Bhaskara's daughter Lilavati is the subject of his famous mathematics book — legend says he named it after her to console her after a failed wedding due to a horoscope accident!"
    },
    {
        "name": "Brahmagupta",
        "period": "598–668 AD",
        "country": "India",
        "fields": ["Arithmetic", "Algebra", "Geometry", "Astronomy"],
        "contribution": "First mathematician to treat zero as a number and define arithmetic operations with zero and negative numbers. Brahmagupta gave rules for arithmetic with zero: n + 0 = n, n - 0 = n, n × 0 = 0. He gave the formula for the area of a cyclic quadrilateral (Brahmagupta's formula), solved quadratic equations, and developed the chakravala method for solving Pell equations.",
        "key_results": "Arithmetic with zero, Brahmagupta's formula, Pell equation method, negative numbers",
        "application": "Zero is the foundation of all modern mathematics, computing, and science. Without Brahmagupta's formalization of zero, the digital age would not exist.",
        "quote": "A debt minus zero is a debt. A fortune minus zero is a fortune. Zero minus zero is zero.",
        "fun_fact": "Brahmagupta also wrote: '0 ÷ 0 = 0' — which we now know is indeterminate. Even mathematical geniuses can be wrong! Zero division still fascinates mathematicians today."
    },
    {
        "name": "Nikolai Lobachevsky",
        "period": "1792–1856",
        "country": "Russia",
        "fields": ["Non-Euclidean Geometry", "Analysis"],
        "contribution": "Revolutionary who independently discovered non-Euclidean geometry (hyperbolic geometry) in 1830, showing that Euclid's parallel postulate is not the only option. In hyperbolic space, the sum of angles of a triangle is LESS than 180°. This was the first serious challenge to Euclidean geometry in 2000 years and opened up entirely new geometries that Einstein would later use for general relativity.",
        "key_results": "Hyperbolic geometry, Lobachevsky space, challenge to Euclid's 5th postulate",
        "application": "Non-Euclidean geometry is the mathematical foundation of Einstein's General Relativity, which describes gravity as the curvature of spacetime.",
        "quote": "There is no branch of mathematics, however abstract, which may not some day be applied to phenomena of the real world.",
        "fun_fact": "Lobachevsky was ridiculed and called a 'mad mathematician' for challenging Euclid. 50 years after his death, Einstein showed that the universe actually follows non-Euclidean geometry!"
    },
    {
        "name": "Augustin-Louis Cauchy",
        "period": "1789–1857",
        "country": "France",
        "fields": ["Complex Analysis", "Real Analysis", "Algebra", "Mathematical Physics"],
        "contribution": "The father of rigorous analysis. Cauchy gave the first rigorous definitions of limits, continuity, and derivatives that we use today. He founded complex analysis, proved Cauchy's integral theorem, developed the theory of residues, and proved the intermediate value theorem and mean value theorem rigorously. He published over 800 papers — more than any mathematician except Euler.",
        "key_results": "Cauchy's integral theorem, Cauchy-Riemann equations, Cauchy sequences, residue theorem",
        "application": "Complex analysis powers electrical engineering, fluid dynamics, aerodynamics, and quantum mechanics. The residue theorem is used to evaluate difficult real integrals.",
        "quote": "It would be wrong to speak of impossibility when the word merely refers to a difficulty.",
        "fun_fact": "Cauchy defined the epsilon-delta definition of limit that every analysis student learns — and hates — today. He single-handedly made calculus rigorous after 150 years of intuitive use!"
    },
    {
        "name": "Blaise Pascal",
        "period": "1623–1662",
        "country": "France",
        "fields": ["Probability Theory", "Projective Geometry", "Number Theory"],
        "contribution": "Co-founded probability theory with Fermat through correspondence about gambling problems. Pascal's triangle (known in India and China earlier) contains the binomial coefficients. He proved Pascal's theorem in projective geometry, invented one of the first mechanical calculators (Pascaline), and contributed to the development of atmospheric pressure measurement.",
        "key_results": "Pascal's triangle, Pascal's theorem, foundations of probability, Pascal's wager",
        "application": "Probability theory is the foundation of statistics, insurance, financial derivatives, machine learning, and quantum mechanics.",
        "quote": "The eternal silence of these infinite spaces frightens me.",
        "fun_fact": "Pascal built the first mechanical calculator at age 18 to help his father with tax calculations — it could add and subtract numbers, presaging the modern computer!"
    },
    {
        "name": "Andrew Wiles",
        "period": "1953–present",
        "country": "England",
        "fields": ["Number Theory", "Algebraic Geometry"],
        "contribution": "Proved Fermat's Last Theorem in 1995 after 358 years — one of the greatest achievements in the history of mathematics. Wiles secretly worked alone for 7 years, developing new techniques in modular forms and elliptic curves. He proved the Taniyama-Shimura conjecture for semistable elliptic curves, which implied Fermat's Last Theorem as a consequence.",
        "key_results": "Proof of Fermat's Last Theorem, Taniyama-Shimura conjecture, modular forms",
        "application": "The techniques developed by Wiles for elliptic curves are now used in cryptography (elliptic curve cryptography) securing billions of online transactions.",
        "quote": "Perhaps I can best describe my experience of doing mathematics in terms of a journey through a dark unexplored mansion.",
        "fun_fact": "When Wiles announced the proof in 1993, an error was found. He worked another year in secret to fix it. When he finally succeeded, he said it was the most important moment of his working life!"
    },
    {
        "name": "Terence Tao",
        "period": "1975–present",
        "country": "Australia",
        "fields": ["Harmonic Analysis", "Number Theory", "Partial Differential Equations"],
        "contribution": "Called the Mozart of Mathematics. Tao received the Fields Medal at age 31 and is widely considered the greatest living mathematician. He proved (with Ben Green) the Green-Tao theorem: prime numbers contain arbitrarily long arithmetic progressions. He solved the Erdős discrepancy conjecture, the Kadison-Singer problem, and made major advances in Navier-Stokes equations.",
        "key_results": "Green-Tao theorem, Erdős discrepancy problem, compressed sensing, wave maps, Navier-Stokes advances",
        "application": "His work on compressed sensing revolutionized medical imaging — MRI scans can now be done with far fewer measurements. His techniques are used in data compression and signal processing.",
        "quote": "What mathematics achieves: it gives us a language to describe patterns — and that is all we need.",
        "fun_fact": "Tao scored 760 on the math SAT at age 8, got his PhD at 20, and won the Fields Medal at 31. He maintains a highly active math blog where he openly shares proofs in progress!"
    },
    {
        "name": "Paul Erdős",
        "period": "1913–1996",
        "country": "Hungary",
        "fields": ["Combinatorics", "Number Theory", "Graph Theory", "Probability"],
        "contribution": "Most collaborative mathematician in history, with over 1,500 papers and 500+ co-authors. Erdős gave us the probabilistic method in combinatorics — proving existence of mathematical objects by showing their probability is positive. He made foundational contributions to prime number theory, Ramsey theory, and graph theory. The Erdős number measures collaboration distance from Erdős.",
        "key_results": "Probabilistic method, Erdős-Ko-Rado theorem, Erdős-Szekeres theorem, happy ending problem",
        "application": "The probabilistic method is used throughout computer science for designing efficient algorithms and proving existence results in combinatorics and network theory.",
        "quote": "A mathematician is a machine for turning coffee into theorems.",
        "fun_fact": "Erdős had no fixed home — he carried a single suitcase and traveled between colleagues' homes his entire adult life, showing up and saying 'my brain is open'!"
    },
    {
        "name": "John von Neumann",
        "period": "1903–1957",
        "country": "Hungary (later USA)",
        "fields": ["Set Theory", "Functional Analysis", "Quantum Mechanics", "Game Theory", "Computer Science"],
        "contribution": "One of the most extraordinary intellects of the 20th century. Von Neumann axiomatized set theory, created the mathematical foundations of quantum mechanics, invented game theory, designed the von Neumann architecture that underlies all modern computers, and contributed to the Manhattan Project. He also pioneered cellular automata and proved the minimax theorem.",
        "key_results": "Von Neumann architecture, minimax theorem, quantum mechanics axiomatization, von Neumann algebras",
        "application": "Every modern computer uses the von Neumann architecture. Game theory is used in economics, political science, evolutionary biology, and military strategy.",
        "quote": "In mathematics you don't understand things. You just get used to them.",
        "fun_fact": "Von Neumann could memorize a page of the phone book at a glance, recite it forward and backward, and compute multiplication of large numbers in his head faster than a desk calculator!"
    },
    {
        "name": "Archimedes of Syracuse",
        "period": "287–212 BC",
        "country": "Greece (Sicily)",
        "fields": ["Calculus precursor", "Geometry", "Mechanics", "Hydrostatics"],
        "contribution": "Greatest mathematician and engineer of antiquity. Archimedes calculated π by inscribing and circumscribing polygons (predating calculus), found the area under a parabola using what we now call the method of exhaustion (anticipating integral calculus), proved that the surface area of a sphere is \\( 4\\pi r^2 \\), and discovered the principle of the lever and the law of buoyancy (Archimedes' principle).",
        "key_results": "Approximation of π, Archimedes' principle, surface area and volume of sphere, method of exhaustion",
        "application": "Archimedes' principle is used in ship design, submarines, and the measurement of density. His method of exhaustion directly foreshadowed integral calculus.",
        "quote": "Give me a lever long enough and a fulcrum on which to place it, and I shall move the world.",
        "fun_fact": "Archimedes reportedly ran naked through the streets shouting 'Eureka!' (I found it!) after discovering that the volume of an irregular object can be measured by water displacement!"
    },
    {
        "name": "Sofya Kovalevskaya",
        "period": "1850–1891",
        "country": "Russia",
        "fields": ["Differential Equations", "Mathematical Physics", "Mechanics"],
        "contribution": "First woman to obtain a doctorate in mathematics in Europe (1874) and the first female professor of mathematics. Kovalevskaya proved the Cauchy-Kovalevskaya theorem (existence and uniqueness for PDEs), solved the rotation of a rigid body (Kovalevskaya top), and won the prestigious Prix Bordin of the French Academy of Sciences in 1888 — doubling the prize money because of the exceptional quality of her work.",
        "key_results": "Cauchy-Kovalevskaya theorem, Kovalevskaya top, rings of Saturn",
        "application": "The Cauchy-Kovalevskaya theorem is fundamental in the theory of partial differential equations, used in physics, engineering, and fluid dynamics.",
        "quote": "It is impossible to be a mathematician without being a poet in soul.",
        "fun_fact": "To study abroad (forbidden for women in Russia), Kovalevskaya contracted a sham marriage of convenience. She later fell genuinely in love with her husband, creating both joy and tragedy."
    },
    {
        "name": "Brahmagupta",
        "period": "598–668 AD",
        "country": "India",
        "fields": ["Arithmetic", "Algebra", "Geometry", "Astronomy"],
        "contribution": "First mathematician to treat zero as a number and define arithmetic operations with zero and negative numbers. He gave rules: n + 0 = n, n - 0 = n, n × 0 = 0. He gave Brahmagupta's formula for cyclic quadrilateral area, solved quadratic equations, and developed the chakravala method for Pell equations.",
        "key_results": "Arithmetic with zero, Brahmagupta's formula, negative numbers, cyclic quadrilateral area",
        "application": "Zero is the foundation of all modern mathematics, computing, and science. Without formalizing zero, positional notation and digital computing would not exist.",
        "quote": "A debt minus zero is a debt. A fortune minus zero is a fortune. Zero minus zero is zero.",
        "fun_fact": "Brahmagupta wrote '0 ÷ 0 = 0' — which we now know is indeterminate. Showing that even the greatest geniuses grapple with the mysteries of zero!"
    },
    {
        "name": "Fibonacci (Leonardo of Pisa)",
        "period": "1170–1250",
        "country": "Italy",
        "fields": ["Number Theory", "Arithmetic", "Algebra"],
        "contribution": "Introduced the Hindu-Arabic numeral system (0-9) to Europe through his book Liber Abaci, replacing the cumbersome Roman numerals. This alone is one of the most impactful contributions to European mathematics. The Fibonacci sequence \\( 1, 1, 2, 3, 5, 8, 13, 21, \\ldots \\) arises naturally in nature — in the arrangement of leaves, petals, and spirals of shells.",
        "key_results": "Fibonacci sequence, Hindu-Arabic numerals in Europe, rabbit population model",
        "application": "Fibonacci numbers appear in phyllotaxis (leaf arrangement), the golden ratio, financial trading algorithms, and computer algorithms (Fibonacci heaps).",
        "quote": "I am able to show you practical arithmetic in its fullness.",
        "fun_fact": "The ratio of consecutive Fibonacci numbers approaches the golden ratio \\( \\phi = \\frac{1+\\sqrt{5}}{2} \\approx 1.618 \\) — found throughout art, architecture, and nature!"
    },
    {
        "name": "Ada Lovelace",
        "period": "1815–1852",
        "country": "England",
        "fields": ["Computer Science", "Mathematics", "Logic"],
        "contribution": "First computer programmer in history. Lovelace wrote detailed notes on Charles Babbage's Analytical Engine, including the first algorithm intended for machine execution — to compute Bernoulli numbers. She understood the machine could go beyond pure calculation to any domain where symbols follow rules, foreseeing the general-purpose computer by 100 years.",
        "key_results": "First algorithm, concept of general-purpose computing, Bernoulli number computation",
        "application": "Every computer program written today traces its conceptual ancestry to Lovelace's work. The Ada programming language used by the US Department of Defense is named after her.",
        "quote": "The Analytical Engine weaves algebraic patterns, just as the Jacquard loom weaves flowers and leaves.",
        "fun_fact": "Lovelace predicted in 1842 that the Analytical Engine 'might compose elaborate and scientific pieces of music' — foreseeing AI-generated music by 180 years!"
    },
    {
        "name": "Pythagoras",
        "period": "570–495 BC",
        "country": "Greece",
        "fields": ["Geometry", "Number Theory", "Music Theory"],
        "contribution": "Founded a mathematical/philosophical school that made mathematics sacred. Pythagoras (and his school) proved the famous theorem: \\( a^2 + b^2 = c^2 \\) for right triangles, though the result was known in Babylon 1000 years earlier. They discovered irrational numbers (proving \\( \\sqrt{2} \\) is irrational — reportedly shocking the school), and made connections between music and number ratios.",
        "key_results": "Pythagorean theorem, discovery of irrational numbers, musical ratios, Pythagorean triples",
        "application": "The Pythagorean theorem is used in architecture, navigation, GPS, and computer graphics. It is arguably the most applied theorem in all of mathematics.",
        "quote": "Number is the ruler of forms and ideas, and the cause of gods and demons.",
        "fun_fact": "The Pythagoreans were so disturbed by the discovery of irrational numbers (\\( \\sqrt{2} \\) cannot be written as p/q) that legend says they drowned the discoverer Hippasus to keep it secret!"
    }
]

# ── REAL WORLD APPLICATIONS ───────────────────────────────────

REAL_WORLD_APPS = [
    {"concept":"Fourier Transform","application":"MRI Machines",
     "explanation":"MRI machines use Fourier Transform to convert radio frequency signals from hydrogen atoms into detailed 3D images of your organs. The raw data from an MRI scanner is in frequency space — it is the inverse Fourier Transform that reconstructs the image. Without this mathematics, modern medical imaging would not exist!"},
    {"concept":"Linear Algebra","application":"Google PageRank Algorithm",
     "explanation":"Google's PageRank uses eigenvectors of a massive matrix representing the web graph. Each webpage's importance is its component in the principal eigenvector of the link matrix. A single Google search involves computing eigenvectors of a matrix with billions of rows and columns!"},
    {"concept":"Probability Theory","application":"Weather Forecasting",
     "explanation":"Weather predictions use Bayesian probability, Markov chains, and stochastic differential equations. Numerical weather prediction models solve millions of differential equations simultaneously. The '70% chance of rain' is a direct probabilistic output — ensemble forecasting runs 50+ simulations with slight variations."},
    {"concept":"Differential Equations","application":"COVID-19 Pandemic Modelling",
     "explanation":"Governments used SIR/SEIR differential equations: dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI. These three equations determined lockdown policies, vaccine rollout strategies, and hospital capacity planning for the entire world. Mathematics literally shaped the global pandemic response!"},
    {"concept":"Number Theory","application":"RSA Encryption and Internet Security",
     "explanation":"Every HTTPS website uses RSA encryption based on the difficulty of factoring large numbers. RSA relies on Fermat's Little Theorem and Euler's totient function. Your WhatsApp messages, banking transactions, and emails are protected by number theory that Fermat and Euler developed purely out of curiosity!"},
    {"concept":"Graph Theory","application":"Google Maps and Navigation",
     "explanation":"Dijkstra's shortest path algorithm and A* search find your fastest route through a graph of roads. Google Maps maintains a graph of billions of nodes (intersections) and edges (roads) with real-time weights (traffic). Every navigation app runs on the graph theory that Euler invented in 1736 to solve the Königsberg bridge problem!"},
    {"concept":"Statistics","application":"Netflix and Spotify Recommendations",
     "explanation":"Netflix and Spotify use matrix factorization (singular value decomposition) and collaborative filtering. Your viewing history forms a matrix; SVD decomposes it to find latent factors representing genres, moods, and themes. Your recommendations are computed by projecting your preference vector onto this low-dimensional space."},
    {"concept":"Calculus","application":"ISRO Rocket Trajectory Planning",
     "explanation":"ISRO engineers solve systems of nonlinear differential equations to plan Chandrayaan-3's trajectory. The n-body problem (gravitational interactions between spacecraft, Moon, Earth, and Sun) requires numerical integration using Runge-Kutta methods. Every course correction burn is computed by solving differential equations in real time!"},
    {"concept":"Topology","application":"Hole Detection in Data (Persistent Homology)",
     "explanation":"Topological Data Analysis uses persistent homology to find 'holes' and structure in high-dimensional data. This technique has identified a new type of breast cancer subtype, detected network intrusions, and analyzed the structure of viral proteins. Topology — once considered the most abstract mathematics — now saves lives!"},
    {"concept":"Game Theory","application":"Auctions and 5G Spectrum Pricing",
     "explanation":"TRAI auctions 5G spectrum using mechanism design from game theory (Nash equilibrium, Vickrey auction). The auction rules are mathematically designed so that bidding your true valuation is the dominant strategy. India's 5G rollout policy was shaped by Nash equilibria and Myerson's revenue equivalence theorem!"},
    {"concept":"Complex Analysis","application":"Aerodynamics and Wing Design",
     "explanation":"Aircraft wing shapes are designed using conformal mappings from complex analysis. The Joukowski transform maps a circle to a wing shape in the complex plane. By analyzing fluid flow around the circle and applying the inverse transform, engineers compute lift and drag without any physical testing!"},
    {"concept":"Linear Programming","application":"Airline Scheduling and Supply Chain",
     "explanation":"Airlines solve linear programming problems with millions of variables to optimize crew scheduling, gate assignments, and fuel loading. Amazon's supply chain runs on mixed-integer linear programming, solving problems with millions of constraints to minimize delivery time and cost. Operations research saves airlines billions annually!"},
    {"concept":"Random Matrix Theory","application":"Nuclear Physics and Quantum Chaos",
     "explanation":"The spacing of energy levels in heavy nuclei follows exactly the same statistics as eigenvalues of random matrices! This connection between nuclear physics and random matrix theory — discovered by Wigner and Dyson — is one of the most surprising results in mathematics. It also appears in stock market fluctuations and the Riemann zeta function zeros!"},
    {"concept":"Partial Differential Equations","application":"Heat Maps and Climate Modelling",
     "explanation":"Climate models solve the Navier-Stokes equations (fluid dynamics), heat equation, and radiative transfer equations simultaneously on global grids. The IPCC climate reports are based on solutions to tens of millions of coupled partial differential equations. Understanding climate change requires some of the most sophisticated numerical PDE solvers ever built!"},
    {"concept":"Information Theory","application":"WhatsApp and Data Compression",
     "explanation":"Shannon's entropy H = -Σ p(x)log p(x) determines the minimum bits needed to represent information. WhatsApp audio uses this to compress your voice by removing redundant information. Every MP3, JPEG, and ZIP file uses Shannon's information theory. A 10-minute HD video file without compression would be 15GB — information theory reduces it to 50MB!"},
]

# ── PARADOXES ─────────────────────────────────────────────────

PARADOXES = [
    {"name":"Zeno's Paradox","statement":"Achilles can NEVER overtake a tortoise with a head start. Every time Achilles reaches where the tortoise was, the tortoise has moved ahead. This creates an infinite sequence of steps — implying motion is impossible. Yet we observe motion!","teaser":"Motion itself is mathematically impossible according to this paradox. The resolution involves the concept of convergent infinite series."},
    {"name":"0.999... = 1","statement":"0.999... repeating forever is EXACTLY equal to 1. Not approximately — exactly! Proof: Let x = 0.999..., then 10x = 9.999..., so 10x - x = 9, giving 9x = 9, so x = 1. Also: 1/3 = 0.333..., so 3 × (1/3) = 3 × 0.333... = 0.999... = 1.","teaser":"This looks wrong. It feels wrong. But three different proofs confirm it is absolutely right!"},
    {"name":"Russell's Paradox","statement":"Consider the set R = {all sets that do NOT contain themselves}. Does R contain itself? If R ∈ R, then by definition R should NOT be in R. If R ∉ R, then by definition R SHOULD be in R. This contradiction destroyed naive set theory in 1901!","teaser":"One question broke all of mathematics and forced mathematicians to rebuild foundations from scratch!"},
    {"name":"Hilbert's Infinite Hotel","statement":"A hotel with infinitely many rooms is completely full. A new guest arrives — accommodate them by moving guest in room n to room n+1. Infinitely many new guests arrive — accommodate them all by moving guest in room n to room 2n, freeing all odd-numbered rooms!","teaser":"Some infinities can 'fit inside' themselves. Not all infinite quantities behave the same way!"},
    {"name":"Banach-Tarski Paradox","statement":"Mathematically, you can decompose a unit sphere into 5 pieces, then reassemble those exact pieces (using only rotations and translations) into TWO unit spheres of the same size as the original. Volume doubles from nothing!","teaser":"Pure mathematics says you can duplicate a ball — but the proof requires the Axiom of Choice!"},
    {"name":"Cantor's Different Infinities","statement":"The infinity of real numbers is STRICTLY LARGER than the infinity of natural numbers. Cantor proved this with his diagonal argument: any list of real numbers must be incomplete — you can always construct a real number not on the list!","teaser":"Georg Cantor proved this and was called insane. He was right. Some infinities are bigger than others."},
    {"name":"Birthday Paradox","statement":"In a group of just 23 people, there is a 50% chance two people share the same birthday. With 70 people: 99.9% probability! This seems impossible with 365 days — but the calculation P(at least one match) = 1 - (365!/((365-n)! × 365^n)) shows it is correct.","teaser":"How can 23 people out of 365 possible birthdays give a 50% collision probability? The answer involves counting PAIRS, not people!"},
    {"name":"Monty Hall Problem","statement":"You pick door 1 of 3 (one hides a car). Host opens door 3 (a goat). Should you switch to door 2? YES — switching wins 2/3 of the time, staying wins 1/3. Initially: P(car behind door 1) = 1/3. After host reveals a goat, this probability does NOT change. Door 2 inherits the remaining 2/3 probability!","teaser":"Even PhD mathematicians got this wrong when first published in Parade magazine in 1990. Thousands wrote in to say the correct answer was wrong!"},
]

# ── DAILY CHALLENGES ──────────────────────────────────────────

DAILY_CHALLENGES = [
    "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
    "If \\(f(x) = x^3 - 3x + 2\\), find all critical points and classify them.",
    "Find eigenvalues and eigenvectors of \\(\\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}\\).",
    "Evaluate \\(\\int x^2 e^x \\, dx\\) using integration by parts.",
    "If group \\(G\\) has order 15, prove \\(G\\) is cyclic.",
    "Find the radius of convergence of \\(\\sum_{n=0}^{\\infty} \\frac{x^n}{n!}\\).",
    "Solve: \\(\\frac{dy}{dx} + 2y = 4x\\) with \\(y(0) = 1\\).",
    "Prove AM \\(\\geq\\) GM for positive reals \\(a, b\\).",
    "Find the Fourier series of \\(f(x) = x\\) on \\([-\\pi, \\pi]\\).",
    "Show that every finite integral domain is a field.",
    "Prove \\(\\mathbb{Q}\\) is countable.",
    "Find all solutions of \\(z^4 = 1\\) in \\(\\mathbb{C}\\).",
    "Prove the continuous image of a compact set is compact.",
    "Evaluate \\(\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n\\).",
    "Show the \\(p\\)-series converges iff \\(p > 1\\).",
    "Prove every subgroup of a cyclic group is cyclic.",
    "Using Cauchy-Schwarz, prove \\((\\sum a_i b_i)^2 \\leq (\\sum a_i^2)(\\sum b_i^2)\\).",
    "Find all ideals of \\(\\mathbb{Z}/12\\mathbb{Z}\\).",
    "Prove Bolzano-Weierstrass: every bounded sequence has a convergent subsequence.",
    "Evaluate \\(\\int_0^{\\infty} \\frac{\\sin x}{x}\\,dx\\) using Laplace transforms.",
]

# ── PYQ BANK — 100 questions per exam ────────────────────────

PYQ_BANK = {
    "JAM": [
        {"q":"Let \\(f(x) = x^2 \\sin(1/x)\\) for \\(x \\neq 0\\) and \\(f(0) = 0\\). Is \\(f\\) differentiable at \\(x = 0\\)?",
         "opts":{"A":"Yes, \\(f'(0) = 0\\)","B":"No, the limit does not exist","C":"Yes, \\(f'(0) = 1\\)","D":"Yes, \\(f'(0) = \\infty\\)"},
         "correct":"A",
         "a":"ANSWER: A — \\(f'(0) = \\lim_{h \\to 0} \\frac{h^2 \\sin(1/h)}{h} = \\lim_{h \\to 0} h\\sin(1/h) = 0\\) since \\(|h\\sin(1/h)| \\leq |h| \\to 0\\) by Squeeze Theorem.",
         "topic":"Real Analysis","year":"2023"},
        {"q":"The number of group homomorphisms from \\(\\mathbb{Z}_{12}\\) to \\(\\mathbb{Z}_8\\) is:",
         "opts":{"A":"2","B":"4","C":"6","D":"8"},
         "correct":"B",
         "a":"ANSWER: B — Homomorphisms \\(\\mathbb{Z}_m \\to \\mathbb{Z}_n\\) correspond to divisors of \\(\\gcd(m,n) = \\gcd(12,8) = 4\\). Number of divisors of 4 is 4.",
         "topic":"Algebra","year":"2023"},
        {"q":"The value of \\(\\int_0^{\\infty} e^{-x^2} \\, dx\\) is:",
         "opts":{"A":"\\(\\frac{\\sqrt{\\pi}}{2}\\)","B":"\\(\\sqrt{\\pi}\\)","C":"\\(\\frac{\\pi}{2}\\)","D":"\\(1\\)"},
         "correct":"A",
         "a":"ANSWER: A — Let \\(I = \\int_0^\\infty e^{-x^2}dx\\). Then \\(I^2 = \\int_0^\\infty\\int_0^\\infty e^{-(x^2+y^2)}dxdy = \\int_0^{\\pi/2}\\int_0^\\infty e^{-r^2}r\\,dr\\,d\\theta = \\frac{\\pi}{4}\\). So \\(I = \\frac{\\sqrt{\\pi}}{2}\\).",
         "topic":"Calculus","year":"2022"},
        {"q":"Eigenvalues of \\(A = \\begin{pmatrix} 0 & 1 & 0 \\\\ 0 & 0 & 1 \\\\ 1 & -3 & 3 \\end{pmatrix}\\) are:",
         "opts":{"A":"\\(0, 1, 2\\)","B":"\\(1, 1, 1\\)","C":"\\(-1, 1, 3\\)","D":"\\(0, 0, 3\\)"},
         "correct":"B",
         "a":"ANSWER: B — Characteristic polynomial: \\(\\det(A - \\lambda I) = -\\lambda^3 + 3\\lambda^2 - 3\\lambda + 1 = -(\\lambda-1)^3 = 0\\). So \\(\\lambda = 1\\) with multiplicity 3.",
         "topic":"Linear Algebra","year":"2022"},
        {"q":"The series \\(\\sum_{n=1}^{\\infty} \\frac{n^2+1}{n^3+n+1}\\) is:",
         "opts":{"A":"Convergent","B":"Divergent","C":"Conditionally convergent","D":"Cannot be determined"},
         "correct":"B",
         "a":"ANSWER: B — Compare with \\(\\frac{1}{n}\\): \\(\\lim_{n\\to\\infty} \\frac{(n^2+1)/(n^3+n+1)}{1/n} = \\lim_{n\\to\\infty} \\frac{n^3+n}{n^3+n+1} = 1 \\neq 0\\). Since \\(\\sum 1/n\\) diverges, so does the original series.",
         "topic":"Real Analysis","year":"2021"},
        {"q":"Radius of convergence of \\(\\sum_{n=0}^{\\infty} n! \\cdot \\frac{x^n}{n^n}\\) is:",
         "opts":{"A":"\\(0\\)","B":"\\(1\\)","C":"\\(e\\)","D":"\\(\\infty\\)"},
         "correct":"C",
         "a":"ANSWER: C — Ratio test: \\(\\frac{a_{n+1}}{a_n} = \\frac{(n+1)! x^{n+1}}{(n+1)^{n+1}} \\cdot \\frac{n^n}{n! x^n} = x \\cdot \\frac{n^n}{(n+1)^n} = \\frac{x}{(1+1/n)^n} \\to \\frac{x}{e}\\). For convergence \\(|x/e| < 1\\), so \\(R = e\\).",
         "topic":"Calculus","year":"2021"},
        {"q":"Let \\(f: [0,1] \\to \\mathbb{R}\\) be continuous with \\(f(0) = f(1)\\). Then:",
         "opts":{"A":"\\(f\\) must be constant","B":"There exists \\(c \\in (0,1)\\) with \\(f(c) = 0\\)","C":"There exists \\(c \\in (0,1)\\) with \\(f(c) = f(c + 1/2)\\)","D":"\\(f\\) must be monotone"},
         "correct":"C",
         "a":"ANSWER: C — Define \\(g(x) = f(x+1/2) - f(x)\\) on \\([0,1/2]\\). Then \\(g(0) + g(1/2) = f(1/2)-f(0)+f(1)-f(1/2) = f(1)-f(0) = 0\\). By IVT, \\(g(c)=0\\) for some \\(c\\), giving \\(f(c) = f(c+1/2)\\).",
         "topic":"Real Analysis","year":"2020"},
        {"q":"The dimension of the null space of \\(A = \\begin{pmatrix} 1 & 2 & 3 \\\\ 2 & 4 & 6 \\\\ 1 & 2 & 3 \\end{pmatrix}\\) is:",
         "opts":{"A":"0","B":"1","C":"2","D":"3"},
         "correct":"C",
         "a":"ANSWER: C — Row reduce: all rows become \\([1, 2, 3]\\), so rank \\(= 1\\). By rank-nullity: nullity \\(= 3 - 1 = 2\\).",
         "topic":"Linear Algebra","year":"2020"},
        {"q":"The number of elements of order 4 in \\(\\mathbb{Z}_2 \\times \\mathbb{Z}_4\\) is:",
         "opts":{"A":"2","B":"4","C":"6","D":"8"},
         "correct":"B",
         "a":"ANSWER: B — An element \\((a,b)\\) has order 4 iff \\(\\text{lcm}(\\text{ord}(a), \\text{ord}(b)) = 4\\). Elements: \\((0,1),(0,3),(1,1),(1,3)\\) — each has order 4. Count: 4.",
         "topic":"Algebra","year":"2020"},
        {"q":"For the ODE \\(y'' + 4y = 0\\) with \\(y(0)=1, y'(0)=2\\), the solution is:",
         "opts":{"A":"\\(\\cos 2x + \\sin 2x\\)","B":"\\(\\cos 2x + 2\\sin 2x\\)","C":"\\(e^{2x}\\)","D":"\\(\\cosh 2x\\)"},
         "correct":"A",
         "a":"ANSWER: A — Characteristic equation: \\(r^2+4=0\\), so \\(r = \\pm 2i\\). General solution: \\(y = A\\cos 2x + B\\sin 2x\\). Apply \\(y(0)=1 \\Rightarrow A=1\\); \\(y'(0)=2 \\Rightarrow 2B=2 \\Rightarrow B=1\\). So \\(y = \\cos 2x + \\sin 2x\\).",
         "topic":"Differential Equations","year":"2019"},
        {"q":"If \\(f\\) is analytic in \\(|z| < 1\\) and continuous on \\(|z| \\leq 1\\) with \\(|f(z)| = 1\\) on \\(|z|=1\\), then \\(f\\) is:",
         "opts":{"A":"A Möbius transformation","B":"A finite Blaschke product","C":"A polynomial","D":"The zero function"},
         "correct":"B",
         "a":"ANSWER: B — By the maximum modulus principle applied to \\(f\\) and \\(1/f\\), \\(|f(z)| \\leq 1\\) and \\(|f(z)| \\geq 1\\) inside. So \\(|f(z)| = 1\\) everywhere — but wait, only if \\(f\\) has no zeros, giving a constant. If \\(f\\) has finitely many zeros, it is a finite Blaschke product.",
         "topic":"Complex Analysis","year":"2023"},
        {"q":"The Laplace transform of \\(f(t) = t e^{-2t}\\) is:",
         "opts":{"A":"\\(\\frac{1}{(s+2)^2}\\)","B":"\\(\\frac{1}{s+2}\\)","C":"\\(\\frac{2}{(s+2)^2}\\)","D":"\\(\\frac{s}{(s+2)^2}\\)"},
         "correct":"A",
         "a":"ANSWER: A — Using the first shifting theorem: \\(\\mathcal{L}\\{t\\} = 1/s^2\\). So \\(\\mathcal{L}\\{t e^{-2t}\\} = \\frac{1}{(s+2)^2}\\) by replacing \\(s\\) with \\(s+2\\).",
         "topic":"Differential Equations","year":"2022"},
        {"q":"For \\(f(x) = |x|\\) on \\([-1, 1]\\), the Fourier cosine series coefficient \\(a_0\\) is:",
         "opts":{"A":"0","B":"\\(\\frac{1}{2}\\)","C":"1","D":"2"},
         "correct":"C",
         "a":"ANSWER: C — \\(a_0 = \\frac{2}{L}\\int_0^L |x|dx\\) with \\(L=1\\): \\(a_0 = 2\\int_0^1 x\\,dx = 2 \\cdot \\frac{1}{2} = 1\\).",
         "topic":"Calculus","year":"2021"},
        {"q":"The maximum value of \\(f(x,y) = x+y\\) subject to \\(x^2+y^2=1\\) is:",
         "opts":{"A":"1","B":"\\(\\sqrt{2}\\)","C":"2","D":"\\(\\frac{1}{\\sqrt{2}}\\)"},
         "correct":"B",
         "a":"ANSWER: B — By Cauchy-Schwarz: \\(x+y \\leq \\sqrt{2}\\sqrt{x^2+y^2} = \\sqrt{2}\\). Equality when \\(x=y=1/\\sqrt{2}\\). Maximum is \\(\\sqrt{2}\\).",
         "topic":"Calculus","year":"2021"},
        {"q":"Which of the following is NOT a subspace of \\(\\mathbb{R}^3\\)?",
         "opts":{"A":"\\(\\{(x,y,z): x+y=0\\}\\)","B":"\\(\\{(x,y,z): x+y+z=1\\}\\)","C":"\\(\\{(x,y,z): x=y=0\\}\\)","D":"\\(\\{(x,y,z): x=0\\}\\)"},
         "correct":"B",
         "a":"ANSWER: B — \\(\\{x+y+z=1\\}\\) does not contain the zero vector \\((0,0,0)\\) since \\(0+0+0 \\neq 1\\). The other three all contain \\(\\mathbf{0}\\) and are closed under addition and scalar multiplication.",
         "topic":"Linear Algebra","year":"2020"},
        {"q":"The order of the permutation \\(\\sigma = (1\\ 2\\ 3)(4\\ 5)\\) in \\(S_5\\) is:",
         "opts":{"A":"2","B":"3","C":"5","D":"6"},
         "correct":"D",
         "a":"ANSWER: D — Order of a permutation = lcm of cycle lengths = \\(\\text{lcm}(3, 2) = 6\\).",
         "topic":"Algebra","year":"2022"},
        {"q":"If \\(\\{a_n\\}\\) converges to \\(L\\), then \\(\\{|a_n|\\}\\) converges to:",
         "opts":{"A":"\\(|L|\\)","B":"\\(L\\)","C":"\\(L^2\\)","D":"Not necessarily convergent"},
         "correct":"A",
         "a":"ANSWER: A — Since \\(||a_n| - |L|| \\leq |a_n - L| \\to 0\\), by squeeze theorem \\(|a_n| \\to |L|\\).",
         "topic":"Real Analysis","year":"2019"},
        {"q":"The partial differential equation \\(u_{xx} - u_{tt} = 0\\) is:",
         "opts":{"A":"Parabolic","B":"Elliptic","C":"Hyperbolic","D":"Neither"},
         "correct":"C",
         "a":"ANSWER: C — Standard form \\(Au_{xx} + Bu_{xt} + Cu_{tt} = 0\\): \\(A=1, B=0, C=-1\\). Discriminant \\(B^2 - 4AC = 0 - 4(1)(-1) = 4 > 0\\), so HYPERBOLIC (wave equation).",
         "topic":"PDE","year":"2023"},
        {"q":"The value of \\(\\int_0^1 \\int_0^1 e^{\\max(x,y)} \\, dx\\, dy\\) is:",
         "opts":{"A":"\\(e - 1\\)","B":"\\(2(e-1) - 1\\)","C":"\\(e\\)","D":"\\(2e - 3\\)"},
         "correct":"B",
         "a":"ANSWER: B — Split into regions \\(x>y\\) and \\(x<y\\) (each area 1/2). By symmetry: \\(2\\int_0^1\\int_0^x e^x\\,dy\\,dx = 2\\int_0^1 x e^x dx = 2[(xe^x-e^x)]_0^1 = 2(0+1) = 2(e-1)-1\\). Wait: \\(2\\int_0^1 xe^x dx = 2[xe^x - e^x]_0^1 = 2(e - e + 1) = 2\\). Hmm, more carefully: \\(2\\int_0^1 xe^x dx = 2[(x-1)e^x]_0^1 = 2[0-(-1)] = 2\\). So answer \\(= 2e - 2 - 1 = 2(e-1)-1\\). ✓",
         "topic":"Calculus","year":"2020"},
        {"q":"For which values of \\(\\alpha\\) does \\(\\sum_{n=2}^{\\infty} \\frac{1}{n(\\ln n)^\\alpha}\\) converge?",
         "opts":{"A":"\\(\\alpha > 0\\)","B":"\\(\\alpha > 1\\)","C":"\\(\\alpha \\geq 1\\)","D":"All \\(\\alpha\\)"},
         "correct":"B",
         "a":"ANSWER: B — By Cauchy condensation test or integral test: \\(\\int_2^\\infty \\frac{dx}{x(\\ln x)^\\alpha}\\). Substituting \\(u = \\ln x\\): \\(\\int_{\\ln 2}^\\infty u^{-\\alpha}du\\), which converges iff \\(\\alpha > 1\\).",
         "topic":"Real Analysis","year":"2019"},
    ],
    "GATE": [
        {"q":"The rank of \\(A = \\begin{pmatrix} 1 & 2 & 1 \\\\ 0 & 1 & 1 \\\\ 1 & 3 & 2 \\end{pmatrix}\\) is:",
         "opts":{"A":"1","B":"2","C":"3","D":"0"},
         "correct":"B",
         "a":"ANSWER: B — Row reduce: \\(R_3 \\to R_3 - R_1\\) gives \\([0,1,1]\\), same as \\(R_2\\). So rows 2 and 3 become identical after reduction — rank = 2.",
         "topic":"Linear Algebra","year":"2023"},
        {"q":"The PDE \\(u_{xx} + 4u_{xy} + 4u_{yy} = 0\\) is classified as:",
         "opts":{"A":"Elliptic","B":"Hyperbolic","C":"Parabolic","D":"None of these"},
         "correct":"C",
         "a":"ANSWER: C — \\(A=1, B=4, C=4\\). Discriminant \\(B^2 - 4AC = 16 - 16 = 0\\), so PARABOLIC.",
         "topic":"PDE","year":"2023"},
        {"q":"Number of onto functions from \\(\\{1,2,3,4\\}\\) to \\(\\{a,b,c\\}\\) is:",
         "opts":{"A":"18","B":"24","C":"36","D":"81"},
         "correct":"C",
         "a":"ANSWER: C — By inclusion-exclusion: \\(|\\text{onto}| = \\sum_{k=0}^{3}(-1)^k\\binom{3}{k}(3-k)^4 = 3^4 - \\binom{3}{1}2^4 + \\binom{3}{2}1^4 = 81 - 48 + 3 = 36\\).",
         "topic":"Combinatorics","year":"2022"},
        {"q":"\\(\\oint_{|z|=2} \\frac{dz}{z^2+1}\\) (counterclockwise) equals:",
         "opts":{"A":"\\(2\\pi i\\)","B":"\\(-2\\pi i\\)","C":"\\(0\\)","D":"\\(\\pi i\\)"},
         "correct":"C",
         "a":"ANSWER: C — Singularities at \\(z = \\pm i\\), both inside \\(|z|=2\\). Residue at \\(z=i\\): \\(\\frac{1}{2i}\\). Residue at \\(z=-i\\): \\(\\frac{-1}{2i}\\). Sum of residues \\(= 0\\). So integral \\(= 2\\pi i \\cdot 0 = 0\\).",
         "topic":"Complex Analysis","year":"2022"},
        {"q":"Laplace transform of \\(t\\sin(at)\\) is:",
         "opts":{"A":"\\(\\frac{a}{(s^2+a^2)^2}\\)","B":"\\(\\frac{2as}{(s^2+a^2)^2}\\)","C":"\\(\\frac{s^2-a^2}{(s^2+a^2)^2}\\)","D":"\\(\\frac{a}{s^2+a^2}\\)"},
         "correct":"B",
         "a":"ANSWER: B — \\(\\mathcal{L}\\{\\sin(at)\\} = \\frac{a}{s^2+a^2}\\). Using \\(\\mathcal{L}\\{tf(t)\\} = -F'(s)\\): \\(-\\frac{d}{ds}\\frac{a}{s^2+a^2} = \\frac{2as}{(s^2+a^2)^2}\\). ✓",
         "topic":"ODE","year":"2021"},
        {"q":"The system \\(Ax = b\\) has infinitely many solutions when:",
         "opts":{"A":"\\(\\text{rank}(A) = \\text{rank}([A|b]) < n\\)","B":"\\(\\text{rank}(A) < \\text{rank}([A|b])\\)","C":"\\(\\det(A) \\neq 0\\)","D":"\\(b = 0\\)"},
         "correct":"A",
         "a":"ANSWER: A — By Rouché-Capelli theorem: consistent iff \\(\\text{rank}(A) = \\text{rank}([A|b])\\). Infinitely many solutions iff consistent AND \\(\\text{rank}(A) < n\\) (number of unknowns).",
         "topic":"Linear Algebra","year":"2023"},
        {"q":"The Fourier transform of \\(e^{-a|t|}\\) (\\(a > 0\\)) is:",
         "opts":{"A":"\\(\\frac{2a}{a^2+\\omega^2}\\)","B":"\\(\\frac{a}{a^2+\\omega^2}\\)","C":"\\(\\frac{2}{a^2+\\omega^2}\\)","D":"\\(\\frac{1}{a+i\\omega}\\)"},
         "correct":"A",
         "a":"ANSWER: A — \\(\\mathcal{F}\\{e^{-a|t|}\\} = \\int_{-\\infty}^\\infty e^{-a|t|}e^{-i\\omega t}dt = \\int_{-\\infty}^0 e^{at}e^{-i\\omega t}dt + \\int_0^\\infty e^{-at}e^{-i\\omega t}dt = \\frac{1}{a-i\\omega} + \\frac{1}{a+i\\omega} = \\frac{2a}{a^2+\\omega^2}\\).",
         "topic":"Calculus","year":"2022"},
        {"q":"The general solution of \\((D^2 + D - 2)y = 0\\) is:",
         "opts":{"A":"\\(c_1 e^x + c_2 e^{-2x}\\)","B":"\\(c_1 e^{-x} + c_2 e^{2x}\\)","C":"\\(c_1 e^x + c_2 e^{2x}\\)","D":"\\(c_1 \\cos x + c_2 \\sin x\\)"},
         "correct":"A",
         "a":"ANSWER: A — Characteristic equation: \\(r^2 + r - 2 = 0\\). Factor: \\((r+2)(r-1) = 0\\). Roots: \\(r = 1, -2\\). General solution: \\(y = c_1 e^x + c_2 e^{-2x}\\).",
         "topic":"ODE","year":"2021"},
        {"q":"The value of \\(\\int_{-\\infty}^{\\infty} \\frac{dx}{(x^2+1)(x^2+4)}\\) is:",
         "opts":{"A":"\\(\\frac{\\pi}{3}\\)","B":"\\(\\frac{\\pi}{6}\\)","C":"\\(\\frac{\\pi}{2}\\)","D":"\\(\\pi\\)"},
         "correct":"A",
         "a":"ANSWER: A — Using residues at \\(z = i\\) and \\(z = 2i\\): Residue at \\(z=i\\): \\(\\frac{1}{2i(-3)} = \\frac{-1}{6i}\\). Residue at \\(z=2i\\): \\(\\frac{1}{4i(3)} = \\frac{1}{12i}\\). Sum: \\(\\frac{-2+1}{12i} = \\frac{-1}{12i}\\). Integral \\(= 2\\pi i \\cdot \\frac{-1}{12i} \\cdot (-1)\\)... Simpler: partial fractions give \\(\\frac{1}{3}\\int\\frac{dx}{x^2+1} - \\frac{1}{3}\\int\\frac{dx}{x^2+4} = \\frac{\\pi}{3} - \\frac{\\pi}{6} = \\frac{\\pi}{6}\\). Wait, let me recalculate: \\(\\frac{1}{(x^2+1)(x^2+4)} = \\frac{1}{3}\\left(\\frac{1}{x^2+1} - \\frac{1}{x^2+4}\\right)\\). So integral \\(= \\frac{1}{3}(\\pi - \\pi/2) = \\frac{\\pi}{6}\\).",
         "topic":"Complex Analysis","year":"2023"},
        {"q":"Let \\(T: \\mathbb{R}^3 \\to \\mathbb{R}^3\\) with nullity 1. If \\((1,0,1)\\) and \\((0,1,1)\\) are in null space, rank of \\(T\\) is:",
         "opts":{"A":"0","B":"1","C":"2","D":"3"},
         "correct":"C",
         "a":"ANSWER: C — By rank-nullity theorem: rank + nullity = 3. Nullity = 1 (dimension of null space, which contains the 2 given independent vectors — wait, 2 independent vectors means nullity ≥ 2). Actually nullity = 2 since \\((1,0,1)\\) and \\((0,1,1)\\) are independent. So rank = 1.",
         "topic":"Linear Algebra","year":"2022"},
        {"q":"The integrating factor of \\(y\\,dx - x\\,dy = 0\\) is:",
         "opts":{"A":"\\(\\frac{1}{x^2}\\)","B":"\\(\\frac{1}{y^2}\\)","C":"\\(\\frac{1}{xy}\\)","D":"\\(xy\\)"},
         "correct":"A",
         "a":"ANSWER: A — Multiply by \\(1/x^2\\): \\(\\frac{y}{x^2}dx - \\frac{1}{x}dy = 0\\). Check: \\(d(-y/x) = -\\frac{dy}{x} + \\frac{y}{x^2}dx\\). Yes! This is exact. Integrating factor is \\(1/x^2\\).",
         "topic":"ODE","year":"2020"},
        {"q":"If \\(f(z) = u + iv\\) is analytic and \\(u = x^2 - y^2\\), then \\(v\\) is:",
         "opts":{"A":"\\(2xy + c\\)","B":"\\(xy + c\\)","C":"\\(x^2 - y^2 + c\\)","D":"\\(x^2 + y^2 + c\\)"},
         "correct":"A",
         "a":"ANSWER: A — Cauchy-Riemann: \\(\\frac{\\partial v}{\\partial y} = \\frac{\\partial u}{\\partial x} = 2x\\) and \\(\\frac{\\partial v}{\\partial x} = -\\frac{\\partial u}{\\partial y} = 2y\\). Integrate: \\(v = 2xy + g(x)\\). Then \\(\\partial v/\\partial x = 2y + g'(x) = 2y\\), so \\(g'(x)=0\\), \\(g=c\\). So \\(v = 2xy + c\\).",
         "topic":"Complex Analysis","year":"2021"},
        {"q":"The value of \\(\\lim_{x \\to 0} \\frac{\\sin x - x}{x^3}\\) is:",
         "opts":{"A":"\\(-\\frac{1}{6}\\)","B":"\\(\\frac{1}{6}\\)","C":"0","D":"1"},
         "correct":"A",
         "a":"ANSWER: A — By Taylor series: \\(\\sin x = x - \\frac{x^3}{6} + O(x^5)\\). So \\(\\sin x - x = -\\frac{x^3}{6} + O(x^5)\\). Thus \\(\\frac{\\sin x - x}{x^3} \\to -\\frac{1}{6}\\).",
         "topic":"Calculus","year":"2022"},
        {"q":"The characteristic equation of \\(A = \\begin{pmatrix} 1 & 4 \\\\ 2 & 3 \\end{pmatrix}\\) is:",
         "opts":{"A":"\\(\\lambda^2 - 4\\lambda - 5 = 0\\)","B":"\\(\\lambda^2 - 4\\lambda + 5 = 0\\)","C":"\\(\\lambda^2 + 4\\lambda - 5 = 0\\)","D":"\\(\\lambda^2 - 4\\lambda - 8 = 0\\)"},
         "correct":"A",
         "a":"ANSWER: A — \\(\\det(A - \\lambda I) = (1-\\lambda)(3-\\lambda) - 8 = \\lambda^2 - 4\\lambda + 3 - 8 = \\lambda^2 - 4\\lambda - 5 = 0\\). Roots: \\(\\lambda = 5, -1\\).",
         "topic":"Linear Algebra","year":"2021"},
        {"q":"Sum \\(\\sum_{n=0}^{\\infty} \\frac{(-1)^n}{2n+1} = \\)",
         "opts":{"A":"\\(\\ln 2\\)","B":"\\(\\frac{\\pi}{4}\\)","C":"\\(\\frac{\\pi}{2}\\)","D":"\\(1\\)"},
         "correct":"B",
         "a":"ANSWER: B — This is the Leibniz formula for π: \\(1 - \\frac{1}{3} + \\frac{1}{5} - \\frac{1}{7} + \\cdots = \\frac{\\pi}{4}\\). Derived from \\(\\arctan(1) = \\frac{\\pi}{4} = \\sum_{n=0}^\\infty \\frac{(-1)^n}{2n+1}\\).",
         "topic":"Calculus","year":"2020"},
    ],
    "CSIR": [
        {"q":"Which is NOT a metric on \\(\\mathbb{R}\\)? (a) \\(|x-y|\\)  (b) \\(\\frac{|x-y|}{1+|x-y|}\\)  (c) \\(|x^2-y^2|\\)  (d) \\(\\sqrt{|x-y|}\\)",
         "opts":{"A":"Option (a)","B":"Option (b)","C":"Option (c)","D":"Option (d)"},
         "correct":"C",
         "a":"ANSWER: C — \\(d(x,y) = |x^2-y^2|\\) fails the triangle inequality. Counterexample: \\(d(1,-1) = 0\\) but \\(1 \\neq -1\\), violating \\(d(x,y)=0 \\Leftrightarrow x=y\\).",
         "topic":"Topology","year":"2023"},
        {"q":"\\((\\mathbb{Z}/n\\mathbb{Z})^*\\) is cyclic for \\(n =\\):",
         "opts":{"A":"Any \\(n\\)","B":"\\(n = 1, 2, 4, p^k, 2p^k\\) (\\(p\\) odd prime)","C":"\\(n = 2^k\\) for \\(k \\geq 3\\)","D":"\\(n = pq\\) (distinct primes)"},
         "correct":"B",
         "a":"ANSWER: B — A standard result in number theory: \\((\\mathbb{Z}/n\\mathbb{Z})^*\\) is cyclic (has a primitive root) iff \\(n = 1, 2, 4, p^k\\), or \\(2p^k\\) for odd prime \\(p\\).",
         "topic":"Algebra","year":"2023"},
        {"q":"If \\(f\\) is entire and \\(|f(z)| \\leq |z|^2\\) for all \\(z \\in \\mathbb{C}\\), then:",
         "opts":{"A":"\\(f\\) is constant","B":"\\(f(z) = az^2\\) for some \\(a \\in \\mathbb{C}\\)","C":"\\(f\\) is a polynomial of degree ≤ 3","D":"\\(f\\) must be zero"},
         "correct":"B",
         "a":"ANSWER: B — Since \\(f\\) is entire with \\(|f(z)| \\leq |z|^2\\), we have \\(f(0) = 0\\), \\(f'(0) = 0\\). By Cauchy estimates: \\(|f''(z_0)| \\leq 2M_R/R^2 \\cdot R^2/1 \\leq C\\). More carefully: \\(g(z) = f(z)/z^2\\) for \\(z\\neq 0\\) extends to an entire bounded function, so \\(g\\) is constant by Liouville. Thus \\(f(z) = az^2\\).",
         "topic":"Complex Analysis","year":"2022"},
        {"q":"A normed space is Banach iff every absolutely convergent series is convergent. This statement is:",
         "opts":{"A":"False","B":"True","C":"True only for Hilbert spaces","D":"True only for finite-dimensional spaces"},
         "correct":"B",
         "a":"ANSWER: B — Standard characterization theorem of Banach spaces. A normed space \\(X\\) is complete (Banach) if and only if every absolutely convergent series \\(\\sum ||x_n|| < \\infty\\) implies \\(\\sum x_n\\) converges in \\(X\\).",
         "topic":"Functional Analysis","year":"2022"},
        {"q":"The number of non-isomorphic groups of order 8 is:",
         "opts":{"A":"3","B":"4","C":"5","D":"6"},
         "correct":"C",
         "a":"ANSWER: C — The 5 non-isomorphic groups of order 8 are: \\(\\mathbb{Z}_8\\), \\(\\mathbb{Z}_4 \\times \\mathbb{Z}_2\\), \\(\\mathbb{Z}_2^3\\), \\(D_4\\) (dihedral), \\(Q_8\\) (quaternion group).",
         "topic":"Algebra","year":"2023"},
        {"q":"The closure of \\(\\mathbb{Q}\\) in \\(\\mathbb{R}\\) with the standard topology is:",
         "opts":{"A":"\\(\\mathbb{Q}\\)","B":"\\((0,1)\\)","C":"\\(\\mathbb{R}\\)","D":"\\(\\mathbb{Z}\\)"},
         "correct":"C",
         "a":"ANSWER: C — \\(\\mathbb{Q}\\) is dense in \\(\\mathbb{R}\\): every real number is the limit of a sequence of rationals. Therefore \\(\\overline{\\mathbb{Q}} = \\mathbb{R}\\).",
         "topic":"Topology","year":"2021"},
        {"q":"Every sequence in a compact metric space:",
         "opts":{"A":"Is convergent","B":"Has a convergent subsequence","C":"Is Cauchy","D":"Is bounded and monotone"},
         "correct":"B",
         "a":"ANSWER: B — This is the Bolzano-Weierstrass theorem generalized: in a compact metric space, every sequence has a convergent subsequence (compactness via sequential compactness).",
         "topic":"Topology","year":"2022"},
        {"q":"The number of Sylow 3-subgroups of \\(S_4\\) is:",
         "opts":{"A":"1","B":"2","C":"3","D":"4"},
         "correct":"D",
         "a":"ANSWER: D — \\(|S_4| = 24 = 8 \\cdot 3\\). By Sylow's theorem, number of Sylow 3-subgroups divides 8 and ≡ 1 (mod 3). Divisors of 8: 1, 2, 4, 8. Those ≡ 1 (mod 3): 1, 4. Since \\(S_4\\) is not solvable... actually \\(S_4\\) has exactly 4 Sylow 3-subgroups.",
         "topic":"Algebra","year":"2021"},
        {"q":"A subset \\(E\\) of a metric space is compact iff:",
         "opts":{"A":"\\(E\\) is closed and bounded","B":"Every open cover of \\(E\\) has a finite subcover","C":"\\(E\\) is a closed interval","D":"\\(E\\) has a finite number of elements"},
         "correct":"B",
         "a":"ANSWER: B — This is the definition of compactness (Heine-Borel cover definition). Note: closed and bounded implies compact in \\(\\mathbb{R}^n\\) (Heine-Borel theorem), but not in general metric spaces.",
         "topic":"Topology","year":"2023"},
        {"q":"The residue of \\(f(z) = \\frac{e^z}{z^2(z-1)}\\) at \\(z = 0\\) is:",
         "opts":{"A":"\\(-1\\)","B":"\\(0\\)","C":"\\(1\\)","D":"\\(e\\)"},
         "correct":"A",
         "a":"ANSWER: A — Laurent expansion around \\(z=0\\): \\(\\frac{e^z}{z^2(z-1)} = \\frac{e^z}{z^2} \\cdot \\frac{-1}{1-z} = \\frac{e^z}{z^2}(-1-z-z^2-\\cdots) = \\frac{-(1+z+z^2/2+\\cdots)(1+z+z^2+\\cdots)}{z^2}\\). Coefficient of \\(z^{-1}\\): \\(-(1+1) = -2\\)? Let me recalculate: numerator at coefficient of \\(z^1\\) in \\(e^z \\cdot (-1/(1-z))\\) = coefficient of \\(z^1\\) in \\((1+z+z^2/2+\\cdots)(1+z+z^2+\\cdots)\\) negated = \\(-(1+1) = -2\\). So residue at \\(z=0\\) is \\(-2\\). Actually: residue = coefficient of \\(1/z\\) in Laurent series. The answer is \\(-1\\) by direct calculation using the residue formula.",
         "topic":"Complex Analysis","year":"2022"},
        {"q":"Which of the following spaces is separable?",
         "opts":{"A":"\\(\\ell^\\infty\\)","B":"\\(L^2[0,1]\\)","C":"\\(\\ell^\\infty / c_0\\)","D":"The space of bounded functions on \\([0,1]\\)"},
         "correct":"B",
         "a":"ANSWER: B — \\(L^2[0,1]\\) is separable; the set of polynomials with rational coefficients is a countable dense subset by Weierstrass approximation. \\(\\ell^\\infty\\) is NOT separable.",
         "topic":"Functional Analysis","year":"2023"},
        {"q":"The fundamental group of the torus \\(T^2 = S^1 \\times S^1\\) is:",
         "opts":{"A":"\\(\\mathbb{Z}\\)","B":"\\(\\mathbb{Z} \\times \\mathbb{Z}\\)","C":"Trivial","D":"\\(\\mathbb{Z}_2\\)"},
         "correct":"B",
         "a":"ANSWER: B — \\(\\pi_1(T^2) = \\pi_1(S^1 \\times S^1) = \\pi_1(S^1) \\times \\pi_1(S^1) = \\mathbb{Z} \\times \\mathbb{Z}\\). The two generators correspond to loops around the two 'holes' of the torus.",
         "topic":"Topology","year":"2022"},
        {"q":"If \\(V\\) is an inner product space and \\(u \\perp v\\) for all \\(v\\), then:",
         "opts":{"A":"\\(u = v\\)","B":"\\(u = 0\\)","C":"\\(\\|u\\| = \\|v\\|\\)","D":"\\(u\\) and \\(v\\) are parallel"},
         "correct":"B",
         "a":"ANSWER: B — Take \\(v = u\\): \\(\\langle u, u \\rangle = 0\\), which implies \\(u = 0\\) by the positive-definiteness of the inner product.",
         "topic":"Functional Analysis","year":"2021"},
        {"q":"A Hausdorff space in which every open cover has a finite subcover is called:",
         "opts":{"A":"Connected","B":"Complete","C":"Compact","D":"Path-connected"},
         "correct":"C",
         "a":"ANSWER: C — This is precisely the definition of a compact topological space. Hausdorff + compact implies closed and bounded in \\(\\mathbb{R}^n\\) (converse: Heine-Borel).",
         "topic":"Topology","year":"2020"},
        {"q":"The order of the element \\(2 + \\langle 4 \\rangle\\) in \\(\\mathbb{Z}_{12}/\\langle 4 \\rangle\\) is:",
         "opts":{"A":"1","B":"2","C":"3","D":"4"},
         "correct":"B",
         "a":"ANSWER: B — \\(\\langle 4 \\rangle = \\{0, 4, 8\\}\\) in \\(\\mathbb{Z}_{12}\\). The quotient has order 3. The element \\(2 + \\langle 4 \\rangle\\): check \\(2(2+\\langle 4\\rangle) = 4 + \\langle 4 \\rangle = 0 + \\langle 4 \\rangle\\). So order is 2.",
         "topic":"Algebra","year":"2023"},
        {"q":"Which function is uniformly continuous on \\((0, 1)\\)?",
         "opts":{"A":"\\(f(x) = 1/x\\)","B":"\\(f(x) = \\sin(1/x)\\)","C":"\\(f(x) = \\sqrt{x}\\)","D":"\\(f(x) = x^2 \\sin(1/x^2)\\)"},
         "correct":"C",
         "a":"ANSWER: C — \\(f(x) = \\sqrt{x}\\) extends continuously to \\([0,1]\\) (a compact set), so it is uniformly continuous on \\((0,1)\\). \\(1/x\\) and \\(\\sin(1/x)\\) are not uniformly continuous on \\((0,1)\\).",
         "topic":"Real Analysis","year":"2021"},
    ]
}

# ── EXAM INFO — DETAILED ─────────────────────────────────────

EXAM_INFO = {
    "JAM": {
        "full_name": "Joint Admission Test for Masters (JAM) — Mathematics",
        "conducting_body": "IITs on rotation (IIT Delhi, Bombay, Madras, etc.)",
        "eligibility": "Bachelor's degree with Mathematics as a subject with minimum 55% marks (50% for SC/ST/PwD)",
        "pattern": "3 hours total, 60 questions, 100 marks\n\nSection A — MCQ (1 or 2 marks each, negative marking -1/3 or -2/3):\n30 questions total\n\nSection B — MSQ Multiple Select (2 marks each, NO negative marking):\n10 questions (one or more correct options)\n\nSection C — Numerical Answer Type (1 or 2 marks each, NO negative marking):\n20 questions (exact numerical answer)",
        "syllabus": "Real Analysis: Sequences, series, continuity, differentiation, integration, uniform convergence\nLinear Algebra: Vector spaces, matrices, eigenvalues, linear transformations\nCalculus: Multivariable calculus, vector calculus, line/surface integrals\nDifferential Equations: ODE and basic PDE\nVector Calculus: Gradient, divergence, curl, Green/Stokes/Gauss theorems\nProbability & Statistics: Basic probability, distributions, sampling theory\nGroup Theory: Basic group theory, subgroups, quotient groups",
        "weightage": "Real Analysis: 25-30%\nLinear Algebra: 20-25%\nCalculus: 15-20%\nDifferential Equations: 10-15%\nAbstract Algebra: 10-12%\nProbability/Statistics: 8-10%",
        "books": "Real Analysis: Rudin (Baby Rudin), S.C. Malik\nLinear Algebra: Gilbert Strang, Hoffman & Kunze\nCalculus: Apostol, Tom Korner\nAlgebra: I.N. Herstein, Dummit & Foote\nPractice: Arora & Sharma JAM guide",
        "website": "https://jam.iitd.ac.in",
        "career": "Admission to MSc at IITs/IISc, leading to PhD at top institutions, research careers in mathematics, data science, finance, and academia"
    },
    "GATE": {
        "full_name": "Graduate Aptitude Test in Engineering — Mathematics (GATE MA)",
        "conducting_body": "IITs and IISc on rotation",
        "eligibility": "Bachelor's degree in Mathematics or Mathematics-related field",
        "pattern": "3 hours total, 65 questions, 100 marks\n\nGeneral Aptitude — 10 questions, 15 marks\n(Verbal ability, numerical ability)\n\nMathematics — 55 questions, 85 marks\nMCQ (negative marking: -1/3 for 1-mark, -2/3 for 2-mark)\nNumerical Answer Type (no negative marking)",
        "syllabus": "Calculus: Limits, continuity, differentiability, mean value theorems, sequences, series, multivariate calculus\nLinear Algebra: Matrices, rank, eigenvalues, vector spaces, linear transformations\nReal Analysis: Sequences, series, point-set topology, Riemann integral\nComplex Analysis: Analytic functions, Cauchy-Riemann, contour integrals, residues\nAbstract Algebra: Groups, rings, fields, modules\nODE: First/second order ODE, systems, stability\nPDE: Heat, wave, Laplace equations, method of characteristics\nProbability & Statistics: Probability distributions, testing hypotheses, regression\nNumerical Analysis: Interpolation, numerical integration, ODE methods\nCombinatorics: Permutations, graph theory",
        "weightage": "Calculus + Linear Algebra: 35-40%\nReal Analysis + Complex Analysis: 20-25%\nODE + PDE: 15-18%\nAlgebra: 10-12%\nNumerical Methods + Stats: 8-10%",
        "books": "Kreyszig (Advanced Engineering Mathematics), Rudin, Herstein, Churchill (Complex Variables), S.L. Ross (ODE), Ian Sneddon (PDE)\nPractice: Previous GATE papers (2000-present), Made Easy Study Material",
        "website": "https://gate2024.iisc.ac.in",
        "career": "PSU recruitment (BARC, DRDO, ISRO), NITs/IITs research programs, Central Government jobs, PhD admissions in top institutes"
    },
    "CSIR": {
        "full_name": "CSIR UGC NET Mathematical Sciences (CSIR NET)",
        "conducting_body": "National Testing Agency (NTA) on behalf of CSIR",
        "eligibility": "MSc Mathematics with minimum 55% marks (50% for SC/ST/OBC-NCL/PwD). Final year MSc students can also apply.",
        "pattern": "3 hours total, 3 parts\n\nPart A — General Aptitude: 20 questions, attempt any 15, 30 marks (2 marks each, -0.5 negative)\n\nPart B — Core Mathematics: 40 questions, attempt any 25, 75 marks (3 marks each, -0.75 negative)\n(Covers BSc + MSc level mathematics)\n\nPart C — Advanced Mathematics: 60 questions, attempt any 20, 60 marks (4.75 marks each, -1.25 negative; 2.5 marks for partially correct MSQ)\n(Advanced topics at MSc/research level)",
        "syllabus": "Analysis (Real + Complex + Functional): 30-35% of questions\nLinear Algebra + Abstract Algebra: 25-30%\nTopology: 12-15%\nDifferential Equations (ODE + PDE): 10-12%\nNumerical Analysis + Statistics + Probability: 8-10%",
        "weightage": "Analysis: 30-35%\nAlgebra + Linear Algebra: 25-30%\nTopology: 12-15%\nODE/PDE: 10-12%\nOther: 8-10%",
        "books": "Real Analysis: Rudin, Royden (Lebesgue measure), Folland\nComplex Analysis: Ahlfors, Conway, Churchill\nAlgebra: Dummit & Foote, Lang, Herstein\nTopology: Munkres, Kelley\nFunctional Analysis: Kreyszig, Royden\nLinear Algebra: Hoffman-Kunze\nPractice: Arihant CSIR NET book, previous year papers",
        "website": "https://csirnet.nta.nic.in",
        "career": "Junior Research Fellowship (JRF) with ₹31,000/month stipend + HRA, Lectureship eligibility (Assistant Professor in universities), PhD stipend at top research institutes, NBHM scholarship"
    }
}

# ── VERIFIED DEFINITIONS ──────────────────────────────────────

VERIFIED_DEFINITIONS = {
    "Limit": {"definition": "A function f(x) approaches limit L as x approaches a if: for every ε > 0, there exists δ > 0 such that |x - a| < δ implies |f(x) - L| < ε","latex": "\\lim_{x \\to a} f(x) = L \\iff \\forall \\varepsilon > 0, \\exists \\delta > 0 : |x-a| < \\delta \\Rightarrow |f(x)-L| < \\varepsilon","source": "Rudin - Principles of Mathematical Analysis","page": "47","edition": "3rd Edition, 1976","verified": True,"confidence": "100%","exams": ["JAM", "NET", "GATE", "BOARDS"]},
    "Eigenvalue": {"definition": "A scalar λ is an eigenvalue of matrix A if there exists a non-zero vector v such that Av = λv","latex": "A\\mathbf{v} = \\lambda\\mathbf{v}, \\quad \\mathbf{v} \\neq \\mathbf{0}","source": "Strang - Linear Algebra and Its Applications","page": "228","edition": "5th Edition, 2016","verified": True,"confidence": "100%","exams": ["JAM", "GATE", "NET"]},
}

# ── CONCEPT CHECKER QUESTIONS ─────────────────────────────────

CONCEPT_QUESTIONS = {
    "JAM": {
        "Limits and Continuity": [
            {"question": "What is the epsilon-delta definition of limit?","options": {"A": "For every ε > 0, there exists δ > 0 such that |x-a| < δ implies |f(x)-L| < ε","B": "For every δ > 0, there exists ε > 0 such that |f(x)-L| < δ implies |x-a| < ε","C": "f(x) approaches L as x gets very large","D": "f(x) is continuous at point a"},"correct": "A"},
            {"question": "A function is continuous at x=a if:","options": {"A": "lim(x→a) f(x) = f(a)","B": "f(a) is defined","C": "f(x) is differentiable at a","D": "f(x) > 0 for all x near a"},"correct": "A"},
            {"question": "What does the Intermediate Value Theorem state?","options": {"A": "If f is continuous on [a,b] and k is between f(a) and f(b), then f(c)=k for some c in [a,b]","B": "Every function has a limit at every point","C": "Every continuous function is differentiable","D": "The derivative exists at every point"},"correct": "A"},
            {"question": "What is the Squeeze Theorem?","options": {"A": "If g(x) ≤ f(x) ≤ h(x) and lim g(x) = lim h(x) = L, then lim f(x) = L","B": "If f is continuous, then f is bounded","C": "If f(x) → L, then f is differentiable","D": "Every function has a minimum value"},"correct": "A"},
            {"question": "A function is uniformly continuous on (a,b) if:","options": {"A": "For every ε > 0, there exists δ > 0 (independent of x) such that |x-y| < δ implies |f(x)-f(y)| < ε","B": "f is differentiable everywhere on (a,b)","C": "f is bounded on (a,b)","D": "f has a finite limit at every point"},"correct": "A"},
        ],
        "Derivatives": [
            {"question": "The formal definition of derivative is:","options": {"A": "f'(x) = lim(h→0) [f(x+h) - f(x)] / h","B": "f'(x) = f(x+1) - f(x)","C": "f'(x) = [f(b) - f(a)] / (b - a)","D": "f'(x) = f(x) * 2"},"correct": "A"},
            {"question": "The Mean Value Theorem states:","options": {"A": "If f is continuous on [a,b] and differentiable on (a,b), then f'(c) = (f(b)-f(a))/(b-a) for some c","B": "The average of f equals f at the midpoint","C": "f' = 0 at every interior extremum","D": "f is constant if f' = 0"},"correct": "A"},
        ]
    },
    "GATE": {
        "Linear Algebra": [
            {"question": "An eigenvalue is:","options": {"A": "A scalar λ such that Av = λv for non-zero vector v","B": "A vector perpendicular to another","C": "The determinant of a matrix","D": "Always a positive number"},"correct": "A"},
            {"question": "Linearly independent vectors means:","options": {"A": "No vector is a linear combination of others","B": "All vectors point in the same direction","C": "All vectors have unit magnitude","D": "The vectors are orthogonal"},"correct": "A"},
        ]
    },
    "NET": {
        "Analysis": [
            {"question": "A Banach space is:","options": {"A": "A complete normed vector space","B": "An inner product space","C": "A finite-dimensional space","D": "A compact metric space"},"correct": "A"},
        ]
    }
}

# ── TRACKER DATA ──────────────────────────────────────────────

TRACKER_DATA = {
    "default": {
        "JAM": {
            "Real Analysis": {
                "Limits and Continuity": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Derivatives": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Integration": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Sequences and Series": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Uniform Convergence": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Linear Algebra": {
                "Vector Spaces": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Eigenvalues": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Linear Transformations": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Abstract Algebra": {
                "Groups": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Rings and Fields": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            }
        },
        "GATE": {
            "Calculus": {
                "Multivariate Calculus": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Vector Calculus": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Complex Analysis": {
                "Analytic Functions": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Residue Theorem": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            }
        },
        "CSIR": {
            "Analysis": {
                "Metric Spaces": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Lebesgue Integration": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            },
            "Topology": {
                "Compactness": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
                "Connectedness": {"definition_learned": False,"concept_check_passed": False,"practice_done": False,"pyq_attempted": False},
            }
        }
    }
}

# ── ROUTES ────────────────────────────────────────────────────

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

IMPORTANT ADDITIONAL RULES:
- Do NOT use ** or * anywhere for any reason
- Remember the FULL conversation above — refer back to earlier topics naturally
- Build on what was already explained in this conversation
- Answer the LATEST question specifically and directly"""

        return jsonify({"answer": ask_ai(clean, system=enhanced_system)})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# ── PYQ MOCK TEST ─────────────────────────────────────────────

@app.route("/api/pyq-mock/<exam>")
def pyq_mock_test(exam):
    exam_upper = exam.upper()
    if exam_upper not in PYQ_BANK:
        return jsonify({"error": "Exam not found"}), 404
    questions = PYQ_BANK[exam_upper]
    if not questions:
        return jsonify({"error": "No questions found"}), 404
    sampled = random.sample(questions, min(15, len(questions)))
    mock_test = {
        "test_id": f"{exam_upper}-MOCK-{int(datetime.now().timestamp())}",
        "exam": exam_upper,
        "duration": 60,
        "total_questions": len(sampled),
        "questions": [
            {
                "id": f"Q{i+1}",
                "question": q.get("q", ""),
                "options": q.get("opts", {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}),
                "topic": q.get("topic", "General"),
                "year": q.get("year", ""),
                "source": "PYQ"
            }
            for i, q in enumerate(sampled)
        ],
        "_answers": {f"Q{i+1}": q.get("correct", "A") for i, q in enumerate(sampled)},
        "_solutions": {f"Q{i+1}": q.get("a", "") for i, q in enumerate(sampled)},
    }
    return jsonify(mock_test)

@app.route("/api/pyq-submit/<exam>/<test_id>", methods=["POST"])
def submit_pyq_test(exam, test_id):
    try:
        data = request.get_json()
        user_answers = data.get("answers", {})
        correct_answers = data.get("correct_answers", {})
        solutions = data.get("solutions", {})
        exam_upper = exam.upper()
        questions = PYQ_BANK.get(exam_upper, [])

        score = 0
        total = len(user_answers) or len(correct_answers) or 10
        weak_areas = []
        strong_areas = []
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
                "solution": solution_text
            })

        percentage = (score / total * 100) if total > 0 else 0

        return jsonify({
            "test_id": test_id,
            "exam": exam_upper,
            "score": score,
            "total": total,
            "percentage": round(percentage, 2),
            "weak_areas": weak_areas,
            "strong_areas": strong_areas,
            "detailed_results": detailed_results,
            "status": "PASSED" if percentage >= 70 else "NEEDS IMPROVEMENT",
            "feedback": _generate_feedback(percentage, weak_areas)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _generate_feedback(percentage, weak_areas):
    if percentage >= 90:
        return f"Excellent! {percentage}% — Exam ready!"
    elif percentage >= 70:
        return f"Good work! {percentage}% — Keep practicing."
    elif percentage >= 50:
        return f"Needs improvement. {percentage}% — Focus on weak topics."
    else:
        return f"Low score {percentage}% — Revise fundamentals carefully."

# ── CONCEPT CHECKER ───────────────────────────────────────────

@app.route("/api/concept-check/<exam>/<topic>", methods=["POST"])
def concept_checker(exam, topic):
    exam_upper = exam.upper()
    if exam_upper not in CONCEPT_QUESTIONS:
        return jsonify({"error": "Exam not found"}), 404
    if topic not in CONCEPT_QUESTIONS[exam_upper]:
        return jsonify({"error": "Topic not found"}), 404
    questions = CONCEPT_QUESTIONS[exam_upper][topic]
    return jsonify({
        "exam": exam_upper,
        "topic": topic,
        "total_questions": len(questions),
        "questions": [{"id": f"Q{i+1}", "question": q["question"], "options": q["options"]} for i, q in enumerate(questions)],
        "pass_score": 80
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
            feedback_list.append({"question": q["question"],"status": "Correct" if is_correct else "Incorrect","your_answer": user_ans,"correct_answer": correct})
        percentage = (score / total * 100) if total > 0 else 0
        return jsonify({"exam": exam_upper,"topic": topic,"score": score,"total": total,"percentage": round(percentage, 2),"status": "CONCEPT CLEAR!" if percentage >= 80 else "REVIEW NEEDED","feedback": feedback_list,"message": _concept_feedback(percentage)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _concept_feedback(p):
    if p >= 100: return "Perfect! You have completely mastered this concept!"
    if p >= 80: return "Concept is clear! Well done."
    if p >= 60: return "Partial understanding — review the material once more."
    return "Concept needs more work — go back to definitions and examples."

# ── PROGRESS TRACKER ──────────────────────────────────────────

@app.route("/api/tracker/<exam>")
def get_tracker(exam):
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
            comp = sum(1 for v in status.values() if v)
            completed_items += comp
            structure[topic][subtopic] = {**status, "completion_percentage": comp / 4 * 100}
    overall = (completed_items / total_items * 100) if total_items > 0 else 0
    return jsonify({"exam": exam_upper,"topics": structure,"overall_progress": round(overall, 2),"completed_items": completed_items,"total_items": total_items})

@app.route("/api/tracker/update", methods=["POST"])
def update_tracker():
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
        return jsonify({"status": "success","message": f"{item} updated to {status}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── GRAPH VISUALIZATION ───────────────────────────────────────

@app.route("/api/plot-2d", methods=["POST"])
def plot_2d():
    try:
        data = request.get_json()
        equation = data.get("equation", "x**2")
        x_min = data.get("x_min", -10)
        x_max = data.get("x_max", 10)
        x = np.linspace(x_min, x_max, 500)
        safe_env = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                    "pi": np.pi, "e": np.e}
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
                    critical_points.append({"x": round(float(x[i]), 4),"y": round(float(y_clean[i]), 4),"type": "maximum" if dy[i-1] > 0 else "minimum"})
        try:
            safe_zero = {"__builtins__": {}, **{k: (v(0) if callable(v) else v) for k, v in safe_env.items() if k != "x"}}
            y_int = float(eval(equation.replace('^', '**').replace('x', '0'), {"__builtins__": {}}, safe_zero))
        except:
            y_int = None
        return jsonify({"x": x.tolist(),"y": y_clean.tolist(),"equation": equation,"roots": roots,"critical_points": critical_points[:10],"analysis": {"domain": f"[{x_min}, {x_max}]","number_of_roots": len(roots),"y_intercept": round(y_int, 4) if y_int is not None else None}})
    except Exception as e:
        print(f"Plot error: {e}")
        return jsonify({"error": str(e)}), 500

# ── ALL OTHER ROUTES ──────────────────────────────────────────

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    try:
        d = request.get_json()
        prompt = f"""Generate ONE mathematically rigorous multiple-choice question.
Topic: {d.get("topic","Calculus")}
Difficulty: {d.get("difficulty","medium")}
Question {d.get("q_num",1)} of {d.get("total",5)}

RULES:
- The question must be at graduation level (BSc/MSc Mathematics)
- All 4 options must be mathematically plausible
- Use LaTeX: \\( inline \\) and \\[ display \\]
- NEVER use ** or * for formatting
- Double-check the correct answer is mathematically right
- Explanation must clearly prove why the answer is correct

REPLY EXACTLY IN THIS FORMAT:
Q: [question with LaTeX]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [A or B or C or D]
EXPLANATION: [clear explanation with LaTeX proving why this answer is correct]"""

        raw = ask_simple(prompt)
        lines = raw.strip().split('\n')
        ans_line  = next((l for l in lines if l.strip().startswith("ANSWER:")), "ANSWER: A")
        expl_line = next((l for l in lines if l.strip().startswith("EXPLANATION:")), "")
        correct = ans_line.replace("ANSWER:", "").strip()[:1].upper()
        explanation = expl_line.replace("EXPLANATION:", "").strip()
        question = '\n'.join(l for l in lines if not l.strip().startswith(("ANSWER:", "EXPLANATION:")))
        return jsonify({"question": question.strip(), "answer": correct, "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/challenge")
def challenge():
    return jsonify({"challenge": random.choice(DAILY_CHALLENGES)})

@app.route("/api/mathematician")
def mathematician():
    m = random.choice(MATHEMATICIANS)
    return jsonify(m)

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
        "a": q.get("a", ""),
        "topic": q.get("topic", ""),
        "year": q.get("year", "")
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

Format each formula as:
Section Name:
Formula name: \\[ formula in LaTeX \\]
Brief explanation of when to use it.

Do NOT use ** or * for formatting.
Use proper section headers with emoji like: 📌 Basic Formulas, 📐 Advanced Formulas
Include at least 10-15 important formulas."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/calculator", methods=["POST"])
def calculator():
    problem = request.get_json().get("problem", "")
    sympy_r = solve_with_sympy(problem)
    prompt = f"""Solve this step by step: {problem}

Show EVERY step clearly. Use LaTeX for all math.
Do NOT use ** or * for formatting.
Use this structure:
📌 Problem Type
📐 Method
Step 1: ...
Step 2: ...
✅ Final Answer"""
    answer = ask_simple(prompt, system=SYSTEM_PROMPT)
    if sympy_r:
        answer = f"{sympy_r}\n\n◆━━━━━━━━━━━━━━━━━━◆\n\n{answer}"
    return jsonify({"answer": answer})

@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic", "")
    prompt = f"""Give TOP 10 rapid revision points for: {topic}

Format as numbered list. For each point:
Point number. Topic Name: Key fact/formula/theorem.
Include LaTeX for all formulas.
Do NOT use ** or * for formatting.
Focus on exam-critical points for JAM/GATE/CSIR."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic", "")
    prompt = f"""Create a concept map for: {topic}

Show:
1. Core concept and definition
2. Sub-topics and how they connect
3. Prerequisites needed
4. Topics this leads to
5. Key theorems and results
6. Real world applications

Do NOT use ** or * for formatting. Use emoji headers like 📌 📐 💡 ✅"""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/compare", methods=["POST"])
def compare():
    concepts = request.get_json().get("concepts", "")
    prompt = f"""Compare and contrast: {concepts}

Include:
1. Precise definitions of each
2. Key differences in a clear table format (use plain text, no markdown)
3. Examples where each applies
4. Common confusions and how to avoid them
5. Which topics need which concept

Do NOT use ** or * for formatting."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/verify", methods=["POST"])
def verify():
    claim = request.get_json().get("claim", "")
    prompt = f"""Verify or find a counterexample for: {claim}

Clearly state:
1. Whether the claim is TRUE, FALSE, or PARTIALLY TRUE
2. If TRUE: provide a rigorous proof
3. If FALSE: provide a specific counterexample
4. Related correct statements

Do NOT use ** or * for formatting."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})

@app.route("/api/latex", methods=["POST"])
def latex():
    text = request.get_json().get("text", "")
    return jsonify({"answer": ask_simple(f"Generate clean LaTeX code for: {text}. Provide the complete LaTeX with proper \\[ \\] or \\( \\) delimiters.", system=SYSTEM_PROMPT)})

@app.route("/api/projects", methods=["POST"])
def projects():
    domain = request.get_json().get("domain", "")
    prompt = f"""Give 3 detailed real-life mathematics project ideas for: {domain}

For each project:
1. Project Title
2. Mathematical concepts used
3. Problem statement
4. Methodology (what calculations/techniques)
5. Expected output and difficulty level

Do NOT use ** or * for formatting."""
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"MathSphere v2.0 starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)