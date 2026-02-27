"""
MathSphere Web v5.0 FINAL — Complete Ultimate Edition
======================================================
✅ 16 Mathematicians with full details + research links
✅ 20 Math Projects with companies + salary ranges
✅ Interactive Theorem Explorer (6 theorems with proofs)
✅ Competition Problems: IMO, Putnam, AIME
✅ Learning Paths: JAM → GATE → CSIR → PhD
✅ Research Hub: 5 categories × 8 topics
✅ Exam-specific Formula Sheets (15-20 formulas)
✅ Concept Maps, Revision, LaTeX Generator
✅ Quiz (5 questions) + Mock Test (30 questions)
✅ PYQ Bank, Daily Challenge
✅ Real-world apps with company names + salaries
✅ No asterisks anywhere
✅ Mobile-friendly

REMOVED: Step Calculator, Compare Concepts, Verify My Claim,
         Proof Builder, Math Debate, Paradoxes

By Anupam Nigam | youtube.com/@pi_nomenal1729
"""

import os, re, json, random, base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GROQ_AVAILABLE  = bool(GROQ_API_KEY)
GEMINI_AVAILABLE= bool(GEMINI_API_KEY)

groq_client   = None
gemini_client = None

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

GROQ_MODELS      = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TEACHER_YOUTUBE  = "https://youtube.com/@pi_nomenal1729"
TEACHER_INSTAGRAM= "https://instagram.com/pi_nomenal1729"
TEACHER_WEBSITE  = "https://www.anupamnigam.com"


# ═══════════════════════════════════════════════════
# ASTERISK REMOVER — runs on every AI response
# ═══════════════════════════════════════════════════

def clean_response(text: str) -> str:
    if not text: return text
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{3}(.+?)_{3}',   r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{2}(.+?)_{2}',   r'\1', text, flags=re.DOTALL)

    # protect LaTeX before stripping stray asterisks
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


# ═══════════════════════════════════════════════════
# AI CORE
# ═══════════════════════════════════════════════════

def ask_ai(messages, system=None):
    if GROQ_AVAILABLE:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        if len(full) > 15: full = [full[0]] + full[-13:]
        for model in GROQ_MODELS:
            try:
                r = groq_client.chat.completions.create(
                    model=model, messages=full, max_tokens=4000, temperature=0.3)
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
        fallback.append({"role": "user", "content": "User uploaded an image. If exact details are not visible, mention assumptions and solve step by step."})
    return ask_ai(fallback, system=system)
# ═══════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are MathSphere — expert Mathematics teacher for graduate students, created by Anupam Nigam.

ABSOLUTE RULES — NO EXCEPTIONS:
1. NEVER use asterisks * or ** anywhere, for any reason
2. Use CAPS for emphasis: IMPORTANT, CRITICAL, NOTE
3. Emoji headers: 📌 Topic  📐 Solution  💡 Insight  ✅ Answer  🔍 Deep Dive
4. ALL math in LaTeX: \\(inline\\)  or  \\[display\\]
5. HTML tags allowed: <br> <hr>

FORMAT EVERY RESPONSE:
📌 [Topic Name]
💡 Real-life Application: [1 sentence]
📖 Definition with LaTeX
📐 Key steps / concepts with LaTeX
✅ Final boxed answer
📚 {TEACHER_YOUTUBE}

TONE: Warm Hinglish — "Dekho...", "Samajh aaya?", "Bohot achha!"
ALWAYS verify calculations twice."""
ASK_ANUPAM_PROMPT = f"""You are Ask Anupam — an all-purpose AI tutor by Anupam Nigam.

MISSION:
- Answer ANY user question: mathematics, science, coding, writing, reasoning, productivity, or general knowledge
- If an image is uploaded, analyse it carefully and solve/explain it STEP BY STEP
- For unclear images, state what is visible, ask 1 concise clarification, then continue with best effort

STYLE RULES:
1. NEVER use asterisks symbols in output
2. Keep answers concise by default: short explanation + direct result
3. Expand details ONLY when user asks follow-up or says explain deeply
4. For math/proof/solve tasks, always use: Step 1, Step 2, Step 3... and then Final Answer
5. Keep textual narration brief, but keep mathematical expressions complete and correct (LaTeX where needed)
6. Use conversation memory from previous messages to keep context consistent

TONE: Friendly, clear, and confident.
"""


# ═══════════════════════════════════════════════════
# MATHEMATICIANS — 16 complete entries
# ═══════════════════════════════════════════════════

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
        "keyresults": "Euler identity e^(iπ)+1=0, Euler formula V-E+F=2, Basel problem ∑1/n²=π²/6, Königsberg bridges",
        "quote": "Mathematics is the queen of sciences",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d7/Leonhard_Euler.jpg",
        "impact": "Internet networking, electrical engineering, quantum mechanics, every branch of modern math",
        "resources": ["Euler Archive: https://scholarlycommons.pacific.edu/euler/"]
    },
    "Carl Friedrich Gauss": {
        "period": "1777–1855", "country": "Germany",
        "fields": ["Number Theory", "Statistics", "Differential Geometry", "Algebra"],
        "contribution": "Prince of Mathematics. Proved Fundamental Theorem of Algebra at age 21. Invented least squares, Gaussian distribution, modular arithmetic.",
        "keyresults": "FTA, bell curve N(μ,σ²), Gauss-Bonnet theorem, quadratic reciprocity, prime number estimates",
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
        "contribution": "Invented calculus, discovered gravity, formulated three laws of motion. Arguably the greatest scientist who ever lived.",
        "keyresults": "Calculus, F=ma, universal gravitation F=Gm₁m₂/r², binomial theorem, Principia Mathematica",
        "quote": "If I have seen further, it is by standing on the shoulders of giants",
        "image": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Principia_Mathematica_1687.jpg",
        "impact": "All classical mechanics, aerospace, civil engineering, space exploration, $100T+ economy",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Newton.html"]
    },
    "Gottfried Leibniz": {
        "period": "1646–1716", "country": "Germany",
        "fields": ["Calculus", "Logic", "Philosophy", "Combinatorics"],
        "contribution": "Co-invented calculus independently. Invented ∫ integral notation, d/dx derivative notation, and ∞ symbol. Universal genius.",
        "keyresults": "Calculus notation (∫, d/dx), Leibniz rule, binary number system, symbolic logic foundations",
        "quote": "There are no wholly useless truths",
        "image": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Gottfried_Wilhelm_Leibniz%2C_Bernhard_Christoph_Francke.jpg",
        "impact": "All mathematics notation, computer science (binary), programming foundations, $10T+ industry",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Leibniz.html"]
    },
    "Augustin-Louis Cauchy": {
        "period": "1789–1857", "country": "France",
        "fields": ["Real Analysis", "Complex Analysis", "Mathematical Rigour"],
        "contribution": "Brought rigour to calculus. Established epsilon-delta definitions. Cauchy sequences, Cauchy integral formula, residue theorem.",
        "keyresults": "Cauchy sequences, Cauchy integral theorem, ε-δ definitions, Cauchy-Schwarz inequality, residue theorem",
        "quote": "I prefer the man of genius who laboreth without ceasing to perfect his works",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/d8/Augustin-Louis_Cauchy_1901.jpg",
        "impact": "Foundation of all modern rigorous mathematics and proof standards worldwide",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Cauchy.html"]
    },
    "David Hilbert": {
        "period": "1862–1943", "country": "Germany",
        "fields": ["Functional Analysis", "Mathematical Logic", "Abstract Algebra"],
        "contribution": "Led formalism movement. His 23 unsolved problems shaped all of 20th-century mathematics. Invented Hilbert spaces.",
        "keyresults": "Hilbert spaces, Hilbert's 23 problems (1900), formalism, metamathematics, spectral theory",
        "quote": "We must know, we will know",
        "image": "https://upload.wikimedia.org/wikipedia/commons/d/da/Hilbert.jpg",
        "impact": "Quantum mechanics foundation, optimization, AI/ML mathematical basis, all functional analysis",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Hilbert.html"]
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
    "Georg Cantor": {
        "period": "1845–1918", "country": "Germany",
        "fields": ["Set Theory", "Mathematical Logic", "Foundations of Mathematics"],
        "contribution": "Invented set theory. Proved there are different sizes of infinity. Called insane by contemporaries — vindicated by history.",
        "keyresults": "Set theory, transfinite cardinals ℵ₀ ℵ₁, diagonal argument, continuum hypothesis, power sets",
        "quote": "The true infinite, the truly infinite, is the Deity",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Georg_Cantor2.jpg",
        "impact": "Foundations of all mathematics, computer science theory, philosophy of infinity, database theory",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Cantor.html"]
    },
    "Henri Poincaré": {
        "period": "1854–1912", "country": "France",
        "fields": ["Topology", "Dynamical Systems", "Celestial Mechanics"],
        "contribution": "Last universal mathematician who mastered all areas. Founded topology, chaos theory. Contributions to special relativity.",
        "keyresults": "Poincaré conjecture (solved 2003), algebraic topology, chaos theory, homology groups, fundamental group",
        "quote": "Mathematics is the art of giving the same name to different things",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Poincare1.jpg",
        "impact": "General relativity, weather prediction (chaos), $1M Millennium Prize, GPS systems",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Poincare.html"]
    },
    "Terence Tao": {
        "period": "1975–present", "country": "Australia",
        "fields": ["Number Theory", "Harmonic Analysis", "PDE", "Combinatorics"],
        "contribution": "Mozart of mathematics. Fields Medal 2006 at age 31. Solved Green-Tao theorem on primes in arithmetic progressions.",
        "keyresults": "Green-Tao theorem, compressed sensing, Navier-Stokes regularity progress, sum-product estimates",
        "quote": "What mathematics achieves is remarkable — it describes all patterns of the universe",
        "image": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Terence_Tao.jpg",
        "impact": "Medical imaging ($10B+ MRI tech), signal processing, AI mathematics, number theory",
        "resources": ["Tao's Blog: https://terrytao.wordpress.com/"]
    },
    "Maryam Mirzakhani": {
        "period": "1977–2017", "country": "Iran",
        "fields": ["Differential Geometry", "Topology", "Teichmüller Theory"],
        "contribution": "FIRST WOMAN ever to win Fields Medal (2014). Revolutionary work on dynamics and geometry of Riemann surfaces and moduli spaces.",
        "keyresults": "Weil-Petersson volume formulas, moduli space dynamics, Teichmüller geodesics, simple closed curves",
        "quote": "The beauty of mathematics only shows itself to more patient followers",
        "image": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Maryam_Mirzakhani.jpg",
        "impact": "String theory, quantum gravity, breaking gender barriers — inspired millions of women into STEM",
        "resources": ["Wikipedia: https://en.wikipedia.org/wiki/Maryam_Mirzakhani"]
    },
    "Ada Lovelace": {
        "period": "1815–1852", "country": "England",
        "fields": ["Algorithm Design", "Computing", "Mathematical Logic"],
        "contribution": "World's first computer programmer. Wrote the first algorithm for Babbage's Analytical Engine. Visionary who predicted modern computing a century early.",
        "keyresults": "First computer program, loop and conditional concepts, algorithm for Bernoulli numbers, symbolic computing",
        "quote": "The Analytical Engine has no pretensions whatever to originate anything. But it may follow analysis.",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Ada_Lovelace_portrait.jpg",
        "impact": "All of computer science, $10T+ software industry, AI, every program ever written",
        "resources": ["Wikipedia: https://en.wikipedia.org/wiki/Ada_Lovelace"]
    },
    "Kurt Gödel": {
        "period": "1906–1978", "country": "Austria-Hungary / USA",
        "fields": ["Mathematical Logic", "Set Theory", "Foundations"],
        "contribution": "Incompleteness theorems destroyed Hilbert's formalism program. Proved some truths are unprovable within any consistent formal system.",
        "keyresults": "First and Second Incompleteness Theorems, Gödel completeness theorem, constructible universe L",
        "quote": "Either mathematics is too big for the human mind, or the human mind is more than a machine",
        "image": "https://upload.wikimedia.org/wikipedia/commons/8/84/KurtGodel.jpg",
        "impact": "Limits of artificial intelligence, philosophy of mind, halting problem, Church-Turing thesis",
        "resources": ["Stanford Encyclopedia: https://plato.stanford.edu/entries/goedel/"]
    },
    "Alan Turing": {
        "period": "1912–1954", "country": "England",
        "fields": ["Computability Theory", "Cryptography", "Artificial Intelligence"],
        "contribution": "Father of computer science. Turing machine defines computation. Cracked Enigma code saving an estimated 14 million lives in WWII.",
        "keyresults": "Turing machine, halting problem undecidability, Turing test, Enigma decryption, morphogenesis",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done",
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Alan_Turing_Aged_16.jpg",
        "impact": "All computation, $10T+ software industry, AI, cybersecurity, saved 14M+ WWII lives",
        "resources": ["MacTutor: http://www-history.mcs.st-and.ac.uk/Biographies/Turing.html"]
    }
}


# ═══════════════════════════════════════════════════
# MATH PROJECTS — 20 detailed entries
# ═══════════════════════════════════════════════════

MATH_PROJECTS = [
    {
        "id": 1,
        "title": "Machine Learning Classification using Linear Algebra",
        "math": ["Linear Algebra", "Eigenvalues", "SVD", "Gradient Descent"],
        "desc": "Build a complete ML classifier using SVD for dimensionality reduction and linear regression for prediction. Implement from scratch using only NumPy.",
        "real": "Google Search ranking, Netflix recommendations, Amazon product suggestions, medical imaging diagnosis",
        "companies": "Google, Netflix, Amazon, IBM Watson, Microsoft Azure ML",
        "salary": "Data Scientist $120K+  |  ML Engineer $140K+  |  AI Researcher $150K+",
        "difficulty": "Advanced"
    },
    {
        "id": 2,
        "title": "Cryptography: RSA Encryption with Number Theory",
        "math": ["Number Theory", "Prime Numbers", "Modular Arithmetic", "Euler's Theorem"],
        "desc": "Implement full RSA encryption using prime factorization. Generate key pairs, encrypt and decrypt messages. Understand computational hardness.",
        "real": "HTTPS (all of internet), WhatsApp E2E encryption, banking, government classified communication",
        "companies": "Apple, Google, Microsoft, all banks, NSA, GCHQ",
        "salary": "Cryptographer $115K+  |  Security Engineer $130K+  |  CISO $200K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 3,
        "title": "3D Graphics Engine using Matrix Transformations",
        "math": ["Rotation Matrices", "Homogeneous Coordinates", "Quaternions", "Projections"],
        "desc": "Build a software 3D renderer using transformation matrices. Implement rotation, scaling, translation, perspective projection, and lighting.",
        "real": "Video games (Unity, Unreal Engine), Pixar/DreamWorks films, VR/AR headsets, AutoCAD, Blender",
        "companies": "Unity, Epic Games, Pixar, Valve, Meta VR, NVIDIA",
        "salary": "Game Developer $110K+  |  Graphics Programmer $135K+  |  VR Engineer $125K+",
        "difficulty": "Advanced"
    },
    {
        "id": 4,
        "title": "Signal Processing: Audio Analysis with Fourier Transform",
        "math": ["Fourier Analysis", "FFT Algorithm", "Complex Numbers", "Convolution"],
        "desc": "Use Fast Fourier Transform to analyse audio signals. Build noise filters, equalizers, pitch detection. Visualise frequency spectra.",
        "real": "Spotify audio processing, Apple Music, noise-cancelling headphones (Bose/Sony), MRI scanners, seismic analysis",
        "companies": "Spotify, Apple, Bose, Sony, Siemens Healthcare, Schlumberger",
        "salary": "Audio Engineer $100K+  |  DSP Engineer $125K+  |  Biomedical Engineer $105K+",
        "difficulty": "Advanced"
    },
    {
        "id": 5,
        "title": "Portfolio Optimisation using Lagrange Multipliers",
        "math": ["Calculus", "Lagrange Multipliers", "Convex Optimisation", "Covariance Matrices"],
        "desc": "Maximise portfolio returns while minimising risk using Markowitz mean-variance optimisation. Implement efficient frontier calculation.",
        "real": "Goldman Sachs, JP Morgan, BlackRock ($10T AUM), Vanguard, all hedge funds worldwide",
        "companies": "Goldman Sachs, JP Morgan, BlackRock, Citadel, Renaissance Technologies",
        "salary": "Quant Analyst $150K+  |  Portfolio Manager $200K+  |  Hedge Fund Manager $500K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 6,
        "title": "Disease Spread Modelling using Differential Equations (SIR)",
        "math": ["ODE Systems", "SIR/SEIR Models", "Numerical Integration", "Phase Portraits"],
        "desc": "Build SIR/SEIR epidemic models using differential equations. Simulate vaccine distribution. Model herd immunity thresholds.",
        "real": "COVID-19 modelling (WHO, CDC, governments worldwide), public health policy, pandemic preparedness",
        "companies": "WHO, CDC, ICMR, NIH, McKinsey Health, national governments",
        "salary": "Epidemiologist $85K+  |  Public Health Modeller $95K+  |  Research Scientist $105K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 7,
        "title": "PageRank Algorithm using Eigenvalues and Markov Chains",
        "math": ["Graph Theory", "Eigenvalues/Eigenvectors", "Markov Chains", "Power Iteration"],
        "desc": "Implement Google's PageRank. Build a web graph. Use power iteration to find the principal eigenvector. Rank pages by importance.",
        "real": "Google Search (handles 99B+ searches/day), Microsoft Bing, citation network analysis, recommendation engines",
        "companies": "Google, Microsoft, Baidu, Yandex, academic citation systems",
        "salary": "Search Engineer $145K+  |  Ranking Scientist $155K+  |  Graph Data Scientist $130K+",
        "difficulty": "Advanced"
    },
    {
        "id": 8,
        "title": "Numerical ODE Solver: Runge-Kutta Methods",
        "math": ["Numerical Analysis", "RK4", "Error Analysis", "Stability Theory"],
        "desc": "Implement and compare Euler, RK2, RK4 ODE solvers. Analyse accuracy vs step-size. Solve stiff equations, chaotic systems.",
        "real": "Weather forecasting (Met Office), aircraft flight simulation, drug pharmacokinetics, astrophysics N-body problems",
        "companies": "Boeing, Airbus, NOAA, NASA, pharmaceutical companies (Pfizer, Roche)",
        "salary": "Computational Scientist $120K+  |  CFD Engineer $135K+  |  Simulation Engineer $125K+",
        "difficulty": "Advanced"
    },
    {
        "id": 9,
        "title": "Natural Language Processing using Probability and Vector Spaces",
        "math": ["Vector Spaces", "Probability Theory", "Bayes Theorem", "Information Theory"],
        "desc": "Build text classifier using TF-IDF, Naive Bayes, and word embeddings. Train on real datasets. Implement spam detection.",
        "real": "ChatGPT (OpenAI), Gmail spam filter, Google Translate, Siri/Alexa, sentiment analysis",
        "companies": "OpenAI, Google, Meta AI, Amazon Alexa, Apple Siri",
        "salary": "NLP Engineer $135K+  |  AI Researcher $155K+  |  Conversational AI Lead $165K+",
        "difficulty": "Advanced"
    },
    {
        "id": 10,
        "title": "Computer Vision: Image Recognition using Convolution Matrices",
        "math": ["2D Convolution", "Matrix Operations", "Eigenfaces (PCA)", "Linear Algebra"],
        "desc": "Build image recognition system using convolutional filters. Implement edge detection, feature extraction, simple CNN from scratch.",
        "real": "Tesla autopilot (vision-only), Face ID (Apple), medical pathology AI, satellite image analysis",
        "companies": "Tesla, Apple, DeepMind, Siemens AI, Palantir, defence contractors",
        "salary": "Computer Vision Engineer $145K+  |  Autonomous Vehicle ML $155K+  |  Vision Scientist $140K+",
        "difficulty": "Advanced"
    },
    {
        "id": 11,
        "title": "Recommendation System using Matrix Factorisation (SVD)",
        "math": ["SVD / Matrix Factorisation", "Collaborative Filtering", "Optimisation", "Regularisation"],
        "desc": "Implement collaborative filtering using SVD. Predict missing ratings. Build a Netflix-style recommendation engine.",
        "real": "Netflix (system worth $1B/year in retention), YouTube, Amazon, Spotify Discover Weekly",
        "companies": "Netflix, YouTube, Amazon, Spotify, TikTok, LinkedIn",
        "salary": "Recommender Systems Scientist $135K+  |  Personalisation Engineer $140K+",
        "difficulty": "Advanced"
    },
    {
        "id": 12,
        "title": "Blockchain and Cryptographic Hashing using Number Theory",
        "math": ["Hash Functions", "Merkle Trees", "Elliptic Curve Cryptography", "Number Theory"],
        "desc": "Understand SHA-256 hashing, Merkle trees, and blockchain consensus. Implement a simplified Bitcoin-like ledger from scratch.",
        "real": "Bitcoin, Ethereum ($2T+ market cap), DeFi, NFTs, supply chain verification (Walmart, Maersk)",
        "companies": "Coinbase, Binance, ConsenSys, IBM Blockchain, JPMorgan Onyx",
        "salary": "Blockchain Dev $145K+  |  Crypto Engineer $155K+  |  DeFi Protocol Engineer $160K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 13,
        "title": "Reinforcement Learning using Markov Decision Processes",
        "math": ["Probability", "MDP Theory", "Bellman Equations", "Dynamic Programming"],
        "desc": "Implement Q-learning agent to master grid environments and classic games. Understand reward functions and value iteration.",
        "real": "AlphaGo (DeepMind), robot locomotion (Boston Dynamics), autonomous trading, game AI",
        "companies": "DeepMind, OpenAI, Boston Dynamics, Waymo, Jane Street",
        "salary": "RL Research Engineer $155K+  |  Robotics ML Engineer $145K+  |  OpenAI Researcher $180K+",
        "difficulty": "Advanced"
    },
    {
        "id": 14,
        "title": "Quantum Computing using Linear Algebra and Complex Numbers",
        "math": ["Complex Vector Spaces", "Unitary Matrices", "Tensor Products", "Quantum Gates"],
        "desc": "Simulate quantum circuits using matrix multiplication. Implement Hadamard, CNOT, Toffoli gates. Simulate Grover's search algorithm.",
        "real": "IBM Quantum Network, Google Sycamore (quantum supremacy), drug discovery, cryptography breaking",
        "companies": "IBM, Google Quantum AI, IonQ, Microsoft Azure Quantum, D-Wave",
        "salary": "Quantum Engineer $185K+  |  Quantum Researcher $175K+  |  PhD almost always required",
        "difficulty": "Very Advanced"
    },
    {
        "id": 15,
        "title": "Time Series Forecasting: ARIMA and Kalman Filters",
        "math": ["Time Series Analysis", "ARIMA", "Fourier Decomposition", "Kalman Filter (ODEs)"],
        "desc": "Build multi-step forecasting model. Decompose time series into trend, seasonality, residual. Apply ARIMA and compare with Kalman.",
        "real": "Equity trading algorithms, electricity demand forecasting, retail inventory (Walmart), flu prediction (CDC)",
        "companies": "JPMorgan quant desks, Walmart supply chain, National Grid, Google Cloud AI",
        "salary": "Quant Researcher $155K+  |  Forecasting Data Scientist $125K+  |  Energy Analyst $115K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 16,
        "title": "Social Network Analysis using Graph Theory",
        "math": ["Graph Theory", "Betweenness Centrality", "Community Detection", "Spectral Graph Theory"],
        "desc": "Analyse real social networks. Find influencers via PageRank/centrality. Detect communities using Louvain algorithm.",
        "real": "Facebook friend recommendations, LinkedIn people you may know, Twitter/X trending detection, epidemiology",
        "companies": "Meta, LinkedIn, Twitter/X, TikTok, government intelligence agencies",
        "salary": "Graph Data Scientist $125K+  |  Social Analytics Engineer $115K+  |  Network Scientist $130K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 17,
        "title": "Supply Chain Optimisation using Linear Programming",
        "math": ["Linear Programming", "Integer Programming", "Simplex Method", "Graph Shortest Paths"],
        "desc": "Formulate and solve warehouse location, truck routing, and inventory optimisation as LP/ILP problems using PuLP/scipy.",
        "real": "Amazon logistics ($40B/year ops), Walmart, FedEx, DHL, global pharmaceutical supply chains",
        "companies": "Amazon, Walmart, FedEx, DHL, McKinsey Operations, Accenture Supply Chain",
        "salary": "Operations Research Analyst $120K+  |  Supply Chain Data Scientist $130K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 18,
        "title": "Bayesian Inference: Medical Diagnosis using Probability",
        "math": ["Bayes Theorem", "Prior/Posterior Distributions", "Bayesian Networks", "MCMC"],
        "desc": "Build Bayesian diagnostic system. Update disease probabilities as test results arrive. Implement MCMC sampling.",
        "real": "Cancer screening (false positive problem), COVID test interpretation, clinical trial analysis",
        "companies": "Flatiron Health, IBM Watson Health, Google Health, Optum, Pfizer clinical trials",
        "salary": "Biostatistician $100K+  |  Medical Data Scientist $115K+  |  Clinical Statistician $110K+",
        "difficulty": "Intermediate"
    },
    {
        "id": 19,
        "title": "Topological Data Analysis using Persistent Homology",
        "math": ["Algebraic Topology", "Simplicial Complexes", "Persistent Homology", "Betti Numbers"],
        "desc": "Apply TDA to find hidden multi-scale structures in high-dimensional datasets. Use Ripser/Gudhi for persistence diagrams.",
        "real": "Protein folding (AlphaFold adjacent), drug discovery, genomics, material science, neuroscience brain connectivity",
        "companies": "Genentech, Roche, IQVIA, national labs (LANL, ANL), Ayasdi",
        "salary": "Computational Biologist $130K+  |  TDA Research Scientist $145K+",
        "difficulty": "Very Advanced"
    },
    {
        "id": 20,
        "title": "Algorithm Design for Tech Interviews using Discrete Mathematics",
        "math": ["Graph Algorithms", "Dynamic Programming", "Combinatorics", "Probability"],
        "desc": "Master core algorithms: BFS, DFS, Dijkstra, A*, all DP patterns. Solve 200+ LeetCode problems systematically.",
        "real": "Technical interviews at Google, Meta, Apple, Amazon, Microsoft — every software job in the world",
        "companies": "Google, Meta, Apple, Amazon, Microsoft, Netflix, Stripe, Airbnb",
        "salary": "SWE L3 $160K+  |  Senior SWE L5 $250K+  |  Staff Engineer L6 $350K+  |  Principal $450K+",
        "difficulty": "Intermediate"
    }
]


# ═══════════════════════════════════════════════════
# THEOREM EXPLORER — 6 complete entries
# ═══════════════════════════════════════════════════

THEOREMS = {
    "Pythagorean Theorem": {
        "statement": "In a right triangle with legs \\(a, b\\) and hypotenuse \\(c\\): \\[a^2 + b^2 = c^2\\]",
        "proof_sketch": "Construct squares on each side. The two large squares (side a+b) have equal area. Rearranging the inner triangles proves the result. Alternatively: similar triangles formed by the altitude from the right angle each satisfy the relation.",
        "formal_proof": "Let \\(\\triangle ABC\\) have right angle at \\(C\\). Drop altitude \\(CD\\) to hypotenuse. Then \\(\\triangle ACD \\sim \\triangle ABC\\), giving \\(AC^2 = AD \\cdot AB\\). Similarly \\(BC^2 = DB \\cdot AB\\). Adding: \\(AC^2 + BC^2 = AB \\cdot (AD+DB) = AB^2\\). ✅",
        "applications": "Distance formula in \\(\\mathbb{R}^n\\), complex modulus, physics (Pythagoras in energy), GPS triangulation, Euclidean geometry",
        "difficulty": "Basic",
        "exam_relevance": "JAM, GATE, CSIR — appears in geometry, vector spaces, metric spaces"
    },
    "Fundamental Theorem of Calculus": {
        "statement": "If \\(f\\) is continuous on \\([a,b]\\) and \\(F'=f\\), then \\[\\int_a^b f(x)\\,dx = F(b) - F(a)\\]",
        "proof_sketch": "Define \\(G(x) = \\int_a^x f(t)\\,dt\\). Show \\(G'(x) = f(x)\\) using the mean value theorem for integrals. Then \\(G = F + C\\), so \\(\\int_a^b f = G(b)-G(a) = F(b)-F(a)\\).",
        "formal_proof": "By MVT: \\(\\frac{G(x+h)-G(x)}{h} = \\frac{1}{h}\\int_x^{x+h}f(t)\\,dt = f(c_h)\\) for some \\(c_h \\in (x,x+h)\\). As \\(h\\to 0\\), \\(c_h\\to x\\), and by continuity of \\(f\\), \\(G'(x)=f(x)\\). ✅",
        "applications": "All of integral calculus, physics (work = ∫F·dx), economics (revenue from marginal revenue), probability (CDF from PDF)",
        "difficulty": "Core",
        "exam_relevance": "CRITICAL for JAM Section A, GATE MA, CSIR Part B — appears every year"
    },
    "Euler's Identity": {
        "statement": "\\[e^{i\\pi} + 1 = 0\\] connecting the five fundamental constants: \\(e, i, \\pi, 1, 0\\).",
        "proof_sketch": "Taylor series: \\(e^x = \\sum x^n/n!\\). Substitute \\(x = i\\theta\\). Real parts give \\(\\cos\\theta\\), imaginary parts give \\(i\\sin\\theta\\). So \\(e^{i\\theta} = \\cos\\theta + i\\sin\\theta\\). At \\(\\theta=\\pi\\): \\(e^{i\\pi} = \\cos\\pi + i\\sin\\pi = -1\\). ✅",
        "formal_proof": "\\(e^{i\\pi} = \\sum_{n=0}^\\infty \\frac{(i\\pi)^n}{n!} = \\left(1 - \\frac{\\pi^2}{2!} + \\cdots\\right) + i\\left(\\pi - \\frac{\\pi^3}{3!} + \\cdots\\right) = \\cos\\pi + i\\sin\\pi = -1\\). ✅",
        "applications": "Complex analysis (Cauchy's theorem), quantum mechanics (wave functions ψ=Ae^(iωt)), electrical engineering (AC circuits), signal processing",
        "difficulty": "Advanced",
        "exam_relevance": "Complex Analysis for JAM, GATE, CSIR — direct formula applications and residue theorem"
    },
    "Prime Number Theorem": {
        "statement": "Let \\(\\pi(x)\\) = number of primes \\(\\leq x\\). Then \\[\\pi(x) \\sim \\frac{x}{\\ln x} \\quad \\text{as } x \\to \\infty\\]",
        "proof_sketch": "Deep analytic proof via Riemann zeta function \\(\\zeta(s) = \\sum n^{-s}\\). Analytic continuation to \\(\\mathbb{C}\\). Zeros of \\(\\zeta\\) on the critical line \\(\\text{Re}(s)=\\frac{1}{2}\\) (Riemann hypothesis, still unproven!) control the error term. Contour integration gives the asymptotic.",
        "formal_proof": "Proved independently by Hadamard and de la Vallée Poussin (1896) using the fact that \\(\\zeta(s) \\neq 0\\) on \\(\\text{Re}(s)=1\\). The explicit formula: \\(\\psi(x) = x - \\sum_\\rho \\frac{x^\\rho}{\\rho} - \\frac{\\zeta'(0)}{\\zeta(0)} + \\cdots\\)",
        "applications": "RSA key generation (prime density), randomised primality testing, cryptographic protocol design",
        "difficulty": "Very Hard",
        "exam_relevance": "CSIR Part C — concept and implications. Number theory sections in JAM/GATE."
    },
    "Cauchy-Schwarz Inequality": {
        "statement": "For vectors \\(u, v\\) in an inner product space: \\[|\\langle u, v \\rangle|^2 \\leq \\langle u, u \\rangle \\cdot \\langle v, v \\rangle\\] Equality iff \\(u, v\\) are linearly dependent.",
        "proof_sketch": "Consider \\(f(t) = \\langle u - tv, u - tv \\rangle \\geq 0\\) for all real \\(t\\). This is a quadratic in \\(t\\) with non-negative values, so its discriminant \\(\\leq 0\\). Expanding gives the inequality.",
        "formal_proof": "\\(0 \\leq \\|u-tv\\|^2 = \\|u\\|^2 - 2t\\langle u,v\\rangle + t^2\\|v\\|^2\\). Setting \\(t = \\frac{\\langle u,v\\rangle}{\\|v\\|^2}\\): \\(0 \\leq \\|u\\|^2 - \\frac{|\\langle u,v\\rangle|^2}{\\|v\\|^2}\\). Rearranging gives the result. ✅",
        "applications": "Linear algebra, quantum mechanics (Heisenberg uncertainty principle!), statistics (correlation ≤ 1), machine learning (cosine similarity), optimisation",
        "difficulty": "Intermediate",
        "exam_relevance": "EXTREMELY important for JAM, GATE, CSIR — appears in linear algebra, functional analysis, probability"
    },
    "Banach Fixed Point Theorem": {
        "statement": "Let \\((X, d)\\) be a complete metric space and \\(T: X \\to X\\) a contraction (\\(d(Tx, Ty) \\leq k\\,d(x,y)\\) for \\(k < 1\\)). Then \\(T\\) has a UNIQUE fixed point.",
        "proof_sketch": "Start with any \\(x_0\\). Define \\(x_{n+1} = T(x_n)\\). The sequence is Cauchy: \\(d(x_m, x_n) \\leq \\frac{k^n}{1-k}d(x_1, x_0)\\to 0\\). By completeness it converges to some \\(x^*\\). Continuity of \\(T\\) gives \\(T(x^*) = x^*\\). Uniqueness: if \\(Tp=p, Tq=q\\) then \\(d(p,q) = d(Tp,Tq) \\leq k\\,d(p,q)\\), so \\(d(p,q)=0\\). ✅",
        "applications": "Newton-Raphson convergence proof, Picard's existence theorem for ODEs, iterative linear system solvers, fractal geometry (IFS), economics equilibria",
        "difficulty": "Hard",
        "exam_relevance": "CSIR Part B/C, GATE — appears in functional analysis, metric spaces, ODE existence"
    }
}


# ═══════════════════════════════════════════════════
# COMPETITION PROBLEMS
# ═══════════════════════════════════════════════════

COMPETITION_PROBLEMS = {
    "IMO": [
        {
            "year": 2019, "number": 1,
            "problem": "Determine all functions \\(f: \\mathbb{Z} \\to \\mathbb{Z}\\) such that \\[f(2a)+2f(b)=f(f(a+b))\\] for all integers \\(a, b\\).",
            "difficulty": "Hard",
            "hint": "Substituting \\(a=0,b=0\\) gives \\(f(0)+2f(0)=f(f(0))\\). Try \\(f\\equiv c\\) (constant) and \\(f(x)=2x+c\\).",
            "answer": "\\(f(x) = 2x+c\\) for any constant \\(c \\in \\mathbb{Z}\\), or \\(f \\equiv 0\\)."
        },
        {
            "year": 2021, "number": 2,
            "problem": "Show that the equation \\(x^6 + x^3 + 1 = 3y^2\\) has no integer solutions.",
            "difficulty": "Hard",
            "hint": "Consider both sides modulo 9. Cubes mod 9 are only \\(\\{0, 1, 8\\}\\).",
            "answer": "The LHS mod 9 is never a quadratic residue times 3 mod 9 — contradiction shows no solutions exist."
        },
        {
            "year": 2022, "number": 1,
            "problem": "Let \\(ABCDE\\) be a convex pentagon such that \\(BC=DE\\). Assume that there is a point \\(T\\) inside the pentagon such that \\(TB=TD\\), \\(TC=TE\\) and \\(\\angle ABT = \\angle CDT = \\angle EAT\\). Prove that the line \\(AB\\) is parallel to \\(CD\\).",
            "difficulty": "Medium",
            "hint": "Use the angle condition to show spiral similarities. Triangles ABT and CDT are similar.",
            "answer": "The equal angle conditions imply \\(\\triangle ABT \\sim \\triangle CDT\\) (spiral similarity). This forces AB ∥ CD."
        }
    ],
    "PUTNAM": [
        {
            "year": 2020, "session": "A1",
            "problem": "How many positive integers \\(N\\) satisfy both of the following: (i) \\(N\\) is a multiple of 5 (ii) The decimal representation of \\(N\\) contains no digit other than 5 or 0?",
            "difficulty": "Medium",
            "hint": "Such numbers look like 5, 50, 55, 500, 505, 550, 555, ... Count by number of digits.",
            "answer": "There are 31 such positive integers."
        },
        {
            "year": 2019, "session": "B3",
            "problem": "Let \\(f: [0,1]\\to\\mathbb{R}\\) be continuous with \\(\\int_0^1 f(x)\\,dx = 0\\). Prove: \\[\\int_0^1 f(x)^2\\,dx \\geq 3\\left(\\int_0^1 xf(x)\\,dx\\right)^2\\]",
            "difficulty": "Very Hard",
            "hint": "Use the Cauchy-Schwarz inequality with the functions 1 and a suitable linear combination involving Legendre polynomials on [0,1].",
            "answer": "Write \\(f = c(2x-1) + g\\) where \\(\\int g = \\int g(2x-1) = 0\\). Then apply Pythagoras in \\(L^2[0,1]\\)."
        },
        {
            "year": 2018, "session": "A2",
            "problem": "Let \\(S_1, S_2, \\ldots, S_{2^n-1}\\) be the nonempty subsets of \\(\\{1,2,\\ldots,n\\}\\). Find \\[\\sum_{i=1}^{2^n-1} (-1)^{|S_i|+1} \\frac{1}{\\max(S_i)}\\]",
            "difficulty": "Hard",
            "hint": "Group subsets by their maximum element \\(k\\). For each \\(k\\), the contribution involves inclusion-exclusion over subsets of \\(\\{1,\\ldots,k-1\\}\\).",
            "answer": "The sum equals \\(H_n = 1 + \\frac{1}{2} + \\frac{1}{3} + \\cdots + \\frac{1}{n}\\), the \\(n\\)th harmonic number."
        }
    ],
    "AIME": [
        {
            "year": 2021, "number": 1,
            "problem": "Zou and Ceci both roll a fair six-sided die. What is the probability that they both roll the same number? Express as \\(\\frac{p}{q}\\) in lowest terms.",
            "difficulty": "Easy",
            "hint": "Fix Zou's roll. What is the probability Ceci matches?",
            "answer": "\\(\\frac{1}{6}\\)"
        },
        {
            "year": 2020, "number": 10,
            "problem": "There is a unique angle \\(\\theta\\) between \\(0°\\) and \\(90°\\) such that for nonneg integers \\(n\\), \\(\\tan(2^n\\theta)\\) is positive when \\(n\\) is a multiple of 3 and negative otherwise. What is \\(\\theta\\) written as \\(p/q\\) degrees in lowest terms?",
            "difficulty": "Hard",
            "hint": "Think about the angle doubling map on \\(\\mathbb{R}/180°\\mathbb{Z}\\). You need a periodic orbit of length 3 with specific sign pattern.",
            "answer": "\\(\\theta = \\frac{400}{7}\\) degrees, so \\(p+q = 407\\)."
        },
        {
            "year": 2019, "number": 15,
            "problem": "As shown in the figure, line segment \\(\\overline{AD}\\) is trisected by points \\(B\\) and \\(C\\) so that \\(AB=BC=CD=2\\). Three semicircles of radius 1, \\(\\overparen{AEB}\\), \\(\\overparen{BFC}\\), \\(\\overparen{CGD}\\), have their diameters on \\(\\overline{AD}\\). A circle of radius 2 has its center on \\(F\\). The area of the region inside the large circle but outside the three semicircles is \\(\\frac{m}{n}\\pi\\). Find \\(m+n\\).",
            "difficulty": "Very Hard",
            "hint": "Use inclusion-exclusion. Compute the area of the large circle minus the areas of overlap with the three semicircles.",
            "answer": "\\(m+n = 32\\)"
        }
    ]
}


# ═══════════════════════════════════════════════════
# LEARNING PATHS
# ═══════════════════════════════════════════════════

LEARNING_PATHS = {
    "Undergraduate to PhD": {
        "overview": "BSc Mathematics → IIT JAM → MSc IIT → CSIR NET JRF → PhD → Research/Faculty",
        "total_time": "Typically 10–12 years from BSc to independent researcher",
        "salary_journey": "₹0 (student) → ₹4–6L (MSc TA) → ₹31K/month (JRF) → ₹70K/month (SRF) → ₹60–150L (faculty/research)",
        "stages": [
            {
                "name": "Bachelor's Degree (3–4 years)",
                "goal": "Build solid foundation in core mathematics",
                "subjects": ["Calculus & Analysis", "Linear Algebra", "Abstract Algebra", "Complex Analysis", "Topology", "Differential Equations"],
                "tips": "Maintain 70%+ marks. Start reading classic texts (Rudin, Artin) from 2nd year. Solve problems daily.",
                "outcome": "BSc Mathematics degree"
            },
            {
                "name": "IIT JAM Preparation (6–12 months)",
                "goal": "Clear IIT JAM to get MSc at IIT/IISc",
                "subjects": ["Real Analysis (25%)", "Linear Algebra (20%)", "Calculus (20%)", "Group Theory (15%)", "Statistics (10%)"],
                "tips": "Solve last 15 years PYQs. Join a test series. Focus on Real Analysis and LA first. Do 3 mock tests per week in final 2 months.",
                "outcome": "Admission to IIT/IISc MSc — top 1% nationally"
            },
            {
                "name": "MSc at IIT/IISc (2 years)",
                "goal": "Master advanced mathematics and begin research exposure",
                "subjects": ["Functional Analysis", "Topology", "PDE", "Numerical Analysis", "Algebra", "Research Project"],
                "tips": "Attend seminars. Meet professors early. Start reading research papers in your interest area. Aim for CGPA > 8.",
                "outcome": "MSc degree + research exposure + network at top institute"
            },
            {
                "name": "CSIR-UGC NET JRF (during/after MSc)",
                "goal": "Get fellowship for PhD funding",
                "subjects": ["All MSc topics", "Topology", "Functional Analysis", "Complex Analysis", "Part C proof-writing"],
                "tips": "CSIR Part C is the key differentiator — practice proof writing. Previous 5 years papers are essential. Joint JRF gives ₹31,000/month.",
                "outcome": "JRF fellowship (₹31K/month → ₹35K SRF) + Lectureship eligibility"
            },
            {
                "name": "PhD Research (4–5 years)",
                "goal": "Produce original mathematical research",
                "subjects": ["Specialised coursework", "Research problem", "Paper writing", "Conference presentations"],
                "tips": "Choose advisor carefully — this is the most important decision. Publish at least 2 papers. Collaborate internationally.",
                "outcome": "PhD degree + published research + international network"
            },
            {
                "name": "Postdoc / Faculty / Industry",
                "goal": "Independent career in mathematics",
                "subjects": ["Independent research programme", "Grants", "Teaching", "Collaboration"],
                "tips": "Apply to 20+ positions. TIFR, IIT, IISER, ICTS are top choices. Industry: data science, quant finance also use PhD math heavily.",
                "outcome": "Faculty position (₹80–150L CTC) or Industry (₹60–200L CTC)"
            }
        ]
    },
    "JAM Fast Track": {
        "overview": "Focused 1-year JAM preparation plan for working/final-year students",
        "total_time": "8–12 months intensive preparation",
        "salary_journey": "₹0 → IIT MSc → ₹31K JRF → ₹40–80L career",
        "stages": [
            {
                "name": "Month 1–2: Foundation",
                "goal": "Revise BSc syllabus systematically",
                "subjects": ["Real Analysis basics", "Linear Algebra", "Calculus — all techniques", "Group Theory fundamentals"],
                "tips": "Use standard books. Make formula cards. Solve 20 problems per day minimum.",
                "outcome": "Solid conceptual foundation"
            },
            {
                "name": "Month 3–5: Topic Mastery",
                "goal": "Deep dive into high-weightage topics",
                "subjects": ["Real Analysis (sequences, series, continuity, differentiability, Riemann integral)", "Linear Algebra (all topics)", "Complex Analysis basics"],
                "tips": "Solve Arora & Sharma JAM book completely. Watch YouTube explanations for concepts you find hard.",
                "outcome": "Mastery of 80% of JAM syllabus"
            },
            {
                "name": "Month 6–8: Previous Year Papers",
                "goal": "Analyse and solve last 15 years JAM PYQs",
                "subjects": ["Timed PYQ practice", "Error analysis", "Weak area revision"],
                "tips": "Note every mistake. Revise that topic immediately. Time yourself strictly.",
                "outcome": "Pattern recognition and exam readiness"
            },
            {
                "name": "Month 9–12: Mock Tests & Revision",
                "goal": "Exam simulation and final preparation",
                "subjects": ["Full mock tests (weekly)", "Rapid revision", "Last-minute formula sheets"],
                "tips": "Take test in exam-like conditions. Aim for 70+ marks. Focus on accuracy over speed.",
                "outcome": "JAM score → IIT MSc admission"
            }
        ]
    },
    "GATE Mathematics Track": {
        "overview": "GATE MA → PSU jobs / IIT MTech / research positions → career growth",
        "total_time": "1 year prep → career of 30+ years",
        "salary_journey": "₹0 → ₹15–20L (PSU/MTech entry) → ₹30–50L (5 years) → ₹60–100L (10 years) → ₹100L+ (leadership)",
        "stages": [
            {
                "name": "GATE Preparation (8–12 months)",
                "goal": "Score 650+ in GATE MA for top PSU or IIT MTech",
                "subjects": ["Calculus (30%)", "Linear Algebra (20%)", "Complex Analysis (15%)", "Probability (15%)", "Numerical Analysis (10%)", "ODE/PDE (10%)"],
                "tips": "GATE has NAT (fill in) questions — practice these. No negative marking for NAT. Use NPTEL courses.",
                "outcome": "GATE score for PSU / IIT MTech / research positions"
            },
            {
                "name": "Entry Level: PSU / IIT MTech (2–5 years)",
                "goal": "Build technical expertise and professional skills",
                "subjects": ["Domain-specific mathematics", "Programming (Python/R)", "Data Analysis", "Communication"],
                "tips": "In PSUs: BHEL, GAIL, ONGC hire GATE scorers. IIT MTech gives research + placement. ₹15–20L starting.",
                "outcome": "Established career with ₹15–25L CTC"
            },
            {
                "name": "Mid-Career Growth (5–10 years)",
                "goal": "Senior technical or managerial roles",
                "subjects": ["Advanced analytics", "Team leadership", "Strategy", "Specialisation"],
                "tips": "Switch to data science / quant finance if interested in high salaries. MBA from IIM is another option.",
                "outcome": "₹30–60L CTC in core or ₹60–100L in data/quant"
            }
        ]
    }
}


# ═══════════════════════════════════════════════════
# EXAM FORMULAS (exam-specific, 15-20 each)
# ═══════════════════════════════════════════════════

EXAM_FORMULAS = {
    "Calculus": {
        "JAM": [
            "Derivative: \\(f'(x) = \\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}\\)",
            "Product Rule: \\((uv)' = u'v + uv'\\)",
            "Quotient Rule: \\(\\left(\\frac{u}{v}\\right)' = \\frac{u'v-uv'}{v^2}\\)",
            "Chain Rule: \\(\\frac{dy}{dx} = \\frac{dy}{du}\\cdot\\frac{du}{dx}\\)",
            "Power Rule: \\(\\int x^n\\,dx = \\frac{x^{n+1}}{n+1}+C\\quad (n\\neq-1)\\)",
            "Integration by Parts: \\(\\int u\\,dv = uv - \\int v\\,du\\)",
            "L'Hôpital: \\(\\lim_{x\\to a}\\frac{f(x)}{g(x)} = \\lim_{x\\to a}\\frac{f'(x)}{g'(x)}\\) when indeterminate",
            "MVT: \\(f'(c) = \\frac{f(b)-f(a)}{b-a}\\) for some \\(c\\in(a,b)\\)",
            "FTC: \\(\\int_a^b f'(x)\\,dx = f(b)-f(a)\\)",
            "Taylor: \\(f(x) = \\sum_{n=0}^\\infty \\frac{f^{(n)}(a)}{n!}(x-a)^n\\)",
            "\\(e^x = \\sum_{n=0}^\\infty\\frac{x^n}{n!}\\), \\(\\sin x = x-\\frac{x^3}{3!}+\\cdots\\), \\(\\cos x = 1-\\frac{x^2}{2!}+\\cdots\\)",
            "Rolle's Theorem: \\(f(a)=f(b)\\Rightarrow\\exists c:f'(c)=0\\)",
        ],
        "GATE": [
            "Double Integral: \\(\\iint_D f\\,dA = \\int_a^b\\int_{g_1(x)}^{g_2(x)}f\\,dy\\,dx\\)",
            "Green's Theorem: \\(\\oint_C(P\\,dx+Q\\,dy)=\\iint_D\\left(\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}\\right)dA\\)",
            "Stokes: \\(\\iint_S(\\nabla\\times\\mathbf{F})\\cdot d\\mathbf{S}=\\oint_C\\mathbf{F}\\cdot d\\mathbf{r}\\)",
            "Divergence: \\(\\oiint_S\\mathbf{F}\\cdot d\\mathbf{S}=\\iiint_V\\nabla\\cdot\\mathbf{F}\\,dV\\)",
            "Lagrange Multipliers: \\(\\nabla f = \\lambda\\nabla g\\) at constrained extrema",
        ],
        "CSIR": [
            "Lebesgue DCT: \\(\\lim\\int f_n\\,d\\mu = \\int\\lim f_n\\,d\\mu\\) under domination",
            "Fubini: \\(\\int\\int f\\,d(x\\times y) = \\int\\left(\\int f(x,y)\\,dy\\right)dx\\)",
            "Measure: \\(\\mu(A\\cup B) = \\mu(A)+\\mu(B)-\\mu(A\\cap B)\\)",
        ]
    },
    "Linear Algebra": {
        "JAM": [
            "Eigenvalue: \\(\\det(A-\\lambda I)=0\\)",
            "Trace: \\(\\text{tr}(A)=\\sum_i\\lambda_i\\), \\(\\det(A)=\\prod_i\\lambda_i\\)",
            "Rank-Nullity: \\(\\text{rank}(A)+\\text{null}(A)=n\\) (number of columns)",
            "Cayley-Hamilton: \\(p(A)=0\\) where \\(p(\\lambda)=\\det(A-\\lambda I)\\)",
            "Gram-Schmidt: \\(e_k = v_k - \\sum_{j<k}\\frac{\\langle v_k,e_j\\rangle}{\\|e_j\\|^2}e_j\\)",
        ],
        "GATE": [
            "SVD: \\(A = U\\Sigma V^T\\)",
            "LU decomposition: \\(A=LU\\) (Gaussian elimination)",
            "QR: \\(A=QR\\) where \\(Q^TQ=I\\)",
        ],
        "CSIR": [
            "Spectral Theorem: self-adjoint \\(\\Rightarrow A=\\sum_i\\lambda_i P_i\\)",
            "Jordan Form: \\(A=PJP^{-1}\\)",
        ]
    }
}


# ═══════════════════════════════════════════════════
# REAL-WORLD APPLICATIONS
# ═══════════════════════════════════════════════════

REALWORLD = [
    {
        "concept": "Fourier Transform",
        "application": "MRI Medical Imaging",
        "explanation": "MRI scanners record raw k-space data (Fourier-encoded radio frequency signals from hydrogen atoms). The Fourier Transform reconstructs this into detailed 3D anatomical images in milliseconds.",
        "companies": "Siemens Healthineers, GE Healthcare, Philips, Canon Medical",
        "impact": "Diagnoses cancer, brain tumours, spinal injuries without ionising radiation. ~100M MRI scans performed globally per year.",
        "salary": "Biomedical Engineer $105K+  |  MRI Physicist $120K+  |  Medical Imaging Scientist $115K+"
    },
    {
        "concept": "Eigenvalues and Linear Algebra",
        "application": "Google PageRank",
        "explanation": "Google models the web as a directed graph. PageRank is the principal eigenvector of a massive stochastic matrix (the normalised adjacency matrix). The power iteration method computes it iteratively.",
        "companies": "Google, Microsoft Bing, DuckDuckGo, Baidu, Yandex",
        "impact": "Handles 8.5 billion searches per day. PageRank patent valued at billions of dollars.",
        "salary": "Search Quality Engineer $145K+  |  Ranking Scientist $160K+  |  Distinguished Engineer $300K+"
    },
    {
        "concept": "Differential Equations",
        "application": "COVID-19 Pandemic Modelling",
        "explanation": "Governments used SIR/SEIR systems of ODEs to model disease spread. Parameters: β (transmission rate), γ (recovery rate). R₀ = β/γ determines epidemic vs endemic.",
        "companies": "WHO, CDC, NHS, ICMR, national governments, McKinsey Health",
        "impact": "Directly shaped lockdown decisions affecting billions. Estimated to have saved 10–50 million lives through timely interventions.",
        "salary": "Epidemiologist $90K+  |  Public Health Quantitative Analyst $100K+  |  WHO Consultant $120K+"
    },
    {
        "concept": "Number Theory and RSA",
        "application": "Internet Security (HTTPS/TLS)",
        "explanation": "RSA encryption relies on the mathematical hardness of factoring large semiprime numbers. If p, q are large primes, computing p×q is easy but recovering p, q from N=pq is infeasible (2048-bit N).",
        "companies": "Apple, Google, Amazon, all banks, NSA, every HTTPS website",
        "impact": "Protects every online transaction. Global e-commerce ($6T+/year) would be impossible without it.",
        "salary": "Cryptographer $115K+  |  Security Engineer $135K+  |  CISO $220K+"
    },
    {
        "concept": "Optimisation Calculus",
        "application": "Quantitative Finance and Portfolio Management",
        "explanation": "Markowitz mean-variance optimisation uses Lagrange multipliers to find the minimum-variance portfolio for a given expected return. The efficient frontier is a parabola in (σ, μ) space.",
        "companies": "Goldman Sachs, Citadel, Renaissance Technologies, BlackRock, JPMorgan",
        "impact": "Controls $100+ trillion in global financial assets. A 0.1% improvement in returns at BlackRock ($10T AUM) = $10 billion annually.",
        "salary": "Quantitative Analyst $160K+  |  Quant Researcher $200K+  |  Portfolio Manager $300K+"
    },
    {
        "concept": "Probability and Bayesian Statistics",
        "application": "Weather Forecasting",
        "explanation": "Modern numerical weather prediction combines physical ODE/PDE models with ensemble Bayesian methods. 50+ model runs with perturbed initial conditions give probabilistic forecasts.",
        "companies": "NOAA, UK Met Office, ECMWF, AccuWeather, IBM The Weather Company",
        "impact": "72-hour forecasts now as accurate as 36-hour forecasts were in 1980. Saves ~$3 billion/year in the US alone through disaster preparedness.",
        "salary": "Meteorologist $95K+  |  Climate Scientist $115K+  |  Atmospheric Modeller $110K+"
    },
    {
        "concept": "Graph Theory",
        "application": "Social Network Analysis and Recommendations",
        "explanation": "Friend recommendation uses graph centrality measures. Content recommendation uses collaborative filtering (matrix factorisation). Community detection uses spectral clustering of graph Laplacian.",
        "companies": "Meta, LinkedIn, Twitter/X, TikTok, YouTube, Netflix",
        "impact": "Meta's recommendation systems drive $100B+ in annual ad revenue. Netflix saves $1B/year in churn through personalised recommendations.",
        "salary": "Graph Data Scientist $130K+  |  Recommendation Engineer $145K+  |  Principal ML Engineer $250K+"
    }
]


# ═══════════════════════════════════════════════════
# RESEARCH HUB
# ═══════════════════════════════════════════════════

RESEARCH_HUB = {
    "Pure Mathematics": [
        "Analytic Number Theory and the Riemann Hypothesis",
        "Abstract Algebra: Groups, Rings, Fields and Galois Theory",
        "Algebraic Topology and Homotopy Theory",
        "Differential Geometry and Riemannian Manifolds",
        "Algebraic Geometry (Schemes, Sheaves, Cohomology)",
        "Category Theory and Homological Algebra",
        "Mathematical Logic, Model Theory and Set Theory",
        "Representation Theory of Lie Groups"
    ],
    "Applied Mathematics": [
        "Numerical Methods for PDEs (FEM, FDM, Spectral Methods)",
        "Convex and Non-Convex Optimisation Theory",
        "Dynamical Systems, Ergodic Theory and Chaos",
        "Fluid Dynamics (Navier-Stokes, Turbulence)",
        "Mathematical Biology (Reaction-Diffusion, Population Dynamics)",
        "Financial Mathematics (Stochastic Calculus, Black-Scholes)",
        "Control Theory and Optimal Control",
        "Mathematical Imaging and Compressed Sensing"
    ],
    "Probability and Statistics": [
        "Stochastic Processes and Brownian Motion",
        "Statistical Learning Theory and PAC Learning",
        "Bayesian Non-Parametrics and Gaussian Processes",
        "High-Dimensional Statistics and Random Matrix Theory",
        "Causal Inference and Potential Outcomes",
        "Information Theory (Shannon Entropy, Channel Capacity)",
        "Extreme Value Theory and Heavy-Tailed Distributions",
        "Spatial Statistics and Geostatistics"
    ],
    "Computational Mathematics": [
        "Quantum Algorithms (Shor, Grover, HHL)",
        "Algorithmic Game Theory and Mechanism Design",
        "Compressed Sensing and Sparse Recovery",
        "Topological Data Analysis and Persistent Homology",
        "Geometric Deep Learning (Graph Neural Networks)",
        "Scientific Machine Learning (Physics-Informed NNs)",
        "High Performance Computing and Parallel Algorithms",
        "Symbolic Computation and Computer Algebra Systems"
    ],
    "Analysis and Geometry": [
        "Harmonic Analysis and Wavelets",
        "Partial Differential Equations (Existence, Regularity)",
        "Functional Analysis and Operator Algebras (C*-algebras)",
        "Complex Analysis in Several Variables",
        "Symplectic Geometry and Mirror Symmetry",
        "Sub-Riemannian Geometry",
        "Geometric Measure Theory",
        "Metric Geometry and Alexandrov Spaces"
    ]
}
def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def parse_json_block(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def ai_dynamic_mathematicians(count=18):
    prompt = f"""Return ONLY valid JSON array with {count} mathematicians from diverse eras and countries.
Each item keys: name, period, country, fields (array of 3-5), contribution, keyresults, quote, image, impact, resources (array of 'Label: URL').
Use real public links (Wikipedia, MacTutor, university pages, archives). No markdown, no extra text."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data:
        return data
    return []


def ai_dynamic_projects(count=24):
    prompt = f"""Return ONLY valid JSON array with {count} mathematics projects useful for students.
Each item keys: title, difficulty, math (array), desc, real, companies, salary, links (array of 'Label: URL').
Make projects practical across AI, finance, optimization, statistics, cryptography, data science.
No markdown, no extra text."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data:
        return data
    return []

# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "groq": GROQ_AVAILABLE, "gemini": GEMINI_AVAILABLE,
                    "mathematicians": len(MATHEMATICIANS), "projects": len(MATH_PROJECTS),
                    "theorems": len(THEOREMS)})


# ── Chat ──────────────────────────────────────────

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
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Mathematicians ─────────────────────────────────

@app.route("/api/mathematician")
def mathematician_random():
    dyn = ai_dynamic_mathematicians(count=12)
    if dyn:
        d = random.choice(dyn)
        return jsonify(d)
    name, d = random.choice(list(MATHEMATICIANS.items()))
    return jsonify({"name": name, **d})

@app.route("/api/mathematicians")
def mathematician_list():
    dyn = ai_dynamic_mathematicians(count=18)
    if dyn:
        slim = [{"name": x.get("name", "Unknown"), "period": x.get("period", "N/A"),
                 "country": x.get("country", "N/A"), "fields": x.get("fields", [])}
                for x in dyn]
        return jsonify({"mathematicians": slim, "total": len(slim), "source": "dynamic"})
    return jsonify({
        "mathematicians": [
            {"name": n, "period": d["period"], "country": d["country"], "fields": d["fields"]}
            for n, d in MATHEMATICIANS.items()],
        "total": len(MATHEMATICIANS),
        "source": "local"
    })

@app.route("/api/mathematician/<name>")
def mathematician_detail(name):
    q = normalize_name(name)
    for n, d in MATHEMATICIANS.items():
        if q and (q in normalize_name(n) or normalize_name(n) in q):
            return jsonify({"name": n, **d, "source": "local"})

    prompt = f"""Return ONLY valid JSON for mathematician: {name}
Keys: name, period, country, fields (array), contribution, keyresults, quote, image, impact, resources (array of 'Label: URL').
Use accurate public references and direct URLs."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, dict) and data.get("name"):
        data["source"] = "dynamic"
        return jsonify(data)
    return jsonify({"error": "Not found"}), 404


# ── Projects ────────────────────────────────────────
DYNAMIC_PROJECTS_CACHE = []
@app.route("/api/projects")
def projects_list():
    global DYNAMIC_PROJECTS_CACHE
    dyn = ai_dynamic_projects(count=24)
    if dyn:
        DYNAMIC_PROJECTS_CACHE = []
        for i, p in enumerate(dyn, start=1):
            DYNAMIC_PROJECTS_CACHE.append({"id": i, **p})
        return jsonify({"projects": DYNAMIC_PROJECTS_CACHE, "total": len(DYNAMIC_PROJECTS_CACHE), "source": "dynamic"})

    return jsonify({"projects": MATH_PROJECTS, "total": len(MATH_PROJECTS), "source": "local"})

@app.route("/api/project/<int:pid>", methods=["POST"])
def project_detail(pid):
    source = DYNAMIC_PROJECTS_CACHE if DYNAMIC_PROJECTS_CACHE else MATH_PROJECTS
    p = next((x for x in source if int(x.get("id", -1)) == pid), None)
    if not p:
        return jsonify({"error": "Not found"}), 404

    links = ", ".join(p.get("links", [])) if isinstance(p.get("links"), list) else ""
    prompt = f"""Explain this maths project for students in concise but complete style.
Project: {p.get('title')}
Math topics: {', '.join(p.get('math', []))}
Description: {p.get('desc')}
Real-world use: {p.get('real')}
Companies: {p.get('companies')}
Reference links: {links}

Give Step 1, Step 2, Step 3... implementation and important formulas in LaTeX.
Keep text concise, math details accurate."""
    return jsonify({"project": p, "explanation": ask_simple(prompt, system=ASK_ANUPAM_PROMPT)})   


# ── Theorems ────────────────────────────────────────

@app.route("/api/theorems")
def theorems_list():
    return jsonify({"theorems": list(THEOREMS.keys()), "total": len(THEOREMS)})

@app.route("/api/theorem/<name>")
def theorem_detail(name):
    for n, d in THEOREMS.items():
        if name.lower() in n.lower():
            return jsonify({"name": n, **d})
    return jsonify({"error": "Not found"}), 404


# ── Competition ─────────────────────────────────────

@app.route("/api/competition/<cat>")
def competition(cat):
    probs = COMPETITION_PROBLEMS.get(cat.upper(), [])
    if not probs: return jsonify({"error": "Category not found. Use IMO, PUTNAM, or AIME"}), 404
    return jsonify({"category": cat.upper(), "problems": probs, "total": len(probs)})


# ── Learning Paths ──────────────────────────────────

@app.route("/api/learning-paths")
def learning_paths_list():
    return jsonify({
        "paths": [{"name": n, "overview": d["overview"]} for n, d in LEARNING_PATHS.items()],
        "total": len(LEARNING_PATHS)
    })

@app.route("/api/learning-path/<name>")
def learning_path_detail(name):
    for n, d in LEARNING_PATHS.items():
        if name.lower().replace("-"," ") in n.lower():
            return jsonify({"name": n, **d})
    return jsonify({"error": "Not found"}), 404


# ── Formula Sheet ───────────────────────────────────

@app.route("/api/formula", methods=["POST"])
def formula():
    data  = request.get_json()
    topic = data.get("topic", "Calculus")
    exam  = data.get("exam",  "JAM")

    stored = EXAM_FORMULAS.get(topic, {}).get(exam, [])
    stored_block = ("\n\nKNOWN FORMULAS TO INCLUDE:\n" +
                    "\n".join(f"• {f}" for f in stored)) if stored else ""

    prompt = f"""Generate a COMPLETE, exam-ready formula sheet for the topic: {topic}
Target exam: {exam} (level: {'BSc/Entry MSc' if exam=='JAM' else 'Advanced MSc' if exam=='GATE' else 'Research MSc/PhD'})
{stored_block}

FORMAT (for each formula):
📌 [Formula Name]
\\[ LaTeX formula \\]
When to use: [1 sentence]
Condition: [any restrictions, e.g. n≠-1]
Exam tip: [common mistake or trick]

Include AT LEAST 18 formulas.
End with: By Anupam Nigam | {TEACHER_YOUTUBE}
NEVER use * or **. ALL math in LaTeX."""

    answer = ask_simple(prompt, system=SYSTEM_PROMPT)
    return jsonify({"answer": answer})


# ── Revision ────────────────────────────────────────

@app.route("/api/revision", methods=["POST"])
def revision():
    topic = request.get_json().get("topic", "Calculus")
    prompt = f"""Give exactly 10 RAPID REVISION POINTS for: {topic} (graduate exam level)

For each point:
[N]. [TOPIC NAME IN CAPS]
Definition: [with LaTeX]
Key formula: \\[ ... \\]
Exam trap: [common wrong answer]
Memory trick: [how to remember]

Be concise but precise. NEVER use * or **. ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Concept Map ─────────────────────────────────────

@app.route("/api/conceptmap", methods=["POST"])
def conceptmap():
    topic = request.get_json().get("topic", "Calculus")
    prompt = f"""Create a DEEP concept map for: {topic}

Structure:
📌 Core Definition + LaTeX
💡 Key Sub-concepts (6-8) each with LaTeX + intuition
📐 How they connect (arrows with reason)
⭐ Top 5 theorems / results
🌍 3 real-world applications
📚 Prerequisites and what builds on this

NEVER use * or **. ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── LaTeX Generator ─────────────────────────────────

@app.route("/api/latex", methods=["POST"])
def latex_gen():
    text = request.get_json().get("text", "")
    prompt = f"""Generate professional LaTeX code for: {text}

Include:
1. Complete compilable code snippet (use \\begin{{document}}...\\end{{document}})
2. Explanation of each command
3. How to compile (pdflatex or overleaf)
4. Alternative simpler version if the formula is complex

Author comment in code: % By Anupam Nigam | {TEACHER_YOUTUBE}
NEVER use * or **."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Quiz ────────────────────────────────────────────

@app.route("/api/quiz/question", methods=["POST"])
def quiz_question():
    d = request.get_json()
    prompt = f"""Generate ONE rigorous MCQ for topic: {d.get('topic','Calculus')}
Difficulty: {d.get('difficulty','medium')}
Question {d.get('q_num',1)} of {d.get('total',5)}

EXACT FORMAT (must follow):
Q: [question text with LaTeX]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [single letter A/B/C/D]
EXPLANATION: [full step-by-step solution with LaTeX]

NEVER use * or **. ALL math in LaTeX."""

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


# ── PYQ ─────────────────────────────────────────────

@app.route("/api/pyq")
def pyq():
    exam   = request.args.get("exam", "JAM")
    topics = {"JAM":  ["Real Analysis","Linear Algebra","Calculus","Group Theory","Complex Analysis"],
              "GATE": ["Calculus","Linear Algebra","Complex Analysis","PDE","ODE"],
              "CSIR": ["Real Analysis","Topology","Algebra","Functional Analysis","Complex Analysis"]}
    topic  = random.choice(topics.get(exam, topics["JAM"]))
    year   = random.randint(2014, 2023)

    prompt = f"""Generate a realistic {exam} PYQ question for {topic} (year ~{year}).

FORMAT:
Question: [challenging problem with LaTeX]
Solution: [complete step-by-step with all LaTeX formulas]
Key Concept: [what theorem this tests]
Exam Tip: [approach for similar questions]

NEVER use * or **. ALL math in LaTeX."""
    raw   = ask_simple(prompt, system=SYSTEM_PROMPT)
    lines = raw.split('\n')
    q = next((l.replace("Question:","").strip() for l in lines if l.startswith("Question:")), raw[:300])
    a = next((l.replace("Solution:","").strip()  for l in lines if l.startswith("Solution:")),  "See full answer above.")
    return jsonify({"q": q, "a": a, "topic": topic, "year": year, "exam": exam})


# ── Challenge ────────────────────────────────────────

@app.route("/api/challenge")
def challenge():
    challenges = [
        "Prove that \\(\\sqrt{2}\\) is irrational using proof by contradiction.",
        "Find all critical points of \\(f(x)=x^3-3x+2\\) and classify them.",
        "Compute the eigenvalues of \\(A=\\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}\\).",
        "Evaluate \\(\\int x^2 e^x\\,dx\\) fully using integration by parts.",
        "Solve \\(\\frac{dy}{dx}+2y=4x\\) with initial condition \\(y(0)=1\\).",
        "Prove the Cauchy-Schwarz inequality: \\(|\\langle u,v\\rangle|^2\\leq\\|u\\|^2\\|v\\|^2\\).",
        "Show every convergent sequence is Cauchy.",
        "Compute \\(\\sum_{n=1}^\\infty\\frac{1}{n^2}\\) using Fourier series of \\(f(x)=x\\) on \\([-\\pi,\\pi]\\).",
        "Let \\(A\\) be an \\(n\\times n\\) matrix. Prove \\(\\text{tr}(AB)=\\text{tr}(BA)\\).",
        "Prove the intermediate value theorem assuming Bolzano-Weierstrass.",
    ]
    return jsonify({"challenge": random.choice(challenges)})


# ── Real World ───────────────────────────────────────

@app.route("/api/realworld")
def realworld_random():
    item = random.choice(REALWORLD)
    return jsonify(item)

@app.route("/api/realworld/<concept>")
def realworld_detail(concept):
    for item in REALWORLD:
        if concept.lower() in item["concept"].lower():
            return jsonify(item)
    return jsonify({"error": "Not found"}), 404


# ── Research Hub ─────────────────────────────────────

@app.route("/api/research-hub")
def research_hub_index():
    return jsonify({
        "categories": list(RESEARCH_HUB.keys()),
        "total_topics": sum(len(v) for v in RESEARCH_HUB.values())
    })

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

Include:
📌 Overview of the research area
🔬 Current state of the field (key open problems)
📐 Core mathematical tools + theorems with LaTeX
💡 Key researchers to follow
📚 Recommended starting papers and textbooks
🚀 How a student can get started

NEVER use * or **. ALL math in LaTeX."""
    return jsonify({"answer": ask_simple(prompt, system=SYSTEM_PROMPT)})


# ── Exam Info ────────────────────────────────────────
@app.route("/api/books", methods=["POST"])
def books_search():
    d = request.get_json() or {}
    topic = (d.get("topic") or "Mathematics").strip()
    exam = (d.get("exam") or "").strip()
    prompt = f"""Return ONLY valid JSON array of best books for topic: {topic}. Exam context: {exam or 'General'}.
Each item keys: name, author, level, why, link.
Use publicly accessible links (publisher, archive, official, reliable store). Keep list focused and student-friendly (8-12 items)."""
    raw = ask_simple(prompt, system=ASK_ANUPAM_PROMPT)
    data = parse_json_block(raw)
    if isinstance(data, list) and data:
        return jsonify({"books": data, "total": len(data), "source": "dynamic"})
    return jsonify({"books": [], "total": 0, "source": "none"})
@app.route("/api/exam/<exam>")
def exam_info(exam):
    info = {
        "JAM": {
            "full_name": "IIT JAM Mathematics",
            "conducting_body": "IITs (rotational) — IIT Delhi/Roorkee recently",
            "eligibility": "Bachelor's degree with Mathematics in at least 2 years",
            "pattern": "3 hours · 60 questions · 100 marks\nSection A: 30 MCQ (1 & 2 marks, ⅓ negative)\nSection B: 10 MSQ (2 marks, NO negative)\nSection C: 20 NAT (1 & 2 marks, NO negative)",
            "syllabus": "Real Analysis · Linear Algebra · Calculus · Differential Equations · Group Theory · Complex Analysis · Numerical Analysis · Statistics & Probability",
            "weightage": "Real Analysis 25% · Linear Algebra 20% · Calculus 20% · Group Theory 15% · Statistics 10% · Others 10%",
            "top_books": ["Rudin — Principles of Mathematical Analysis",
                         "Artin — Algebra", "Churchill — Complex Variables",
                         "Apostol — Calculus Vol 1 & 2"],
            "strategy": "Solve 15 years PYQs. Strong in Real Analysis = 60% of rank. Take 1 full mock test per week in last 3 months.",
            "website": "https://jam.iitd.ac.in"
        },
        "GATE": {
            "full_name": "GATE Mathematics (Paper Code: MA)",
            "conducting_body": "IITs / IISc (rotational)",
            "eligibility": "Bachelor's in Mathematics/Statistics/CS or related fields",
            "pattern": "3 hours · 65 questions · 100 marks\nGeneral Aptitude: 15 marks\nCore MA: 85 marks (MCQ + MSQ + NAT)",
            "syllabus": "Calculus · Linear Algebra · Real Analysis · Complex Analysis · ODE · PDE · Abstract Algebra · Functional Analysis · Numerical Analysis · Probability & Statistics",
            "weightage": "Calculus + LA: 30% · Real Analysis: 20% · Complex: 15% · Algebra: 15% · Others: 20%",
            "top_books": ["Apostol — Calculus", "Hoffman-Kunze — Linear Algebra",
                         "Conway — Complex Analysis", "Dummit-Foote — Abstract Algebra"],
            "strategy": "GATE score valid 3 years. NAT questions have no negative marking — attempt all. NPTEL videos are excellent for free.",
            "website": "https://gate.iitd.ac.in"
        },
        "CSIR": {
            "full_name": "CSIR UGC NET Mathematics",
            "conducting_body": "NTA (National Testing Agency)",
            "eligibility": "Master's in Mathematics with 55% (50% SC/ST/PwD)",
            "pattern": "3 hours · 200 marks total\nPart A: 20Q (General Science, 30 marks)\nPart B: 40Q (Core math, 70 marks, ⅓ negative)\nPart C: 60Q (Advanced, 100 marks, ⅓ negative, PROOF-BASED)",
            "syllabus": "Analysis (Real+Complex+Functional) · Algebra (Linear+Abstract) · Topology · ODE · PDE · Numerical Methods · Probability & Statistics · Differential Geometry",
            "weightage": "Analysis: 30% · Algebra: 25% · Complex: 20% · Topology: 10% · Others: 15%",
            "top_books": ["Rudin — Real & Complex Analysis",
                         "Dummit-Foote — Abstract Algebra",
                         "Munkres — Topology",
                         "Conway — Functions of One Complex Variable"],
            "strategy": "Part C is the key differentiator. Master proof writing. JRF = ₹31,000/month for PhD. Lectureship = teaching eligibility.",
            "website": "https://csirnet.nta.nic.in"
        }
    }
    return jsonify(info.get(exam, {"error": "Not found. Use JAM, GATE, or CSIR"}))


# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n🧮 MathSphere v5.0 FINAL — starting on port {port}")
    print(f"   👥 {len(MATHEMATICIANS)} Mathematicians")
    print(f"   🚀 {len(MATH_PROJECTS)} Projects")
    print(f"   📐 {len(THEOREMS)} Theorems")
    print(f"   🏆 {sum(len(v) for v in COMPETITION_PROBLEMS.values())} Competition Problems")
    print(f"   🎓 {len(LEARNING_PATHS)} Learning Paths")
    print(f"   🔬 {sum(len(v) for v in RESEARCH_HUB.values())} Research Topics")
    print(f"   📺 {TEACHER_YOUTUBE}\n")
    app.run(host="0.0.0.0", port=port, debug=False)