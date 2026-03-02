import os,sys,io,json,logging,re
from flask import Flask,request,jsonify,send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

if sys.platform=='win32':
    sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY=os.getenv('GROQ_API_KEY','')
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY','')

try:
    from groq import Groq
    groq_client=Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE=True
except:GROQ_AVAILABLE=False

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE=True
except:GEMINI_AVAILABLE=False

try:
    from sympy import symbols,N
    from sympy.parsing.sympy_parser import parse_expr,standard_transformations,implicit_multiplication_application,convert_xor
    SYMPY_AVAILABLE=True
except:SYMPY_AVAILABLE=False

try:
    from numpy import isfinite,isnan
except:
    isfinite=lambda x:True
    isnan=lambda x:False

base_dir=os.path.abspath(os.path.dirname(__file__))
static_dir=os.path.join(base_dir,'static')
app=Flask(__name__,static_folder=static_dir,static_url_path='')
app.config['SECRET_KEY']='mathsphere'
app.config['MAX_CONTENT_LENGTH']=16*1024*1024

CORS(app)
limiter=Limiter(app=app,key_func=get_remote_address,storage_uri="memory://")
cache=Cache(app,config={'CACHE_TYPE':'SimpleCache'})

def clean(t):
    if not t:return ""
    for s in ['**','***','##','###','#','__','~~','```','`']:
        t=str(t).replace(s,'')
    return t

def sanitize(u):
    if not u:return ""
    return str(u).strip()[:5000]

def ask(p,temp=0.2,tk=1500):
    if not p:return ""
    try:
        if GROQ_AVAILABLE:
            r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",
                messages=[{"role":"system","content":"Expert mathematician. No markdown. Plain text."},{"role":"user","content":p}],
                temperature=temp,max_tokens=tk)
            return clean(r.choices[0].message.content)
        if GEMINI_AVAILABLE:
            m=genai.GenerativeModel('gemini-2.5-flash')
            r=m.generate_content(p)
            return clean(r.text if r.text else "")
    except:pass
    return ""

@app.route("/")
def index():
    try:return send_from_directory(static_dir,'index.html')
    except:return ""

@app.route("/<path:f>")
def serve(f):
    try:return send_from_directory(static_dir,f)
    except:return ""

@app.route("/api/health")
def health():return jsonify({"ok":True})

@app.route("/api/chat",methods=["POST"])
@limiter.limit("50 per minute")
def chat():
    d=request.get_json()or{}
    m=d.get("message","")
    if not m and d.get("messages"):
        for msg in reversed(d.get("messages",[])):
            if msg.get("role")=="user":m=msg.get("content","");break
    return jsonify({"answer":ask(sanitize(m),0.3,2000)})

@app.route("/api/formula",methods=["POST"])
def formula():
    d=request.get_json()or{}
    t=sanitize(d.get("topic","Calculus"))
    return jsonify({"answer":ask(f"Generate 50+ formulas for {t}. Organized. No markdown.",0.05,4000)})

@app.route("/api/graph",methods=["POST"])
def graph():
    if not SYMPY_AVAILABLE:return jsonify({"points":[]})
    d=request.get_json()or{}
    e=sanitize(d.get("expression","x**2"))
    try:
        f=parse_expr(e,transformations=(standard_transformations+(implicit_multiplication_application,convert_xor)))
        x=symbols('x')
        pts=[]
        for i in range(301):
            xv=-5+(10/300)*i
            try:
                yv=float(N(f.subs(x,xv),5))
                pts.append({"x":round(xv,4),"y":round(yv,4)if(yv and isfinite(yv)and not isnan(yv)and abs(yv)<1e4)else None})
            except:pts.append({"x":round(xv,4),"y":None})
        return jsonify({"points":pts,"expression":e})
    except:return jsonify({"points":[]})

@app.route("/api/competition/problems",methods=["POST"])
def comp():
    d=request.get_json()or{}
    c=sanitize(d.get("category","IMO"))
    n=min(int(d.get("count",10)),100)
    return jsonify({"problems":ask(f"Generate {n} {c} problems with solutions. No markdown.",0.2,6000)})

@app.route("/api/quiz/generate",methods=["POST"])
def quiz():
    d=request.get_json()or{}
    t=sanitize(d.get("topic","Calculus"))
    n=min(int(d.get("count",10)),100)
    return jsonify({"questions":ask(f"Generate {n} quiz questions on {t} with solutions. No markdown.",0.2,5000)})

@app.route("/api/research",methods=["POST"])
def research():
    d=request.get_json()or{}
    q=sanitize(d.get("query",""))
    return jsonify({"response":ask(f"Research on: {q}. Concepts, methods, resources. No markdown.",0.2,3000)})

@app.route("/api/exam/info",methods=["POST"])
def exam():
    d=request.get_json()or{}
    e=sanitize(d.get("exam","jam")).lower()
    exs={"jam":{"title":"IIT JAM","link":"https://jam.iitd.ac.in/"},"gate":{"title":"GATE","link":"https://gate.iitm.ac.in/"},"csir":{"title":"CSIR NET","link":"https://www.csirnet.nta.ac.in/"}}
    return jsonify({"details":exs.get(e,exs["jam"])})

@app.route("/api/pyq/load",methods=["POST"])
def pyq():
    d=request.get_json()or{}
    e=sanitize(d.get("exam","jam")).lower()
    n=min(int(d.get("count",10)),100)
    return jsonify({"questions":ask(f"Generate {n} PYQs from {e} with solutions. No markdown.",0.1,6000)})

@app.route("/api/mathematician",methods=["POST"])
def math():
    d=request.get_json()or{}
    n=sanitize(d.get("name",""))
    if n:p=f"Info on mathematician {n}. Biography, work, impact. No markdown."
    else:p="Random mathematician. Complete info. No markdown."
    return jsonify({"response":ask(p,0.2,3000)})

@app.route("/api/theorem/prove",methods=["POST"])
def theorem():
    d=request.get_json()or{}
    t=sanitize(d.get("theorem","Pythagorean Theorem"))
    return jsonify({"proof":ask(f"Prove {t} completely. No markdown.",0.1,4000)})

@app.route("/api/projects/generate",methods=["POST"])
def proj():
    d=request.get_json()or{}
    t=sanitize(d.get("topic","Machine Learning"))
    return jsonify({"projects":ask(f"10 projects for {t}. Title, code, resources. No markdown.",0.2,6000)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)),debug=False)