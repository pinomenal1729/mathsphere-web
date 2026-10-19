"""Reference retrieval and bounded calculation planning, separate from model training."""
import json
import math
from pathlib import Path
import re

from math_tools import run_calculations

NOTES = json.loads((Path(__file__).resolve().parent / 'data' / 'math_reference.json').read_text())['notes']
STOP = set('a an the is are was were to of on in and or for with how what why can you me please explain show that this it as be by from'.split())
ALIASES = {'matrices':'matrix', 'differentiation':'derivative', 'differentiate':'derivative',
           'differentiating':'derivative', 'integration':'integral', 'integrate':'integral',
           'integrating':'integral', 'invertible':'inverse', 'continuity':'continuous',
           'eigenvectors':'eigenvalue', 'eigenvalues':'eigenvalue', 'convergent':'convergence'}


def tokens(text):
    terms = re.findall(r'[a-z]+', str(text).lower())
    return {ALIASES.get(word, word[:-1] if word.endswith('s') and len(word) > 4 else word)
            for word in terms if word not in STOP}


def retrieve_references(question, limit=3):
    query = tokens(question)
    scored = []
    for note in NOTES:
        title = tokens(note['title'])
        keywords = tokens(' '.join(note['keywords']))
        content = tokens(note['content'])
        overlap = query & (title | keywords)
        if not overlap:
            continue
        score = (4 * len(query & title) + 3 * len(query & keywords) + len(query & content)) / math.sqrt(len(content) + 10)
        scored.append((score, note['id'], note))
    return [note for _, _, note in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


PLANNER_SYSTEM = r'''You translate a mathematical question into bounded calculator requests, not an answer.
Return one JSON object only: {"calculations": [...], "clarification": ""}.
If the question is ambiguous, put a concise clarification question in clarification and leave calculations empty. Never invent missing data, equations, or image contents.
For a conceptual question or proof without a relevant explicit calculation, return an empty list.
Use at most three operations from:
- evaluate, simplify: expression
- differentiate: expression, variable, optional order (1..3), optional at
- integrate: expression, variable, optional lower AND upper
- solve: expression (an equation such as "x^2=4"), variable, domain (real or complex)
- limit: expression, variable, point, direction (both, +, -)
- sum: expression, variable (summation index), lower, upper
- determinant, inverse, rank, eigenvalues: matrix (array of rows of expression strings)
All numeric/expression values must be strings, except derivative order, which is an integer.
Allowed symbols: x,y,z,t,n,a,b,c,r,pi,e,E,I,oo. Allowed functions: sin,cos,tan,exp,log,ln,sqrt,abs,sinh,cosh,atan.
Use explicit multiplication (*), ^ for powers, and exact fractions. Never emit Python code, attributes, imports, lists inside expressions, or function definitions.
Use domain=real unless the question explicitly concerns complex values. Preserve the original equation, denominator exclusions, all bounds, and requested variable. Do not reduce a two-sided limit to one side.
A calculator result verifies that calculation only; it cannot validate your interpretation of a word problem or prove a general theorem.'''


def needs_calculation(question):
    return bool(re.search(r'\b(differentiat\w*|derivative|integrat\w*|integral|solve|calculat\w*|evaluat\w*|simplif\w*|determinant|eigenvalue\w*|inverse|rank|limit|summation|sum|factor|expand)\b|\d\s*[+*/^=−-]\s*\d', question, re.I))


def decode_plan(text):
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return [], ''
        tasks = obj.get('calculations', [])
        clarification = obj.get('clarification', '')
        if not isinstance(tasks, list) or len(tasks) > 3 or any(not isinstance(task, dict) for task in tasks):
            return [], ''
        if isinstance(clarification, str) and clarification.strip():
            return [], clarification.strip()[:500]
        return tasks, ''
    except (ValueError, TypeError):
        return [], ''


def prepare_context(question, plan_text=None):
    notes = retrieve_references(question)
    tasks, clarification = decode_plan(plan_text) if plan_text is not None else ([], '')
    results = run_calculations(tasks)
    completed = [result for result in results if result.get('status') == 'computed']
    verification = {
        'status': 'computed' if completed else 'not_checked',
        'calculations': results,
        'notice': 'These results check the displayed calculations, not the interpretation of your question or every claim in the explanation.',
    }
    return notes, verification, clarification


def grounded_system(base_prompt, notes, verification, style='step_by_step'):
    if not isinstance(style, str):
        style = 'step_by_step'
    styles = {
        'step_by_step': 'Use clear, short, numbered steps. Explain each important mathematical move in everyday language, then give the formal statement.',
        'concise': 'Give a concise derivation, the answer, and the essential assumptions. Keep the mathematical checks even when brief.',
        'exam': 'Give an exam-ready derivation with precise notation, stated hypotheses, and the quickest justified method. Add one common trap.',
    }
    context = {'reference_notes': [dict(id=n['id'], title=n['title'], content=n['content']) for n in notes],
               'symbolic_calculations': verification['calculations']}
    return base_prompt + '\n\nTEACHING STYLE:\n' + styles.get(style, styles['step_by_step']) + r'''
TRUSTED REFERENCE AND CALCULATOR CONTEXT:
The JSON below contains locally authored reference notes and calculator results. Use only relevant facts and retain all hypotheses. Cite note IDs only if you actually use those notes. Never fabricate a book, theorem, source, or citation.
For a computed result, confirm that the displayed input and assumptions match the question before using it. Preserve original domains, excluded points, interval conditions, units, solution completeness, and +C for antiderivatives. Include the calculator result in your final answer when relevant, in exact form first.
A result marked unavailable or unevaluated is NOT a completed check. A numerical example is NOT a proof. The calculator cannot establish that the interpretation of a word problem is correct.
If the context does not resolve something, work carefully and state uncertainty rather than making up a checked result. Do not claim that the entire answer is certified, independently verified, or guaranteed correct.
''' + json.dumps(context, ensure_ascii=False)
