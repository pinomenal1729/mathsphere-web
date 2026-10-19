"""Bounded symbolic tools. User expressions are parsed as a restricted AST, never eval'd."""
import ast
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys

MAX_TASKS = 3
MAX_EXPRESSION = 280


def _calculate(task):
    import sympy as sp
    from sympy.calculus.util import continuous_domain

    if not isinstance(task, dict):
        raise ValueError('A calculation must be an object.')
    domain_name = task.get('domain', 'real')
    if domain_name not in ('real', 'complex'):
        raise ValueError('Domain must be real or complex.')
    symbols = {name: sp.Symbol(name, real=True) if domain_name == 'real' else sp.Symbol(name)
               for name in ('x', 'y', 'z', 't', 'n', 'a', 'b', 'c', 'r')}
    constants = {'pi': sp.pi, 'e': sp.E, 'E': sp.E, 'I': sp.I, 'oo': sp.oo}
    functions = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp,
                 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'abs': sp.Abs,
                 'sinh': sp.sinh, 'cosh': sp.cosh, 'atan': sp.atan}
    denominators = []

    def parse(value):
        text = str(value).strip().replace('^', '**').replace('π', 'pi').replace('×', '*')
        if not text or len(text) > MAX_EXPRESSION:
            raise ValueError('Expression is empty or too long.')
        tree = ast.parse(text, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError('Expression is too complex.')

        def convert(node, depth=0):
            if depth > 15:
                raise ValueError('Expression nesting is too deep.')
            walk = lambda child: convert(child, depth + 1)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if not math.isfinite(node.value) or abs(node.value) > 10**30:
                    raise ValueError('Numeric constant is too large.')
                literal = ast.get_source_segment(text, node).replace('_', '')
                exponent = re.search(r'[eE]([+-]?\d+)$', literal)
                if exponent and abs(int(exponent.group(1))) > 100:
                    raise ValueError('Scientific-notation exponent is outside the supported range.')
                # Preserve the source digits instead of accepting AST float rounding.
                return sp.Rational(literal)
            if isinstance(node, ast.Name):
                if node.id in symbols:
                    return symbols[node.id]
                if node.id in constants:
                    return constants[node.id]
                raise ValueError('Unknown symbol. Use x, y, z, t, n, a, b, c, or r.')
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
            if isinstance(node, ast.BinOp):
                left, right = walk(node.left), walk(node.right)
                if isinstance(node.op, ast.Add):
                    return sp.Add(left, right, evaluate=False)
                if isinstance(node.op, ast.Sub):
                    return sp.Add(left, -right, evaluate=False)
                if isinstance(node.op, ast.Mult):
                    return sp.Mul(left, right, evaluate=False)
                if isinstance(node.op, ast.Div):
                    if right.is_zero:
                        raise ValueError('Division by zero is undefined.')
                    denominators.append(right)
                    return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
                if isinstance(node.op, ast.Pow):
                    if left.is_zero and right.is_zero:
                        raise ValueError('0^0 depends on the mathematical context; no unconditional value is certified.')
                    if right.is_number and (not right.is_real or abs(right) > 100):
                        raise ValueError('Exponent is outside the supported range.')
                    if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Pow):
                        raise ValueError('Power towers are not supported.')
                    if right.is_number and right.is_negative:
                        denominators.append(left)
                    return sp.Pow(left, right, evaluate=False)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                arity = (1, 2) if name in ('log', 'ln') else (1,)
                if name not in functions or len(node.args) not in arity or node.keywords:
                    raise ValueError('Unsupported function or arguments.')
                return functions[name](*(walk(arg) for arg in node.args), evaluate=False)
            raise ValueError('Only mathematical arithmetic and supported functions are allowed.')
        result = convert(tree.body)
        if result.has(sp.zoo, sp.nan):
            raise ValueError('Expression is undefined.')
        return result

    def text(value):
        output = str(value)
        if len(output) > 8000:
            raise ValueError('Result is too large to display reliably.')
        return output

    operation = task.get('operation')
    variable_name = task.get('variable', 'x')
    if variable_name not in symbols:
        raise ValueError('Unsupported variable.')
    variable = symbols[variable_name]
    if domain_name == 'complex' and (operation == 'limit' or (operation == 'differentiate' and 'at' in task)):
        raise ValueError('Complex limits and pointwise complex derivatives require all-path or holomorphicity analysis, beyond this calculator.')
    result = {'operation': operation, 'status': 'computed',
              'assumptions': f'Variables are {domain_name}; principal branches are used for multivalued functions.'}
    if operation in ('determinant', 'inverse', 'rank', 'eigenvalues'):
        rows = task.get('matrix')
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 5
                or any(not isinstance(row, list) or not 1 <= len(row) <= 5 for row in rows)
                or len({len(row) for row in rows}) != 1):
            raise ValueError('Use a rectangular matrix no larger than 5 by 5.')
        matrix = sp.Matrix([[parse(cell) for cell in row] for row in rows])
        if operation != 'rank' and matrix.rows != matrix.cols:
            raise ValueError('This operation requires a square matrix.')
        result['input_latex'] = sp.latex(matrix)
        if operation == 'determinant':
            answer = matrix.det()
        elif operation == 'inverse':
            answer = matrix.inv()
            if (matrix * answer - sp.eye(matrix.rows)).applyfunc(sp.simplify) != sp.zeros(matrix.rows):
                raise ValueError('Inverse identity could not be checked.')
            result['check'] = 'Multiplying by the original matrix gives the identity; requires determinant != 0.'
        elif operation == 'rank':
            answer = matrix.rank()
            if matrix.free_symbols:
                result['assumptions'] += ' Symbolic rank is generic; special parameter values may lower it.'
        else:
            eigenvalues = matrix.eigenvals()
            answer = sp.Dict(eigenvalues)
            result['check'] = 'Eigenvalues are mapped to algebraic multiplicities.'
    else:
        expression_text = task.get('expression', '')
        if operation == 'solve':
            pieces = str(expression_text).split('=')
            if len(pieces) > 2:
                raise ValueError('Provide one equation.')
            expression = parse(pieces[0]) - (parse(pieces[1]) if len(pieces) == 2 else 0)
        else:
            expression = parse(expression_text)
        result['input_latex'] = sp.latex(expression)
        calculation_domain = sp.S.Reals if domain_name == 'real' else sp.S.Complexes
        if domain_name == 'real' and expression.free_symbols <= {variable}:
            try:
                calculation_domain = continuous_domain(expression, variable, sp.S.Reals)
                for denominator in denominators:
                    calculation_domain -= sp.solveset(denominator, variable, domain=sp.S.Reals)
                result['input_domain'] = text(calculation_domain)
            except (NotImplementedError, ValueError):
                result['input_domain'] = 'Domain could not be fully determined; check restrictions separately.'
        if calculation_domain is sp.S.EmptySet:
            raise ValueError('The expression has no real domain.')
        if operation == 'evaluate':
            if expression.free_symbols:
                raise ValueError('Numerical evaluation requires an expression without free variables.')
            answer = sp.simplify(expression)
            if domain_name == 'real' and answer.is_real is False:
                raise ValueError('This value is not real; select the complex domain explicitly.')
        elif operation == 'simplify':
            answer = sp.simplify(expression)
            result['check'] = 'Simplification applies on the original expression domain, not at excluded points.'
        elif operation == 'differentiate':
            order = task.get('order', 1)
            if type(order) is not int or not 1 <= order <= 3:
                raise ValueError('Derivative order must be between 1 and 3.')
            answer = sp.diff(expression, variable, order)
            result['check'] = 'Derivative formula applies where the original function is differentiable; continuity alone is insufficient.'
            if 'at' in task:
                point = parse(task['at'])
                if calculation_domain.contains(point) is sp.S.false:
                    raise ValueError('The evaluation point is outside the original expression domain.')
                if calculation_domain.contains(point) is not sp.S.true:
                    result['conditions'] = 'Evaluation requires: ' + text(calculation_domain.contains(point))
                previous = sp.diff(expression, variable, order - 1)
                quotient = (previous - previous.subs(variable, point)) / (variable - point)
                left = sp.limit(quotient, variable, point, dir='-')
                right = sp.limit(quotient, variable, point, dir='+')
                if left != right or left.is_finite is False:
                    result.update(result='The derivative does not exist at this point.',
                                  latex=r'\text{Derivative does not exist at this point}',
                                  check=f'Left difference quotient: {text(left)}; right: {text(right)}.')
                    return result
                answer = right
                result['check'] = 'Both one-sided difference quotients agree and give the displayed finite value.'
        elif operation == 'integrate':
            if ('lower' in task) != ('upper' in task):
                raise ValueError('A definite integral needs both bounds.')
            if 'lower' in task:
                answer = sp.integrate(expression, (variable, parse(task['lower']), parse(task['upper'])))
                result['check'] = 'Definite integral; convergence and any Piecewise conditions must be respected.'
            else:
                answer = sp.integrate(expression, variable)
                result['constant_of_integration'] = True
                if domain_name == 'real':
                    answer = answer.replace(
                        lambda node: node.func == sp.log and node.args[0].is_real is True,
                        lambda node: sp.log(sp.Abs(node.args[0])),
                    )
                if not answer.has(sp.Integral) and sp.simplify(sp.diff(answer, variable) - expression) == 0:
                    result['check'] = 'Differentiating this antiderivative reproduces the integrand. Add +C.'
                else:
                    result['check'] = 'Antiderivative could not be independently checked; add +C if evaluated.'
        elif operation == 'solve':
            domain = calculation_domain
            for denominator in denominators:
                domain -= sp.solveset(denominator, variable, domain=domain)
            answer = sp.solveset(expression, variable, domain=domain)
            if isinstance(answer, sp.FiniteSet):
                if any(sp.simplify(expression.subs(variable, root)) != 0 for root in answer):
                    raise ValueError('A proposed root failed substitution into the original equation.')
                result['check'] = 'Each returned root was substituted into the original equation.'
        elif operation == 'limit':
            point = parse(task.get('point', '0'))
            direction = task.get('direction', 'both')
            if direction not in ('both', '+', '-'):
                raise ValueError('Direction must be both, +, or -.')
            if point in (sp.oo, -sp.oo) or direction != 'both':
                answer = sp.limit(expression, variable, point, dir='+' if direction == 'both' else direction)
            else:
                left = sp.limit(expression, variable, point, dir='-')
                right = sp.limit(expression, variable, point, dir='+')
                if left != right:
                    result.update(result='The two-sided limit does not exist.', latex=r'\text{Two-sided limit does not exist}',
                                  check=f'Left limit: {text(left)}; right limit: {text(right)}.')
                    return result
                answer = right
        elif operation == 'sum':
            index = sp.Symbol(variable_name, integer=True)
            lower, upper = parse(task.get('lower', '1')), parse(task.get('upper', 'oo'))
            if any(bound not in (sp.oo, -sp.oo) and (bound.is_integer is not True or abs(bound) > 1000)
                   for bound in (lower, upper)):
                raise ValueError('Use integer summation bounds up to 1000 or infinity.')
            answer = sp.summation(expression.subs(variable, index), (index, lower, upper))
        else:
            raise ValueError('Unsupported symbolic operation.')
        if answer.has(sp.Integral, sp.Derivative, sp.Sum, sp.Limit, sp.ConditionSet):
            result['status'] = 'unevaluated'
            result['check'] = 'The symbolic engine did not resolve this calculation; do not label it checked.'
    if hasattr(answer, 'has') and answer.has(sp.zoo, sp.nan):
        raise ValueError('The calculation is undefined.')
    result['result'] = text(answer)
    if denominators:
        result['restrictions'] = [text(value) + ' != 0' for value in denominators]
    result['latex'] = sp.latex(answer)
    if isinstance(answer, sp.Expr) and answer.is_number and answer.is_finite:
        result['approximation'] = str(sp.N(answer, 15))
    if len(result['latex']) > 10000:
        raise ValueError('Result is too large to display.')
    return result


def run_calculations(tasks, timeout=5):
    """One isolated worker per request; process/CPU/memory limits bound expensive algebra."""
    if not isinstance(tasks, list) or len(tasks) > MAX_TASKS:
        return [{'status': 'unavailable', 'error': 'At most three symbolic calculations are supported.'}]
    if not tasks:
        return []
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(tasks), capture_output=True, text=True,
            timeout=timeout, check=False,
            env={'PATH': os.defpath, 'PYTHONNOUSERSITE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
        )
        if process.returncode != 0 or len(process.stdout) > 50000:
            raise ValueError('Worker did not produce a bounded result.')
        return json.loads(process.stdout)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return [{'status': 'unavailable', 'error': 'Symbolic checking reached its resource limit; no calculation was certified.'}]


def _worker():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    except (ImportError, ValueError, OSError):
        # Parent wall-clock timeout remains enforced on platforms without rlimits.
        pass
    tasks = json.loads(sys.stdin.read(12000))
    if not isinstance(tasks, list) or len(tasks) > MAX_TASKS:
        raise ValueError('Invalid task count.')
    results = []
    for task in tasks:
        try:
            results.append(_calculate(task))
        except Exception as error:
            results.append({'operation': task.get('operation') if isinstance(task, dict) else None,
                            'status': 'unavailable', 'error': str(error)[:220]})
    print(json.dumps(results))


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
