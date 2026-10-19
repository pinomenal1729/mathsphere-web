"""Gold mathematical examples and adversarial input checks; no provider is needed."""
import unittest
from math_tools import run_calculations
from math_tutor import retrieve_references, decode_plan, prepare_context


class MathematicalAccuracyTests(unittest.TestCase):
    def calculate(self, task):
        result = run_calculations([task])[0]
        self.assertEqual(result['status'], 'computed', result)
        return result

    def test_exact_arithmetic_and_decimal_rationals(self):
        self.assertEqual(self.calculate({'operation':'evaluate','expression':'0.1+0.2'})['result'], '3/10')
        self.assertEqual(self.calculate({'operation':'evaluate','expression':'0.100000000000000001+0.2'})['result'], '300000000000000001/1000000000000000000')

    def test_chain_rule(self):
        self.assertEqual(self.calculate({'operation':'differentiate','expression':'sin(x^2)'})['result'], '2*x*cos(x**2)')

    def test_derivative_at_a_point(self):
        self.assertEqual(self.calculate({'operation':'differentiate','expression':'x^3','at':'2'})['result'], '12')

    def test_cusp_derivative_is_not_sign_zero(self):
        result = self.calculate({'operation':'differentiate','expression':'abs(x)','at':'0'})
        self.assertIn('does not exist', result['result'])

    def test_definite_integral(self):
        self.assertEqual(self.calculate({'operation':'integrate','expression':'x^2','lower':'0','upper':'1'})['result'], '1/3')

    def test_indefinite_integral_includes_constant_metadata(self):
        result = self.calculate({'operation':'integrate','expression':'x^2'})
        self.assertEqual(result['result'], 'x**3/3')
        self.assertTrue(result['constant_of_integration'])
        self.assertIn('reproduces', result['check'])

    def test_real_log_antiderivative_keeps_absolute_value(self):
        self.assertEqual(self.calculate({'operation':'integrate','expression':'1/x'})['result'], 'log(Abs(x))')

    def test_quadratic_has_both_real_roots(self):
        self.assertEqual(self.calculate({'operation':'solve','expression':'x^2=4'})['result'], '{-2, 2}')

    def test_real_and_complex_solution_domains_are_distinct(self):
        results = run_calculations([
            {'operation':'solve','expression':'x^2+1=0'},
            {'operation':'solve','expression':'x^2+1=0','domain':'complex'},
        ])
        self.assertEqual(results[0]['result'], 'EmptySet')
        self.assertEqual(results[1]['result'], '{-I, I}')

    def test_cancelling_denominator_does_not_create_a_root(self):
        result = self.calculate({'operation':'solve','expression':'(x^2-1)/(x-1)=2'})
        self.assertEqual(result['result'], 'EmptySet')
        self.assertIn('x - 1 != 0', result['restrictions'])

    def test_identity_solution_preserves_excluded_zero(self):
        result = self.calculate({'operation':'solve','expression':'x/x=1'})
        self.assertNotEqual(result['result'], 'Reals')
        self.assertIn('x != 0', result['restrictions'])

    def test_sqrt_square_does_not_become_x(self):
        self.assertEqual(self.calculate({'operation':'simplify','expression':'sqrt(x^2)'})['result'], 'Abs(x)')

    def test_two_sided_pole_limit_does_not_exist(self):
        result = self.calculate({'operation':'limit','expression':'1/x','point':'0'})
        self.assertIn('does not exist', result['result'])

    def test_sine_limit_and_one_sided_limit(self):
        results=run_calculations([
            {'operation':'limit','expression':'sin(x)/x','point':'0'},
            {'operation':'limit','expression':'1/x','point':'0','direction':'+'},
        ])
        self.assertEqual(results[0]['result'],'1')
        self.assertEqual(results[1]['result'],'oo')

    def test_determinant_and_rank(self):
        results=run_calculations([
            {'operation':'determinant','matrix':[['2','1'],['3','4']]},
            {'operation':'rank','matrix':[['1','2'],['2','4']]},
        ])
        self.assertEqual(results[0]['result'],'5')
        self.assertEqual(results[1]['result'],'1')

    def test_inverse_is_multiplied_back(self):
        result=self.calculate({'operation':'inverse','matrix':[['1','2'],['3','4']]})
        self.assertEqual(result['result'],'Matrix([[-2, 1], [3/2, -1/2]])')
        self.assertIn('identity', result['check'])

    def test_eigenvalues_include_multiplicity(self):
        result=self.calculate({'operation':'eigenvalues','matrix':[['2','0'],['0','2']]})
        self.assertEqual(result['result'],'{2: 2}')

    def test_infinite_geometric_series(self):
        result=self.calculate({'operation':'sum','expression':'(1/2)^n','variable':'n','lower':'0','upper':'oo'})
        self.assertEqual(result['result'],'2')

    def test_undefined_and_singular_inputs_are_not_certified(self):
        for task in [
            {'operation':'evaluate','expression':'1/0'},
            {'operation':'evaluate','expression':'sqrt(-1)'},
            {'operation':'evaluate','expression':'0^0'},
            {'operation':'integrate','expression':'1/x','lower':'-1','upper':'1'},
            {'operation':'limit','expression':'abs(z)/sqrt(z^2)','variable':'z','point':'0','domain':'complex'},
            {'operation':'differentiate','expression':'abs(z)^2','variable':'z','at':'1','domain':'complex'},
            {'operation':'differentiate','expression':'log(x)','at':'-1'},
            {'operation':'inverse','matrix':[['1','2'],['2','4']]},
        ]:
            with self.subTest(task=task):
                self.assertEqual(run_calculations([task])[0]['status'], 'unavailable')

    def test_python_execution_and_oversized_inputs_are_rejected(self):
        expressions=["__import__('os').getcwd()", '(1).__class__', '[x for x in (1,2)]',
                     'lambda x: x', '2**1000000', 'x+'*300+'1']
        for expression in expressions:
            with self.subTest(expression=expression[:40]):
                self.assertEqual(run_calculations([{'operation':'evaluate','expression':expression}])[0]['status'],'unavailable')

    def test_unsupported_operations_and_task_limits_fail_explicitly(self):
        self.assertEqual(run_calculations([{'operation':'execute','expression':'1'}])[0]['status'],'unavailable')
        self.assertEqual(run_calculations([{}]*4)[0]['status'],'unavailable')


class ReferenceAndPlanningTests(unittest.TestCase):
    def test_mean_value_references_retain_hypotheses(self):
        notes=retrieve_references('Explain the mean value theorem')
        self.assertEqual(notes[0]['id'],'CALC-MVT')
        self.assertIn('continuous on [a,b]',notes[0]['content'])

    def test_matrix_inverse_retrieval(self):
        notes=retrieve_references('Why does a singular matrix not have an inverse?')
        self.assertIn('LIN-INVERSE',[n['id'] for n in notes])

    def test_non_math_greeting_has_no_invented_sources(self):
        self.assertEqual(retrieve_references('hello there'),[])

    def test_invalid_plans_do_not_generate_claimed_checks(self):
        for text in ['not JSON','[]','{"calculations":[1]}','{"calculations":[{},{},{},{}]}']:
            self.assertEqual(decode_plan(text),([],''))

    def test_ambiguity_prevents_calculation(self):
        tasks,question=decode_plan('{"calculations":[{"operation":"evaluate","expression":"2+2"}],"clarification":"What are the bounds?"}')
        self.assertEqual(tasks,[])
        self.assertIn('bounds',question)

    def test_no_plan_means_no_verification(self):
        _,verification,_=prepare_context('Explain compactness')
        self.assertEqual(verification['status'],'not_checked')
        self.assertEqual(verification['calculations'],[])


if __name__=='__main__':
    unittest.main()
