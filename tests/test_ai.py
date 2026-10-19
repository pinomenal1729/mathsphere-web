"""Exercise real provider SDKs through mock HTTP transports; no live keys/calls."""
import base64
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from google import genai
from google.genai import types
from groq import Groq


class AIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='mathsphere-ai-')
        cls.root = Path(cls.workspace.name)
        source = Path(__file__).resolve().parents[1] / 'app.py'
        (cls.root / 'app.py').symlink_to(source)
        spec = importlib.util.spec_from_file_location('mathsphere_ai_test', cls.root / 'app.py')
        cls.backend = importlib.util.module_from_spec(spec)
        cwd = Path.cwd()
        try:
            os.chdir(cls.root)
            with patch.dict(os.environ, {'GROQ_API_KEY': '', 'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': ''}):
                spec.loader.exec_module(cls.backend)
        finally:
            os.chdir(cwd)
        cls.backend.app.config['TESTING'] = True

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.app = self.backend
        self.patches = [
            patch.object(self.app, 'GROQ_AVAILABLE', False),
            patch.object(self.app, 'GEMINI_AVAILABLE', False),
            patch.object(self.app, 'groq_client', None),
            patch.object(self.app, 'gemini_client', None),
            patch.object(self.app, 'ACTIVE_AI_MODELS', {
                'groq': self.app.GROQ_MODEL, 'gemini': self.app.GEMINI_MODEL,
            }),
            patch.object(self.app, 'AI_STATUS', {
                name: {'configured': False, 'client_ready': False, 'last_error_code': None}
                for name in ('groq', 'gemini')
            }),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = self.app.app.test_client()
        self.provider_requests = []
        self.model_requests = []
        self.gemini_models = []

    def groq(self, status=200, models=(), discovery_status=200, content='The answer is 4.'):
        def respond(request):
            if request.method == 'GET':
                self.model_requests.append('groq')
                return httpx.Response(discovery_status, json={'data': [
                    {'id': name, 'object': 'model', 'created': 0, 'owned_by': 'test'}
                    for name in models
                ]})
            self.provider_requests.append(('groq', json.loads(request.content)))
            current_status = status(request) if callable(status) else status
            if current_status == 200:
                return httpx.Response(200, json={
                    'id': 'test-completion', 'object': 'chat.completion', 'created': 0,
                    'model': self.app.GROQ_MODEL,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content(request) if callable(content) else content}, 'finish_reason': 'stop'}],
                })
            return httpx.Response(current_status, json={'error': {
                'message': 'private-provider-detail', 'type': 'request_error',
            }})
        # This placeholder is used only by an in-memory transport, never a server.
        client = Groq(api_key='unit-test-placeholder', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        self.app.groq_client = client
        self.app.GROQ_AVAILABLE = True
        self.app.AI_STATUS['groq'].update(configured=True, client_ready=True)

    def gemini(self, status=200, models=(), empty=False, content='The answer is 4.'):
        def respond(request):
            if request.method == 'GET':
                self.model_requests.append('gemini')
                return httpx.Response(200, json={'models': [
                    {'name': 'models/' + name, 'supportedGenerationMethods': ['generateContent']}
                    for name in models
                ]})
            self.provider_requests.append(('gemini', json.loads(request.content)))
            self.gemini_models.append(request.url.path.split('/models/')[-1].split(':')[0])
            current_status = status(request) if callable(status) else status
            if current_status == 200:
                return httpx.Response(200, json={'candidates': [{
                    'content': {'role': 'model', 'parts': [] if empty else [{'text': content(request) if callable(content) else content}]},
                    'finishReason': 'STOP',
                }]})
            return httpx.Response(current_status, json={'error': {
                'code': current_status, 'message': 'private-provider-detail', 'status': 'FAILED_PRECONDITION',
            }})
        client = genai.Client(api_key='unit-test-placeholder', http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(respond)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        self.addCleanup(client.close)
        self.app.gemini_client = client
        self.app.GEMINI_AVAILABLE = True
        self.app.AI_STATUS['gemini'].update(configured=True, client_ready=True)

    def chat(self, **payload):
        return self.client.post('/api/chat', json={'message': 'Explain addition briefly.', **payload})

    def test_groq_sdk_success(self):
        self.groq()
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['answer'], 'The answer is 4.')
        self.assertEqual(self.provider_requests[0][1]['model'], self.app.GROQ_MODEL)

    def test_gemini_sdk_success(self):
        self.gemini()
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['answer'], 'The answer is 4.')
        body = self.provider_requests[0][1]
        self.assertIn('systemInstruction', body)
        self.assertEqual(body['contents'][0]['role'], 'user')

    def test_groq_failure_falls_back_to_gemini(self):
        self.groq(429)
        self.gemini()
        response = self.chat(messages=[
            {'role': 'user', 'content': 'Explain addition'},
            {'role': 'assistant', 'content': 'We combine quantities.'},
            {'role': 'user', 'content': 'How does that work?'},
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual([name for name, _ in self.provider_requests], ['groq', 'gemini'])
        self.assertEqual(self.provider_requests[-1][1]['contents'][1]['role'], 'model')

    def test_missing_keys_are_distinct_from_provider_failures(self):
        response = self.chat()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json['code'], 'ai_not_configured')

    def test_initialization_failure_is_not_reported_as_missing_key(self):
        self.app.AI_STATUS['groq']['configured'] = True
        response = self.chat()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json['code'], 'ai_initialization_failed')

    def test_authentication_failure_is_safe_and_actionable(self):
        self.groq(401)
        with self.assertLogs(level='WARNING') as captured:
            response = self.chat()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json['code'], 'ai_authentication_failed')
        self.assertNotIn('private-provider-detail', response.get_data(as_text=True))
        self.assertNotIn('private-provider-detail', '\n'.join(captured.output))
        self.assertNotIn('unit-test-placeholder', response.get_data(as_text=True))

    def test_quota_failure_remains_a_quota_error(self):
        self.gemini(429)
        response = self.chat()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json['code'], 'ai_quota_exceeded')
        self.assertIn('quota', response.json['error'])

    def test_missing_model_has_a_model_error(self):
        self.gemini(404)
        response = self.chat()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json['code'], 'ai_model_unavailable')
        self.assertIn('GEMINI_MODEL', response.json['error'])

    def test_both_failed_providers_are_identified(self):
        self.groq(401)
        self.gemini(429)
        response = self.chat()
        self.assertEqual(len(response.json['providers']), 2)
        self.assertEqual({item['provider'] for item in response.json['providers']}, {'groq', 'gemini'})
        self.assertNotIn('No AI service available', response.get_data(as_text=True))

    def test_image_uses_modern_gemini_parts(self):
        self.gemini()
        encoded = base64.b64encode(b'test-image-bytes').decode()
        response = self.chat(image_b64=encoded, image_type='image/png')
        self.assertEqual(response.status_code, 200)
        parts = self.provider_requests[-1][1]['contents'][0]['parts']
        image = next(part['inlineData'] for part in parts if 'inlineData' in part)
        self.assertEqual(image.get('mimeType', image.get('mime_type')), 'image/png')
        self.assertEqual(base64.b64decode(image['data']), b'test-image-bytes')

    def test_image_failure_preserves_provider_error(self):
        self.gemini(429)
        response = self.chat(image_b64='aGVsbG8=', image_type='image/png')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json['code'], 'ai_quota_exceeded')

    def test_other_ai_tools_preserve_structured_failures(self):
        self.gemini(404)
        for route, payload in [
            ('/api/formula', {'topic': 'Calculus'}),
            ('/api/quiz/generate', {'topic': 'Calculus', 'count': 1}),
            ('/api/theorem/prove', {'theorem': 'Pythagorean theorem'}),
            ('/api/mathematician', {'name': 'Noether'}),
            ('/api/projects/generate', {'topic': 'Calculus'}),
        ]:
            with self.subTest(route=route):
                response = self.client.post(route, json=payload)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json['code'], 'ai_model_unavailable')

    def test_graph_points_survive_ai_outage(self):
        self.gemini(429)
        response = self.client.post('/api/graph', json={
            'expression': 'x^2', 'x_min': -2, 'x_max': 2, 'points': 100,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['point_count'], 101)
        self.assertEqual(response.json['analysis'], '')
        self.assertIn('quota', response.json['analysis_error'])
        for point in response.json['points']:
            self.assertAlmostEqual(point['y'], point['x'] ** 2, places=5)

    def test_health_exposes_configuration_not_secrets(self):
        self.groq()
        self.gemini()
        response = self.client.get('/api/health')
        self.assertTrue(response.json['ai']['providers']['gemini']['client_ready'])
        self.assertEqual(response.json['ai']['models']['gemini'], 'gemini-flash-latest')
        self.assertNotIn('unit-test-placeholder', response.get_data(as_text=True))

    def test_retired_groq_model_recovers_and_caches_success(self):
        self.app.ACTIVE_AI_MODELS['groq'] = 'retired-model'
        self.groq(
            lambda request: 404 if json.loads(request.content)['model'] == 'retired-model' else 200,
            models=['whisper-large-v3', 'openai/gpt-oss-20b'],
        )
        self.assertEqual(self.chat().status_code, 200)
        self.assertEqual([body['model'] for _, body in self.provider_requests], ['retired-model', 'openai/gpt-oss-20b'])
        self.assertEqual(self.model_requests, ['groq'])
        self.assertEqual(self.chat().status_code, 200)
        self.assertEqual(self.model_requests, ['groq'])
        health = self.client.get('/api/health').json
        self.assertEqual(health['ai']['active_models']['groq'], 'openai/gpt-oss-20b')

    def test_discovery_never_guesses_unlisted_models(self):
        self.groq(404, models=['whisper-large-v3', 'meta-llama/llama-guard-4'])
        response = self.chat()
        self.assertEqual(response.json['code'], 'ai_model_unavailable')
        self.assertEqual(len(self.provider_requests), 1)

    def test_model_discovery_failure_keeps_original_error(self):
        self.groq(404, discovery_status=503)
        response = self.chat()
        self.assertEqual(response.json['code'], 'ai_model_unavailable')
        self.assertEqual(len(self.provider_requests), 1)

    def test_authentication_and_quota_failures_do_not_probe_models(self):
        self.groq(401, models=['openai/gpt-oss-20b'])
        self.gemini(429, models=['gemini-3.8-flash'])
        response = self.chat()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.model_requests, [])

    def test_overloaded_gemini_model_uses_listed_alternative(self):
        self.gemini(
            lambda request: 503 if self.app.GEMINI_MODEL in request.url.path else 200,
            models=[self.app.GEMINI_MODEL, 'gemini-3.8-flash', 'gemini-3.8-flash-tts'],
        )
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.gemini_models, [self.app.GEMINI_MODEL, 'gemini-3.8-flash'])
        self.assertEqual(self.model_requests, ['gemini'])

    def test_gemini_generation_attempts_are_bounded(self):
        self.gemini(503, models=['gemini-3.8-flash', 'gemini-2.5-flash'])
        response = self.chat()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(self.provider_requests), 2)
        self.assertEqual(self.model_requests, ['gemini'])
        self.assertIn('HTTP 503', response.json['error'])

    def test_no_alternative_keeps_service_outage_explicit(self):
        self.gemini(503)
        response = self.chat()
        self.assertEqual(response.json['code'], 'ai_provider_unavailable')
        self.assertIn('HTTP 503', response.json['error'])
        self.assertEqual(len(self.provider_requests), 1)

    def test_empty_response_is_not_a_network_outage(self):
        self.gemini(empty=True)
        response = self.chat()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json['code'], 'ai_empty_response')
        self.assertEqual(self.model_requests, [])

    def test_timeout_is_identified_without_leaking_provider_details(self):
        error = httpx.ReadTimeout('private-provider-detail')
        failure = self.app.provider_failure('gemini', error)
        self.assertEqual(failure.code, 'ai_timeout')
        self.assertNotIn('private-provider-detail', str(failure))

    def test_chat_computes_before_generating_a_grounded_explanation(self):
        def response(request):
            system = json.loads(request.content)['messages'][0]['content']
            if 'calculator requests, not an answer' in system:
                return json.dumps({'calculations': [{'operation': 'differentiate', 'expression': 'x^3', 'variable': 'x'}]})
            return '## Answer\n\\(3x^2\\) by the power rule.'
        self.groq(content=response)
        result = self.chat(message='Differentiate x^3').json
        self.assertEqual(result['verification']['status'], 'computed')
        self.assertEqual(result['verification']['calculations'][0]['result'], '3*x**2')
        final_context = self.provider_requests[-1][1]['messages'][0]['content']
        self.assertIn('3*x**2', final_context)
        self.assertIn('CALC-POWER', final_context)
        self.assertEqual(len(self.provider_requests), 2)

    def test_conceptual_question_gets_references_without_fake_verification(self):
        self.groq()
        result = self.chat(message='Explain the mean value theorem').json
        self.assertEqual(result['verification']['status'], 'not_checked')
        self.assertIn('CALC-MVT', [item['id'] for item in result['references']])
        self.assertEqual(len(self.provider_requests), 1)

    def test_ambiguous_question_requests_clarification(self):
        self.groq(content=json.dumps({'calculations': [], 'clarification': 'What is the upper integration bound?'}))
        result = self.chat(message='Calculate the integral from 0 to the missing endpoint').json
        self.assertIn('upper integration bound', result['answer'])
        self.assertEqual(result['verification']['status'], 'not_checked')
        self.assertEqual(len(self.provider_requests), 1)

    def test_invalid_planner_output_does_not_claim_a_check(self):
        self.groq()
        result = self.chat(message='Calculate 2 + 2').json
        self.assertEqual(result['verification']['status'], 'not_checked')
        self.assertEqual(result['verification']['calculations'], [])

    def test_malicious_planned_expression_is_not_executed(self):
        self.groq(content=json.dumps({'calculations': [{'operation': 'evaluate', 'expression': "__import__('os').getcwd()"}]}))
        result = self.chat(message='Calculate 2 + 2').json
        self.assertEqual(result['verification']['status'], 'not_checked')
        self.assertEqual(result['verification']['calculations'][0]['status'], 'unavailable')

    def test_gemini_planning_uses_json_and_exact_integral_context(self):
        def response(request):
            body = json.loads(request.content)
            system = body['systemInstruction']['parts'][0]['text']
            if 'calculator requests, not an answer' in system:
                self.assertEqual(body['generationConfig']['responseMimeType'], 'application/json')
                return json.dumps({'calculations':[{'operation':'integrate','expression':'x^2','lower':'0','upper':'1'}]})
            self.assertIn('1/3', system)
            return 'The exact integral is 1/3.'
        self.gemini(content=response)
        result = self.chat(message='Integrate x^2 from 0 to 1').json
        self.assertEqual(result['verification']['calculations'][0]['result'], '1/3')

    def test_request_budget_prevents_more_network_calls(self):
        self.groq()
        with self.assertRaises(self.app.AIServiceError) as error:
            self.app.ask_ai([{'role':'user','content':'Hello'}], deadline=0)
        self.assertEqual(error.exception.code, 'ai_timeout')
        self.assertEqual(self.provider_requests, [])


if __name__ == '__main__':
    unittest.main()
