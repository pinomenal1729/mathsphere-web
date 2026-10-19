# Mathematical rigor and explainability

This change adds reference grounding and independent symbolic calculations. It does **not** retrain or fine-tune Groq or Gemini, import textbook training sets, or guarantee that every generated answer is correct.

## Answer workflow

For text chat, the tutor retrieves up to three relevant notes from `data/math_reference.json`. These are 27 locally authored mathematical reference notes covering calculus, analysis, algebra, linear algebra, probability, differential equations, complex analysis, and numerical reasoning. They include hypotheses and common counterexamples. Further-reading links identify background resources; they are not claims that those sources were downloaded or used to train the models. The notes are separate from the question bank and are never labelled official exam questions.

For questions that appear to need a calculation, a short structured model request proposes up to three calculator tasks. Ambiguous input can produce a clarification question rather than a guessed calculation. The server validates and executes supported tasks in an isolated SymPy worker, then supplies the results and retrieved notes to the answer model. Malformed plans and unsupported calculations never become successful verification claims. Conceptual explanations and proofs can use reference notes without making an unnecessary calculator request.

The tutor prompt requires explicit assumptions, domains, theorem hypotheses, units, exact values before rounding, original-equation substitution, branch restrictions, convergence checks, and constants of integration. The explanation styles are step by step (default), concise, and exam-ready. A shared 90-second generation budget bounds the multi-stage workflow in addition to existing provider timeouts and fallback limits.

## Supported symbolic operations

- Exact arithmetic and simplification.
- First through third derivatives, including finite point evaluations checked using one-sided difference quotients.
- Definite and indefinite integrals; indefinite results explicitly carry a `+C` flag.
- One-variable real or complex equation solving, with original denominator restrictions retained and finite roots substituted back.
- Real one-sided and two-sided limits.
- Finite or infinite symbolic sums.
- Determinants, rank, inverses, and eigenvalue multiplicities for matrices up to 5 × 5. Inverses are multiplied back against the original matrix.

The parser accepts a small mathematical AST and an explicit list of functions/symbols. It does not call Python `eval`, SymPy's string evaluator, or execute model-generated code. Expressions, tree size, nesting, powers, task count, and output size are bounded. Workers receive no API-key environment variables. Parent wall-clock limits and Linux CPU/address-space limits prevent expensive algebra from blocking the web process indefinitely.

The calculator can still return unevaluated, unavailable, or conditional results. Complex limits and pointwise complex derivatives require all-path/holomorphicity analysis and are not certified by this real-axis checker. General proof verification, diagrams, statistical modelling assumptions, word-problem interpretation, and image transcription remain outside its certification scope. Symbolic parameter ranks are generic and can change at exceptional parameter values. Engine results must be interpreted on their stated domains; not all branch or parameter conditions are automatically resolved.

## What users see

Successful calculations appear in expandable **Symbolic calculations** cards with their input, result, assumptions, domain restrictions, and any extra check. They remain available when a saved conversation is reopened. **Reference notes** lists the local notes supplied to the model and any background-reading links.

The interface explicitly states that a calculation check does not certify the interpretation or the full generated explanation. If no calculation completed, it says so. Image answers use the stronger teaching prompt and relevant textual references, but image interpretation and solutions are not independently checked by this text-calculation pipeline.

## Extending the reference collection

Each JSON note has a stable `id`, `title`, `topic`, `keywords`, and `content`, with optional `further_reading` containing a public HTTPS title/URL. Add only material you have permission to use. Include assumptions and counterexamples, avoid duplicates, and add a retrieval test for the intended queries. Editing these notes changes retrieval context, not model weights. Increasing note count alone does not establish higher accuracy.

## Evaluation

Run:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The gold mathematical cases include exact decimal arithmetic, the chain rule, cusp derivatives, definite integrals, real logarithmic antiderivatives, extraneous roots after cancellation, real versus complex roots, limits from both sides, singular matrices, inverse identities, eigenvalue multiplicity, and a geometric series. Unsupported or dangerous inputs must remain unverified. Provider integration tests use actual SDKs with in-memory HTTP responses; they check that computed results reach the final model context, JSON planning is requested correctly, ambiguity is handled, and malformed plans do not claim checks.

The browser suite verifies evidence rendering, persisted metadata, explanation styles, saved conversations, escaping, and mobile overflow. These are regression checks, not a measured live-model accuracy benchmark. Real model planning and narrative quality still require representative live questions after deployment; an automated pass does not establish universal correctness.
