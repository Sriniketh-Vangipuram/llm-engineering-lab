# Experiment 005 — Multilingual Tokenization Engineering

## Objective

Measure how `Qwen/Qwen2.5-1.5B-Instruct` tokenizes controlled multilingual text and determine how tokenizer efficiency affects:

- token count
- context utilization
- effective text capacity
- inference compute
- KV-cache requirements
- potential inference cost

The experiment focuses on English, Hindi, Telugu, Tamil, Malayalam, and Odia.

## Hypothesis

A multilingual tokenizer does not necessarily allocate tokens equally efficiently across languages.

If a language requires more tokens to represent the same approximate amount of text, then a fixed model context budget can hold less textual content in that language.

This can affect downstream inference systems through increased prompt token count, prefill computation, KV-cache memory, latency, token-based API cost, and context truncation risk.

## Model / Tokenizer

| Property | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Tokenizer | `Qwen2Tokenizer` |
| Fast tokenizer | Yes |
| Vocabulary size | 151,665 |
| Model context metadata | 131,072 tokens |
| Special EOS token | `<|im_end|>` |
| PAD token | `<|endoftext|>` |

Tokenizer was loaded using Hugging Face Transformers with `trust_remote_code=False`.

## Corpus

The controlled corpus contains 6 languages, 20 sentences per language, and 120 total samples.

Languages:

1. English
2. Hindi
3. Telugu
4. Tamil
5. Malayalam
6. Odia

The sentences cover common LLM engineering concepts such as tokenization, language models, inference, context windows, batching, latency, memory, model serving, and cost.

The corpus is intentionally controlled rather than intended to represent a statistically representative multilingual dataset.

## Experiments

### 1. Tokenization Benchmark

Measured for every sample:

- characters
- UTF-8 bytes
- whitespace-separated words
- tokens
- tokens/character
- tokens/word
- tokens/UTF-8 byte
- tokenization latency
- context utilization
- context-limit overflow

### 2. Token Segmentation Inspection

Individual token IDs and token representations were inspected.

An important observation was that decoding individual byte-level token pieces can produce replacement characters (`�`).

This does **not** imply input corruption. Individual token pieces are not necessarily independently valid UTF-8 sequences.

Therefore individual-token decoding must not be confused with decoding the complete token sequence.

### 3. Encode/Decode Round-Trip Validation

All 120 samples were tested using:

```text
text
  ↓
encode
  ↓
token IDs
  ↓
decode
  ↓
text
```

Result:

```text
English      20/20 PASS
Hindi        20/20 PASS
Telugu       20/20 PASS
Tamil        20/20 PASS
Malayalam    20/20 PASS
Odia         20/20 PASS
```

Total:

```text
120/120 samples passed
0 round-trip failures
```

Therefore the tokenization experiments did not reveal input corruption.

## Normalized Tokenization Results

| Language | Tokens | Tokens/Char | Tokens/Byte | Tokens/Word | Chars/Token |
|---|---:|---:|---:|---:|---:|
| English | 191 | 0.1690 | 0.1690 | 1.2244 | 5.9162 |
| Hindi | 1,083 | 0.9178 | 0.3445 | 5.0845 | 1.0896 |
| Tamil | 1,431 | 1.1224 | 0.4014 | 11.0930 | 0.8910 |
| Malayalam | 1,747 | 1.3245 | 0.4725 | 13.6484 | 0.7550 |
| Telugu | 1,796 | 1.5105 | 0.5451 | 13.3037 | 0.6620 |
| Odia | 2,207 | 2.0360 | 0.7386 | 14.6159 | 0.4912 |

### English-relative token density

| Language | Relative token density |
|---|---:|
| English | 1.00× |
| Hindi | 5.43× |
| Tamil | 6.64× |
| Malayalam | 7.84× |
| Telugu | 8.94× |
| Odia | 12.05× |

These values describe this controlled corpus and tokenizer configuration. They must **not** be interpreted as universal language-level tokenization ratios.

## Context Capacity Experiment

A synthetic token-budget probe was performed at 512, 1,024, 2,048, and 4,096 tokens.

At approximately 4,096 tokens:

| Language | Tokens | Characters | Words |
|---|---:|---:|---:|
| English | 4,089 | 24,873 | 3,375 |
| Hindi | 4,071 | 4,451 | 794 |
| Tamil | 4,030 | 3,601 | 359 |
| Malayalam | 4,019 | 3,025 | 292 |
| Telugu | 4,048 | 2,722 | 306 |
| Odia | 3,983 | 1,972 | 270 |

This demonstrates that a fixed token budget can correspond to substantially different amounts of textual content across languages.

### Important limitation

The context-capacity experiment repeatedly cycles through the controlled corpus to approach the token budget. Therefore it is a synthetic capacity probe rather than a natural-language corpus benchmark.

The result should be interpreted as:

> How much text can this tokenizer represent within a fixed token budget under this controlled corpus?

It should not be interpreted as a measurement of natural-language information density.

## Engineering Interpretation

The experiment establishes a direct relationship between tokenizer efficiency and downstream LLM systems.

```text
Tokenizer efficiency
        ↓
Token count
        ↓
Context utilization
        ↓
Prefill computation
        ↓
KV-cache memory
        ↓
Inference latency
        ↓
Throughput
        ↓
Cost per request
```

For multilingual production systems, tokenizer behavior therefore becomes an infrastructure concern rather than merely a preprocessing detail.

A language that requires substantially more tokens for comparable textual content can experience:

- earlier context exhaustion
- higher prompt-processing cost
- greater KV-cache consumption
- higher memory pressure
- increased latency
- lower effective information capacity per context window
- potentially higher API cost when billing is token-based

## Important Findings

### Finding 1 — Token count is language-dependent

The tokenizer produced substantially different token densities across the six languages.

In this controlled corpus, English was the most token-efficient by character count and Odia was the least token-efficient.

### Finding 2 — Individual token decoding can be misleading

Byte-level token representations may not independently form valid UTF-8 sequences.

Seeing `�` while decoding individual token IDs is therefore insufficient evidence of corrupted input.

Full-sequence round-trip validation is the correct integrity test.

### Finding 3 — Context windows are token-based

A model's context limit is expressed in tokens rather than characters or words.

Therefore the amount of human-readable text that fits into a context window varies significantly with tokenizer efficiency.

### Finding 4 — Tokenization affects systems engineering

Tokenizer behavior propagates into:

- memory requirements
- latency
- throughput
- serving capacity
- cost
- context management

This makes tokenizer analysis relevant to LLM infrastructure design.

## Limitations

1. The corpus contains only 20 sentences per language.
2. The corpus is controlled and synthetic rather than statistically representative.
3. Sentence lengths and word counts are not identical across languages.
4. The languages represent only a small subset of global languages.
5. Results are specific to `Qwen/Qwen2.5-1.5B-Instruct`.
6. Tokenization efficiency can change across tokenizer versions and model families.
7. Token count alone does not measure semantic information density.
8. The context-capacity experiment uses repeated corpus samples and is therefore synthetic.
9. Tokenization latency was measured locally and should not be treated as a production throughput benchmark.

## Reproducibility

From the repository root:

```cmd
python experiments\005_multilingual_tokenization\run_benchmark.py
python experiments\005_multilingual_tokenization\inspect_segmentation.py
python experiments\005_multilingual_tokenization\validate_roundtrip.py
python experiments\005_multilingual_tokenization\compare_aligned.py
python experiments\005_multilingual_tokenization\context_capacity.py
```

## Conclusion

The experiment confirms that tokenizer behavior can create substantial differences in effective context capacity across languages.

The most important lesson is not simply that some languages use more tokens.

The systems-level lesson is:

> A fixed token budget does not represent a fixed amount of textual content.

Therefore tokenizer efficiency must be considered when designing multilingual LLM systems, particularly when optimizing context limits, KV-cache memory, inference latency, throughput, and cost.

This experiment provides the foundation for later work on:

- model architecture
- inference optimization
- KV-cache engineering
- context management
- batching
- serving
- cost optimization

Those topics are intentionally deferred to their corresponding phases of `llm-engineering-lab`.
