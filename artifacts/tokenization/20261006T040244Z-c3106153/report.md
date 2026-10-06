# Multilingual Tokenization Benchmark

**Experiment:** `multilingual-tokenization`

**Tokenizer:** `Qwen/Qwen2.5-1.5B-Instruct`

## Tokenization summary

| Language | Samples | Tokens | Tok/Char Mean | Tok/Char P95 | Tok/Word Mean | Tok/Byte Mean | English-relative tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 20 | 191 | 0.1688 | 0.1906 | 1.2342 | 0.1688 | 1.000x |
| Hindi | 20 | 1083 | 0.9181 | 0.9500 | 5.1812 | 0.3444 | 5.670x |
| Telugu | 20 | 1796 | 1.5095 | 1.5579 | 13.4640 | 0.5447 | 9.403x |
| Tamil | 20 | 1431 | 1.1251 | 1.1771 | 11.2198 | 0.4025 | 7.492x |
| Malayalam | 20 | 1747 | 1.3235 | 1.4000 | 13.7342 | 0.4724 | 9.147x |
| Odia | 20 | 2207 | 2.0356 | 2.0959 | 14.7577 | 0.7383 | 11.555x |

## Tokenizer metadata

- Tokenizer class: `Qwen2Tokenizer`
- Fast tokenizer: `True`
- Vocabulary size: `151665`
- Model max length metadata: `131072`
- Special tokens map: `{'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>'}`

## Interpretation

The English-relative value compares each language's total token count against the English corpus total.

These measurements describe this controlled corpus. They should not be interpreted as universal language-level tokenization ratios.

Tokenization latency is recorded at sample level, but these short sample timings are not treated as a production latency benchmark.
