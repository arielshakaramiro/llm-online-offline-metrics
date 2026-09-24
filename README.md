# LLM Evaluation: Offline Metrics & Online Metrics Dashboard

Two complementary notebooks on evaluating LLM applications: one measures
answer quality against a fixed dataset (**offline**), the other measures
real user interaction in a running chat service (**online**). Together
they cover both halves of the same question: "is this LLM's output good?"

| | Notebook | What it measures |
|---|---|---|
| Offline | `Offline_Evaluation_LLM.ipynb` | Exact match, token-level F1, fuzzy similarity, BLEU, ROUGE, perplexity, semantic similarity |
| Online | `Online_Metrics_LLM_with_MySQL.ipynb` | Reaction rate (like/dislike/regenerate), latency, cache hit behavior, daily trend |

All numbers in this README come from actual notebook runs, not estimates.

## Part 1 — Offline metrics

A single 4-case dataset (gold answer vs. model prediction) is scored by
every metric, with consistent text normalization across all of them, so
the differences between metrics reflect how each one actually works, not
inconsistent preprocessing.

| Case | EM | F1 (token) | Fuzzy | BLEU | ROUGE-L | Cosine similarity |
|---|---|---|---|---|---|---|
| 1. Capitalization only | 1 | 1.000 | 97.0 | 1.000 | 1.000 | 0.998 |
| 2. Missing + extra word | 0 | 0.833 | 83.3 | 0.254 | 0.833 | 0.869 |
| 3. Paraphrase (correct answer) | 0 | 0.727 | 44.7 | 0.103 | 0.545 | 0.993 |
| 4. Wrong fact (hallucination) | 0 | 0.800 | 86.6 | 0.669 | 0.800 | 0.682 |

Case 3 and 4 are the interesting pair: a *correct* answer worded
differently (case 3) scores low on every lexical metric (BLEU 0.103,
ROUGE-L 0.545), while a *factually wrong* answer that happens to share
most of its words with the gold answer (case 4) scores deceptively high
(BLEU 0.669). Sentence-embedding cosine similarity reverses that ranking
(0.993 vs. 0.682) — it catches the paraphrase's meaning and is noticeably
less fooled by the wrong-fact case, though it is not a substitute for an
explicit fact-check.

**Perplexity** (GPT-2 fine-tuned for Indonesian,
[`flax-community/gpt2-small-indonesian`](https://huggingface.co/flax-community/gpt2-small-indonesian),
rather than the English-only default GPT-2, so the score reflects the
sentence and not a language mismatch):

| Sentence | Perplexity |
|---|---|
| "Jakarta adalah ibu kota Indonesia." | 17.29 |
| "Jakarta merupakan pusat pemerintaha Indonesia." (typo: *pemerintaha*) | 145.16 |

**Semantic search** (multilingual sentence embeddings + FAISS,
`IndexFlatIP` on normalized vectors so it matches cosine similarity
exactly) correctly ranks "harga logam mulia hari ini" closest to
"Harga emas hari ini mengalami kenaikan" (0.674) and "Emas Antam naik dua
persen" (0.450), and lowest for the unrelated "Nasabah dapat mengajukan
pinjaman KCA" (−0.007).

## Part 2 — Online metrics dashboard

A local LLM chat service (FastAPI + llama.cpp) with MySQL-backed response
caching, a reaction-feedback loop (like / dislike / regenerate), and a
Streamlit dashboard for reaction rate, per-response latency, and daily
trends — the kind of metric that can only be collected from real usage,
not a fixed dataset.

- **Response caching**: fuzzy-matched cache (RapidFuzz) so a repeated or
  near-duplicate question can skip inference.
- **Feedback loop**: `like` / `dislike` / `regenerate` reactions are logged
  per response, and a `dislike` or `regenerate` invalidates that response's
  cache entry so it isn't served again.
- **Metrics, not just counts**: reaction *rate* (reactions ÷ total chats in
  the same window), not a bare count, plus per-response latency.

### Architecture

```
+-------------+      like/dislike/regenerate      +--------------+
|  Streamlit  | ---------------------------------> |   FastAPI    |
|  dashboard  | <---- /total-reactions, etc. ------ |  (port 8000) |
+-------------+         (via localhost)             +------+-------+
       |                                                    |
   ngrok tunnel                                       llama-cpp-python
  (dashboard only,                                    (local GGUF model)
   public access)                                            |
                                                        +------v-------+
                                                        |    MySQL     |
                                                        | cached_chat  |
                                                        | chat_history |
                                                        |  analytics   |
                                                        +--------------+
```

The dashboard calls the FastAPI backend over `localhost`, not through a
second public tunnel. Free-tier ngrok accounts in testing only kept one
public HTTP tunnel routable at a time — running two tunnels concurrently
(one for the API, one for the dashboard) caused requests to silently land
on the wrong service. Keeping that traffic on localhost (both processes
run in the same environment) sidesteps it entirely; the public tunnel is
only needed for the dashboard so it can be opened from outside the
environment it runs in.

### Model

[`gmonsoon/llama3-8b-cpt-sahabatai-v1-instruct-GGUF`](https://huggingface.co/gmonsoon/llama3-8b-cpt-sahabatai-v1-instruct-GGUF)
(Q4_K_M quantization, ~4.9 GB) — a Llama 3 8B Instruct model continually
pretrained and instruction-tuned for Indonesian (plus Javanese, Sundanese,
and English) by GoTo Group and Indosat Ooredoo Hutchison under the Llama 3
Community License. Downloaded from Hugging Face at runtime, not bundled
in this repo.

### Setup

Requires: Google Colab, a free [freedb.tech](https://freedb.tech) MySQL
database, and a free [ngrok](https://ngrok.com) account.

1. Create a MySQL database on freedb.tech and note its host, port,
   database name, username, and password.
2. In Colab, add these as Secrets (key icon in the sidebar): `DB_HOST`,
   `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `NGROK_AUTHTOKEN`. A
   `getpass` prompt is the fallback if a secret isn't set.
3. Run the notebook top to bottom. The model downloads once and is cached
   to both Google Drive (across sessions) and local Colab disk (loading
   directly from a Drive mount is dramatically slower than loading from
   local disk, since `llama-cpp-python` reads the file with a pattern that
   is expensive over Drive's FUSE mount).
4. The dashboard cell disconnects the API's own public tunnel before
   opening the dashboard's (see Architecture above) and prints a public
   URL for the Streamlit dashboard.

### Verified results

22 chat responses were logged in the run below, across a repeated
question ("Siapa presiden pertama Indonesia?"), a second, unrelated
question ("Siapa presiden ketiga Indonesia?"), and one each of a `like`,
two `dislike`, and one `regenerate` reaction.

| Reaction type | Total | Rate (of 22 chats) |
|---|---|---|
| Dislike | 2 | 9.1% |
| Like | 1 | 4.5% |
| Regenerate | 1 | 4.5% |

<p align="center">
  <img src="images/dashboard-dislike-rate.png" width="32%" alt="Dashboard showing dislike total and rate">
  <img src="images/dashboard-like-rate.png" width="32%" alt="Dashboard showing like total and rate">
  <img src="images/dashboard-regenerate-rate.png" width="32%" alt="Dashboard showing regenerate total and rate">
</p>

**Cache invalidation:** after the `dislike` and `regenerate` reactions
above, the corresponding cache entries were confirmed removed from
`cached_chat` -- a repeat of the same question after a negative reaction
goes back to the model instead of replaying the disliked answer.

**Cache matching threshold:** the fuzzy cache started at a similarity
threshold of 80 using RapidFuzz's `partial_ratio` (substring-style
matching), which scored an unrelated question ("Siapa presiden ketiga
Indonesia?") at 88.2 against a cached "presiden pertama" entry -- a false
cache hit. Switching to `fuzz.ratio` (whole-string comparison) brought
that same pair down to 85.7, still above a threshold of 85, so the
threshold was raised to 92.

**Known limitation:** even at 92, character-level fuzzy matching cannot
reliably separate two questions that differ only in a short number if the
rest of the sentence is identical. A constructed test pair -- "siapa
presiden ke-2 di konoha" vs. "siapa presiden ke-4 di konoha" -- scored
about 96.5 with `fuzz.ratio`, close enough to genuine duplicates (97-99)
that no single threshold cleanly separates the two cases. A stricter
threshold reduces this risk but also reduces how often genuine
near-duplicates hit the cache; this is a precision/recall trade-off, not
something a threshold alone resolves. A more robust fix (comparing key
entities or numbers explicitly, or a semantic-similarity check) is future
work.

## License

MIT -- see [LICENSE](LICENSE). Models used are distributed separately by
their authors under their own licenses (see the Model sections above).
