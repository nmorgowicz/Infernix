# Held-out roleplay evaluation corpus

`tools/eval/roleplay_heldout` rebuilds a small evaluation corpus of human roleplay-forum text from a
pinned public dataset revision, checks it against the texts a model was calibrated or evaluated on,
and prints a SHA-256 that is compared with the published one. It exists to measure the
full-vocabulary KL of a quantization on prose that was not used to choose or build any arm. The
corpus text is **not** in this repository: it is generated on your machine from the dataset.

It is evaluation only. It is never calibration text, never part of a conversion, and it is not a
substitute for the calibration-disjoint pinned corpora of [Perplexity](perplexity.md).

## Source, licence and attribution

| | |
|---|---|
| Dataset | [`lemonilia/roleplaying-forums-raw`](https://huggingface.co/datasets/lemonilia/roleplaying-forums-raw), one row per forum thread (`thread_url`, `thread_title`, `contents` = raw HTML pages) |
| Revision | `6ba79e71d8266428eee590f634bf77aedead39b9` (`pins.REVISION`; last modified 2025-01-11) |
| Licence | CC-BY-4.0 (the dataset card's `license: cc-by-4.0`; ungated) |
| Author | lemonilia (forum scrapes of Blue Moon Roleplaying Forums and Elliquiy, 2023-03/04, and RPNation, 2025-01) |
| Files | six parquet files, 305 MB, each pinned by size and SHA-256 in `pins.SOURCES` |

Attribution: the forum scrapes were collected and published by **lemonilia** under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The posts themselves are the forum
members' writing. This tool changes the data (HTML is rendered to plain text, posts are filtered and
cut into excerpts) and records that in `provenance.json`; usernames and timestamps are never
rendered. `provenance.json` holds the public thread URLs and titles for audit, no post text.

The raw dataset is used instead of the processed `lemonilia/Roleplay-Forums_2023-04` because the
processed set's card warns that spaces between adjacent HTML tags were lost, which damages prose as
evaluation text; the raw HTML keeps them.

## Reproduce

Python with `lxml` and `pyarrow` (`tools/eval/roleplay_heldout/requirements.txt`; the published hash
was produced with lxml 6.1.2 / libxml2 2.11.9 and pyarrow 25.0.1). From the repository root:

```bash
python -m tools.eval.roleplay_heldout fetch --out <raw>                # six files, SHA-256 verified, resumes by skipping verified files
python -m tools.eval.roleplay_heldout build --raw <raw> --out <corpus> # cleans, selects, writes, prints the removal counts and the corpus SHA-256
python -m tools.eval.roleplay_heldout check --corpus <corpus> \
    --against <calibration.txt> --against <dir-of-earlier-eval-texts> ...
```

`fetch` downloads the pinned files at the pinned revision from `huggingface.co` and discards any file whose
size or SHA-256 differs. `build` verifies the same hashes again and refuses a non-empty output directory.
`check` exits 0 only if every stream matches its recorded hash, no two streams share a text, no stream shares
any 60-character shingle with the `--against` texts (a directory contributes its `*.txt` files), and the corpus
SHA-256 equals the published one.

**Published corpus SHA-256 (default selection):**

```text
61ea33b9e0ac6901078e035b044efc4d8f68ae783d73ae2a7414e84e8d465cc5
```

(`pins.PUBLISHED_CORPUS_SHA256`.) A corpus built with another `--seed`, `--per-stratum`, `--also-exclude`,
`--no-default-exclusions`, filter version or dataset revision is a different corpus: `check` then says no
published hash applies, or compares with `--expected`. A different parser stack that recovers other markup
shows up as a hash mismatch; `provenance.json` records the versions the build used.

## Corpus format

```text
<corpus>/manifest.json     scorer manifest (corpus_id ninfer-rp-forum-fresh-v1; streams rpf-00..; domain roleplay_forum; modes quick/full = all streams)
<corpus>/data/rpf-NN.txt   one excerpt per stream: UTF-8 without BOM, LF newlines, posts separated by one blank line, no trailing newline
<corpus>/provenance.json   dataset revision and file hashes, seed, versions, filter rules, per-stratum removal counts, per-stream thread URL/title/hash
```

The text is plain continuous prose: no speaker names, no turn markers, no usernames. The corpus SHA-256 is the
SHA-256 of the stream files in manifest order, each followed by one NUL byte (`corpus_digest`); the per-stream
SHA-256 values are in `provenance.json`. In a stream record `chars` and `sha256` describe the written file,
`authors` counts the authors of the whole run of posts the excerpt was cut from and `posts` the posts the cut text covers.

## What is selected

Six strata, one raw file each: Elliquiy non-adult RP, Elliquiy one-shot stories, Blue Moon group RP, Blue Moon
fan-based 1x1, RPNation quests, RPNation one-on-one (part 1). **6 threads per stratum, one excerpt per thread,
36 excerpts of 2,420-2,982 characters (98,820 in all).** With the Gemma 4 tokenizer an excerpt is 515-722
tokens (`tokenizers`, no special tokens), which gives two windows per excerpt at context 512 / stride 256: 72
windows, 22,877 scored positions (`infernix-perplexity` reports the same counts, with token-window hash `8a7974c8751af8cd`).

1. Threads of each file are ordered by `sha256(seed + ":" + thread_url)`, seed `20261125`.
2. In that order each thread is dropped or kept by the frozen rules below; the first six survivors of a file
   each contribute one excerpt (no score and no external text is read, so the corpus is a pure function of the
   raw files, the seed and the exclusion list).
3. Threads listed in `excluded_threads.json` are skipped: the 42 threads of the earlier 512-token roleplay set
   (`development_short`) and the 32 threads of the earlier 65k roleplay windows (`long_context`). Because the
   two groups are listed explicitly, the corpus does not depend on having those texts. `check` additionally
   verifies disjointness by shingles against whatever texts you pass.
4. An excerpt that shares a 60-character shingle with an earlier excerpt is dropped.

### Cleaning rules (`cleaning.py`, version `rp-forum-filters/1`)

- **Parse.** Pages are split at `<!-- [NEW PAGE STARTS BELOW] -->` and each post is extracted with a
  forum-specific DOM parser (Elliquiy print pages: `dt.postheader` + `dd.postbody`; XenForo:
  `article.message--post`, `div.bbWrapper`). Text is rendered from the DOM without inserting or removing
  whitespace at inline tags; `<br>` is a newline, block tags are line or paragraph breaks. Quotes, spoilers,
  signatures, link previews, scripts, images and emoticons are dropped. NFC, NBSP to space, zero-width characters
  removed, blank runs collapsed, three or more newlines to two.
- **Post filter** (a failing post is ineligible): 150-6,000 characters; no OOC marker or double parenthesis; no
  URL; no dice notation (`NdM`); at most 3 % non-ASCII characters (typographic quotes, dashes and ellipsis
  excepted); English by stopword rate (at least 25 words, at least 30 % of them from a fixed 50-word list); not
  a character sheet (six or more short `label: value` lines).
- **Thread filter**: title does not match the OOC/administrative pattern (`ooc`, interest check, sign-up,
  character sheet, rules, lore, discussion, recruit*, application, idea, request, search, looking for, wanted,
  resource, help, question, advertis*); at least 8 posts and 8 eligible posts; and the **minors-risk keyword
  screen** over the whole thread: (a) `loli*`, `shota*`, `underage`, `pre-teen*`, `child porn*` or an age 0-17
  written as "N years/yrs old"; (b) any school/teen/student/minor term (`school*`, `classroom`, `homeroom`,
  `teacher(s)`, `student(s)`, `teen*`, `minor(s)`, `brat`, `little girl/boy`) together with any sexual term.
- **Excerpt**: the first run of consecutive eligible posts after the first two eligible posts with at least two
  posts, two distinct authors and 2,400 characters, cut to at most 3,000 characters at the last blank line (at
  least 2,000 kept) or else at the last sentence end.

### Counts of the published build

Every number is reported by `build` and stored in `provenance.json`. A thread is counted once, at the first rule
that removes it; "posts" are the posts of the threads that reached the excerpt stage.

| Stratum | Threads in file | Examined | Excluded | Title | Too short | Minors screen | No excerpt | Duplicate | Kept | Posts: eligible / length / OOC / URL / dice / non-English / sheet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Elliquiy non-adult | 969 | 25 | 0 | 0 | 11 | 8 | 0 | 0 | 6 | 221 / 30 / 10 / 2 / 0 / 0 / 0 |
| Elliquiy one-shots | 3,647 | 24 | 0 | 0 | 6 | 12 | 0 | 0 | 6 | 116 / 0 / 2 / 7 / 0 / 0 / 0 |
| Blue Moon group | 728 | 29 | 0 | 0 | 13 | 10 | 0 | 0 | 6 | 167 / 10 / 1 / 0 / 0 / 2 / 18 |
| Blue Moon fan-based 1x1 | 3,871 | 20 | 0 | 0 | 10 | 4 | 0 | 0 | 6 | 257 / 10 / 38 / 0 / 0 / 0 / 0 |
| RPNation quests | 388 | 58 | 2 | 1 | 39 | 8 | 2 | 0 | 6 | 149 / 289 / 1 / 0 / 1 / 9 / 0 |
| RPNation one-on-one | 3,710 | 23 | 1 | 1 | 12 | 2 | 1 | 0 | 6 | 531 / 354 / 2 / 0 / 0 / 25 / 0 |
| **All** | 13,313 | 179 | 3 | 2 | 91 | 44 | 3 | 0 | 36 | |

No thread failed to parse. The minors screen removed 44 of 179 examined threads; it is a conservative keyword
screen, not a content guarantee (see Limits).

## Score a model on it

The text is scored with the pinned continuity protocol of the Gemma 4 quality studies (see
[Native full-vocabulary KL](maintainer/gemma4-benchmarks.md#native-full-vocabulary-kl)): context 512, stride 256,
chunk 128, the Gemma chat prefix as an explicit unscored prefix, FP8 global KV, BF16 sliding ring, BF16 QK and
BF16 logits.

1. **Reference.** Write a full-vocabulary `.nklref` of the model's own Q8_0 (a libllama reference through
   llama-local-tooling's `scripts/llamacpp/token_logprobs.py reference --protocol-from <stub>`, or
   `infernix-perplexity ... --full-vocab-reference-write <cache>` for an infernix reference artifact) on this
   corpus's windows. `<stub>` is any `.nklref` that `infernix-perplexity --full-vocab-reference-write` wrote on this
   corpus with the protocol flags below (only its protocol header is read). The cache holds one F32 row per scored
   position, about 1.05 MB at vocabulary 262,144: 23,988,277,248 bytes for the 22,877 positions. Use the same prefix
   ids for every cache and every arm.
2. **Score** each arm against the reference of its own model (diode against diode, plain against plain):

   ```bash
   infernix-perplexity <arm.ninfer> --corpus <corpus>/manifest.json \
     --context 512 --stride 256 --prefill-chunk 128 --kv-dtype fp8 --gemma4-qk bf16 --logits-dtype bf16 \
     --full-vocab-prefix-ids 2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101 \
     --full-vocab-reference <ref.nklref> --full-vocab-resamples 4000 --output <arm-run>
   ```

3. **Compare** two arms on the same windows with the stream-clustered paired bootstrap; each stream is one forum
   thread, so the cluster is the thread:

   ```bash
   python -m tools.bench.kv_eval.kl_compare --baseline <a-run>/report.json --candidate <b-run>/report.json \
     --cluster stream --out <paired.json>
   ```

## Scope and limits

- **Evaluation, not calibration.** Use it to measure arms that were selected and built without it. Selecting or
  tuning an arm on these windows makes them development text; draw a new seed for the next confirmation.
- **Mature content.** The corpus is human roleplay forum prose from adult and general forums. The keyword screen
  removes threads with minor-related terms; it is a conservative screen, not a guarantee, and no human reviewed
  the excerpts.
- **Contamination.** Roleplay finetunes may have trained on these forums; the unpublished mixes cannot be checked.
  The paired KL of a quantization against its own reference stays valid for ranking arms, but absolute KL may
  read low.
- **Coverage.** 36 threads from three forums and six sections, block-prefill KL at context 512 only. The
  intervals cover thread sampling, not forum, era or model sampling, and not incremental decode or long context.
- **Rendering.** The scored text is plain prose inside the model turn, not a multi-turn chat with control tokens.
- The shingle check finds common substrings of at least 79 characters; it does not detect paraphrase.

## Tests

`python -m pytest tests/eval` covers the cleaning rules, the minors screen, the deterministic selection (hash
order, seed, exclusions, one excerpt per thread, removal counts), the corpus hash, the shingle check and the
download verification on synthetic forum-shaped fixtures; no forum text is used.
