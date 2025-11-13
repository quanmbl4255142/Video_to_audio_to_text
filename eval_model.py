import os
import json
import csv
import time
import argparse
from typing import List, Tuple
import numpy as np
from transformers import pipeline


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + 1,                        # deletion
                dp[i, j - 1] + 1,                        # insertion
                dp[i - 1, j - 1] + (ref_words[i - 1] != hyp_words[j - 1])  # substitution
            )
    return dp[-1, -1] / max(1, n)


def load_items(jsonl_path: str) -> List[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned PhoWhisper on test set")
    parser.add_argument("--model-dir", type=str, default="./phowhisper-finetuned", help="Path to fine-tuned model dir")
    parser.add_argument("--test-jsonl", type=str, default="data/test.jsonl", help="Path to test JSONL")
    parser.add_argument("--audio-dir", type=str, default="archive/vivos/train/waves", help="Base audio directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--out-csv", type=str, default="eval_results.csv", help="CSV output file for detailed results")
    parser.add_argument("--device", type=int, default=None, help="GPU device index; None=auto; -1=CPU")
    parser.add_argument("--chunk-length-s", type=int, default=30, help="Chunk length seconds for long audio")
    parser.add_argument("--stride-length-s", type=int, default=5, help="Stride seconds between chunks")
    parser.add_argument("--no-spellcheck", action="store_true", help="Disable Vietnamese spell-check post-processing")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not os.path.exists(args.test_jsonl):
        raise FileNotFoundError(f"Test JSONL not found: {args.test_jsonl}")

    # Decide device
    if args.device is None:
        # Use GPU if visible, else CPU
        device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else -1
    else:
        device = args.device

    # Load pipeline (only fine-tuned model)
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model_dir,
        device=device,
        chunk_length_s=args.chunk_length_s,
        stride_length_s=args.stride_length_s,
    )

    # Lazy init LanguageTool for Vietnamese spell-check (optional)
    spell_tool = None
    def correct_text_vi(text: str) -> str:
        nonlocal spell_tool
        if args.no-spellcheck:
            return text
        try:
            if spell_tool is None:
                import language_tool_python  # type: ignore
                # Use local server if available; otherwise, it will start a Java process
                spell_tool = language_tool_python.LanguageTool('vi')
            matches = spell_tool.check(text)
            # language_tool_python.utils.correct may be unavailable in some versions; implement minimal correct
            try:
                from language_tool_python.utils import correct as lt_correct  # type: ignore
                return lt_correct(text, matches)
            except Exception:
                # Manual simple apply replacements sequentially
                corrected = text
                # Apply from end to start to keep indices valid
                for m in sorted(matches, key=lambda m: (m.offset + m.errorLength), reverse=True):
                    rep = (m.replacements[0] if m.replacements else None)
                    if rep is None:
                        continue
                    start = m.offset
                    end = m.offset + m.errorLength
                    corrected = corrected[:start] + rep + corrected[end:]
                return corrected
        except Exception:
            return text

    items = load_items(args.test_jsonl)
    if args.max_samples and args.max_samples > 0:
        items = items[:args.max_samples]

    refs: List[str] = []
    hyps: List[str] = []
    rows: List[Tuple[str, str, str, float]] = []  # (audio, ref, hyp, wer)

    start = time.time()
    # Prefer loading audio ourselves to avoid ffmpeg dependency in pipeline
    try:
        import librosa  # type: ignore
    except Exception as _e:
        librosa = None

    for idx, d in enumerate(items, 1):
        audio_rel = d["audio"]
        # Resolve audio path (absolute or join with base dir)
        audio_path = audio_rel if os.path.isabs(audio_rel) else os.path.join(args.audio_dir, audio_rel)
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio file: {audio_path}")
            continue

        ref = d["sentence"].strip()

        # If librosa available, pass numpy array to avoid ffmpeg
        if librosa is not None:
            try:
                samples, sr = librosa.load(audio_path, sr=16000, mono=True)
                out = asr(samples, return_timestamps=False)
            except Exception:
                # Fallback to filename (requires ffmpeg installed)
                out = asr(audio_path, return_timestamps=False)
        else:
            # Without librosa, pass filename (requires ffmpeg)
            out = asr(audio_path, return_timestamps=False)
        hyp = (out.get("text") or "").strip()
        # Vietnamese spell-check post-processing (optional)
        hyp = correct_text_vi(hyp)

        w = compute_wer(ref, hyp)
        refs.append(ref)
        hyps.append(hyp)
        rows.append((audio_rel, ref, hyp, w))

        if idx % 50 == 0:
            elapsed = time.time() - start
            print(f"- Processed {idx}/{len(items)} samples (elapsed {elapsed:.1f}s)")

    if not rows:
        print("No samples evaluated (missing files?).")
        return

    macro_wer = sum(r[3] for r in rows) / len(rows)
    print("\n===== Evaluation Summary =====")
    print(f"Samples evaluated: {len(rows)}")
    print(f"WER: {macro_wer:.4f}")
    print(f"Time: {time.time() - start:.1f}s")

    # Show a few examples
    print("\nExamples:")
    for audio_name, ref, hyp, w in rows[:5]:
        print(f"- {os.path.basename(audio_name)}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print(f"  WER: {w:.4f}\n")

    # Save CSV
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "reference", "hypothesis", "wer"])
        writer.writerows(rows)
    print(f"Saved details to: {args.out_csv}")


if __name__ == "__main__":
    main()

