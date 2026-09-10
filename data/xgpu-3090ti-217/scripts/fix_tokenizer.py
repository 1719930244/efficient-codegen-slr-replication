"""Fix speculative decoding tokenizer issue in eval_humaneval.py"""
import sys

filepath = sys.argv[1]
with open(filepath, "r") as f:
    code = f.read()

fixes = 0

# Fix 1: generate_completion signature - add assistant_tokenizer
old = "    assistant_model=None,\n    device: str = \"cuda:0\",\n) -> tuple[str, float, float, int]:"
new = "    assistant_model=None,\n    assistant_tokenizer=None,\n    device: str = \"cuda:0\",\n) -> tuple[str, float, float, int]:"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 1: generate_completion signature")

# Fix 2: pass tokenizer/assistant_tokenizer to generate()
old = '    if assistant_model is not None:\n        gen_kwargs["assistant_model"] = assistant_model\n\n    torch.cuda.synchronize()'
new = '    if assistant_model is not None:\n        gen_kwargs["assistant_model"] = assistant_model\n        # [FIX #18] transformers 5.x requires explicit tokenizers for speculative decoding\n        gen_kwargs["tokenizer"] = tokenizer\n        if assistant_tokenizer is not None:\n            gen_kwargs["assistant_tokenizer"] = assistant_tokenizer\n\n    torch.cuda.synchronize()'
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 2: gen_kwargs tokenizers")

# Fix 3: adaptive_sampling signature
old = "    assistant_model=None,\n    device: str = \"cuda:0\",\n) -> tuple[list[str], list[bool], float, float, int]:"
new = "    assistant_model=None,\n    assistant_tokenizer=None,\n    device: str = \"cuda:0\",\n) -> tuple[list[str], list[bool], float, float, int]:"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 3: adaptive_sampling signature")

# Fix 4: adaptive_sampling call to generate_completion
old = "            assistant_model=assistant_model,\n            device=device,\n        )\n        completions.append(comp)"
new = "            assistant_model=assistant_model,\n            assistant_tokenizer=assistant_tokenizer,\n            device=device,\n        )\n        completions.append(comp)"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 4: adaptive_sampling -> generate_completion call")

# Fix 5: run_benchmark - keep draft tokenizer
old = '    assistant_model = None\n    if decoding == "speculative" and draft_model_path:\n        print(f"Loading draft model (FP16): {draft_model_path}")\n        assistant_model, _ = load_model(draft_model_path, "fp16", device)'
new = '    assistant_model = None\n    assistant_tokenizer = None\n    if decoding == "speculative" and draft_model_path:\n        print(f"Loading draft model (FP16): {draft_model_path}")\n        assistant_model, assistant_tokenizer = load_model(draft_model_path, "fp16", device)'
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 5: keep draft tokenizer")

# Fix 6: run_benchmark greedy path
old = "                assistant_model=assistant_model,\n                device=device,\n            )\n            passed = check_correctness"
new = "                assistant_model=assistant_model,\n                assistant_tokenizer=assistant_tokenizer,\n                device=device,\n            )\n            passed = check_correctness"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 6: greedy path assistant_tokenizer")

# Fix 7: run_benchmark adaptive path
old = "                assistant_model=assistant_model,\n                device=device,\n            )\n            best_idx"
new = "                assistant_model=assistant_model,\n                assistant_tokenizer=assistant_tokenizer,\n                device=device,\n            )\n            best_idx"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 7: adaptive path assistant_tokenizer")

# Fix 8: cleanup assistant_tokenizer
old = "    if assistant_model is not None:\n        del assistant_model\n    gc.collect()"
new = "    if assistant_model is not None:\n        del assistant_model\n    if assistant_tokenizer is not None:\n        del assistant_tokenizer\n    gc.collect()"
if old in code:
    code = code.replace(old, new)
    fixes += 1
    print("Fix 8: cleanup assistant_tokenizer")

with open(filepath, "w") as f:
    f.write(code)

print(f"\nTotal fixes applied: {fixes}/8")
if fixes < 8:
    print("WARNING: Some fixes did not match - manual review needed")
    sys.exit(1)
