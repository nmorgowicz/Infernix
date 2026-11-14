"""Offline checkpoint-to-artifact conversion.

sources reads and interprets checkpoint values. qwen3_5 maps those values and
configuration to model's logical parameters; resources supplies frontend bytes.
recipe selects sources, representations, grouping, and methods. official_recipes
supplies the maintained assignments; proposal adds the indexed proposal head.
methods adapts numerical quantization or encoded import to artifact.tensor_output.
pipeline prepares jobs, runs them through the artifact writer, and reports results.
__main__ supplies the CLI and loads explicitly selected recipe functions.
gemma4 maps Gemma 4 text, MTP assistant and vision; gemma4_q4 composes the Gemma 4 31B Q4/FP8 recipe and
gemma4_q4_cli is its `python -m tools.convert.gemma4_q4` command line.
"""
