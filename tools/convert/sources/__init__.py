"""Checkpoint bytes, logical value views, and source-format interpretation.

logical owns source values and row/axis transforms; safetensors owns local file
access; compressed_tensors interprets the current FP8/NVFP4 checkpoint encoding;
packed_int4 reads compressed-tensors INT4 checkpoints as encoded q4 rows.
User sources can construct LogicalSource without using either checkpoint reader.
"""
