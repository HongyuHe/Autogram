"""Typed random generation and mutation helpers for the collapsed discovery loop.

The old MAP-Elites / Thompson stack is gone.  Search now lives in
``autogram.discovery.loop`` as one proposer feeding a simple Pareto archive; this package keeps
only the schema-typed AST variation routines used by the random proposer.
"""
