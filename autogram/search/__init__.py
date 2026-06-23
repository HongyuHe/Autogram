"""Evolutionary rule-set search (design Sec. 10.3-10.4).

The middle loop: a quality-diversity (MAP-Elites) search over the typed DSL-AST
genotype, with island parallelism, Thompson-sampling budget allocation, and a
submodular mine-and-cover assembler that turns the archive of accepted rules into a
parsimonious, de-duplicated portfolio.  The band/threshold is *never* searched here --
it is fit analytically inside the evaluator (the inner loop, Sec. 5.4).
"""
