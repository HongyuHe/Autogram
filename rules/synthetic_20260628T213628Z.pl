[forall node] demand_self == 0                               # EXACT  eps=0.0000 cov=1.000[1.00,1.00] op_cov=1.000 supp=1.00 lift=1e+12 p=0.0000 stab=0.000 mdl=+9.366
[forall node] m_src ~= SUM(demand_row)                       # SOFT   eps=0.0541 cov=0.994[0.99,1.00] op_cov=0.993 supp=1.00 lift=60.92 p=0.0041 stab=0.002 mdl=+4.484
[forall link] o0_rev == o1                                   # SOFT   eps=0.0887 cov=0.998[1.00,1.00] op_cov=1.000 supp=1.00 lift=33.36 p=0.0138 stab=0.000 mdl=+4.009

# Autogram discovered invariants (data-only; no catalogue, no oracle)
# dataset    : synthetic
# run time   : 2026-06-28T21:36:28Z (UTC)
# git        : d3ccde7
# proposer   : random  seed=0  rounds=6  reinductions=0
# portfolio  : 3 invariant(s); each line is `<invariant>  # <strictness> <metrics>`
