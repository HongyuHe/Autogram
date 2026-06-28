[forall node] H[X,X] == 0                                  # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=1e+12 delta=+0.0000 score=2.470
[forall link] e[X->Y] >= 0                                 # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=0.7366 delta=+0.0000 score=2.470
[forall node] t(X) >= MAX(H[*,X])                          # EXACT            eps=0.0000 kappa=0.992 supp=1.00 lift=1.575 delta=+0.0000 score=2.455
[forall link] i[X<-Y] != i[Y<-X]                           # ANTI             eps=0.9424 kappa=0.979 supp=1.00 lift=1.377 delta=+0.0000 score=2.448
[forall link] e[X->Y] != e[Y->X]                           # ANTI             eps=0.9424 kappa=0.979 supp=1.00 lift=1.375 delta=+0.0000 score=2.448
[forall network] MAX(t(*)) + MAX(H[*,*]) >= 0 + MIN(o(*))  # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=0.9663 delta=+0.0000 score=2.417
[forall link] H[X,Y] != H[Y,X]                             # ANTI             eps=0.9813 kappa=0.927 supp=1.00 lift=1.566 delta=+0.0000 score=2.395
[forall network] SUM(o(*)) ~= 22*AVG(o(*))                 # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=9.866e+11 delta=+0.0000 score=2.373
[forall network] SUM(o(*)) ~= SUM(t(*))                    # EXACT            eps=0.0133 kappa=0.932 supp=1.00 lift=206.8 delta=+0.0000 score=2.358
[forall link] e[X->Y] ~= i[Y<-X]                           # EXACT            eps=0.0262 kappa=0.899 supp=1.00 lift=234.2 delta=+0.0000 score=2.330
[forall link] i[X<-Y] ~= e[Y->X]                           # EXACT            eps=0.0261 kappa=0.898 supp=1.00 lift=234.5 delta=+0.0000 score=2.330
[forall network] SUM(o(*)) ~= SUM(H[*,*])                  # SOFT_STRUCTURAL  eps=0.0411 kappa=0.908 supp=1.00 lift=46.59 delta=-0.0213 score=2.312
[forall node] o(X) + SUM(i[X<-*]) ~= t(X) + SUM(e[X->*])   # EXACT            eps=0.0105 kappa=0.898 supp=1.00 lift=301.7 delta=+0.0000 score=2.296
[forall node] t(X) ~= SUM(H[*,X])                          # SOFT_STRUCTURAL  eps=0.1078 kappa=0.906 supp=1.00 lift=30.22 delta=-0.0159 score=2.263
[forall node] o(X) ~= SUM(H[X,*])                          # SOFT_STRUCTURAL  eps=0.1086 kappa=0.905 supp=1.00 lift=29.51 delta=-0.0175 score=2.261
[forall link] e[Y->X] >= 0 + H[Y,X] + H[X,Y]               # SOFT             eps=0.1212 kappa=0.899 supp=1.00 lift=1.218 delta=+0.0000 score=2.223

# Autogram learned invariants
# dataset    : geant
# run time   : 2026-06-23T17:44:13Z (UTC)
# git        : c49ecfc6 (dirty)
# proposer   : subagent  seed=0  deployed=False  rel_noise=0.02
# recall     : 100.00% form | 100.00% strict | 8/8 full
# portfolio  : 16 invariant(s); each rule line is `<invariant>  # <verdict> <metrics>`
