[forall link] e[X->Y] >= 0                                   # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=0.6234 delta=+0.0000 score=2.470
[forall node] H[X,X] <= o(X)                                 # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=1 delta=+0.0000 score=2.470
[forall link] H[X,Y] != H[Y,X]                               # ANTI             eps=0.9830 kappa=0.991 supp=1.00 lift=1.593 delta=+0.0000 score=2.460
[forall link] i[X<-Y] != i[Y<-X]                             # ANTI             eps=0.9437 kappa=0.991 supp=1.00 lift=1.468 delta=+0.0000 score=2.460
[forall link] e[X->Y] != e[Y->X]                             # ANTI             eps=0.9445 kappa=0.991 supp=1.00 lift=1.475 delta=+0.0000 score=2.460
[forall network] MIN(t(*)) <= SUM(H[*,*])                    # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=0.9304 delta=+0.0000 score=2.455
[forall network] SUM(H[*,*]) + -1*AVG(o(*)) >= 0             # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=0.9586 delta=+0.0000 score=2.433
[forall node] AVG(H[X,*]) + t(X) >= 0 + 0.5*SUM(H[*,X])      # EXACT            eps=0.0000 kappa=1.000 supp=1.00 lift=1.809 delta=+0.0000 score=2.417
[forall link] i[X<-Y] ~= e[Y->X]                             # EXACT            eps=0.0273 kappa=0.900 supp=1.00 lift=191.5 delta=+0.0000 score=2.330
[forall link] e[X->Y] ~= i[Y<-X]                             # EXACT            eps=0.0272 kappa=0.899 supp=1.00 lift=192.2 delta=+0.0000 score=2.329
[forall network] SUM(o(*)) ~= SUM(t(*))                      # EXACT            eps=0.0147 kappa=0.900 supp=1.00 lift=172 delta=+0.0002 score=2.325
[forall network] SUM(H[*,*]) ~= SUM(t(*))                    # SOFT_STRUCTURAL  eps=0.0467 kappa=0.904 supp=1.00 lift=41.86 delta=+0.0220 score=2.303
[forall node] SUM(e[X->*]) ~= SUM(i[X<-*]) + o(X) + -1*t(X)  # EXACT            eps=0.0168 kappa=0.899 supp=1.00 lift=167 delta=-0.0003 score=2.293
[forall network] SUM(H[*,*]) ~= SUM(o(*))                    # SOFT_STRUCTURAL  eps=0.0443 kappa=0.882 supp=1.00 lift=43.56 delta=+0.0215 score=2.283
[forall node] o(X) ~= SUM(H[X,*])                            # SOFT_STRUCTURAL  eps=0.1089 kappa=0.911 supp=1.00 lift=28.98 delta=-0.0188 score=2.267
[forall node] t(X) ~= SUM(H[*,X])                            # SOFT_STRUCTURAL  eps=0.1067 kappa=0.899 supp=1.00 lift=29.22 delta=-0.0169 score=2.257

# Autogram learned invariants
# dataset    : abilene
# run time   : 2026-06-23T17:44:03Z (UTC)
# git        : c49ecfc6 (dirty)
# proposer   : subagent  seed=0  deployed=False  rel_noise=0.02
# recall     : 87.50% form | 87.50% strict | 7/8 full
# portfolio  : 16 invariant(s); each rule line is `<invariant>  # <verdict> <metrics>`
