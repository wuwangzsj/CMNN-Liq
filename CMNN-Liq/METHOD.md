# Method and implementation map

## Reference-state construction

Let a=PGA/g, z=(Mw,D,r,qt1N,Ic). In the adopted stress-based implementation, FS(a,z)=kappa(z)/a. With a_ref=0.1, define u=ln(a/a_ref), s=ln(FS(a_ref,z)) and t=ln(FS(a,z)); then t=s-u. `features.inputs` orders the columns as [Mw,D,r,qt1N,Ic,s,u].

For a current-state function G(z,u,t), the PGA-path derivative at fixed z is G_u-G_t. The constraint G_u>=0 alone does not establish this derivative's sign. In reference coordinates, fixed z also fixes s, so the same path changes only u. Nonnegative u weights, nondecreasing mixed activations and nonnegative output weights provide the sufficient structural condition.

This guarantee concerns the specified PGA path. It does not establish all engineering response directions or probability calibration. The independent s-coordinate response is unconstrained in M.

## Additional constraints

For each hidden unit, reference coefficients (d,K) and current-state coefficients (beta,alpha) satisfy d=beta and K=alpha-beta. All four matched configurations impose K>=0. R adds d<=0; C adds K+d>=0; A0 adds both. These are unit-level sufficient constraints. `factorial.py` converts standardized coefficients before interpreting the constraints. Its internal group labels G1 and G3 denote M0 and A0, respectively. Other G labels are archived implementation variants, not additional main models.

The current-state A network constrains alpha>=0 and beta<=0; N only constrains alpha>=0. B1 constrains u in a reference-state Logistic model. B0 uses current ln(FS), with a nonpositive coefficient. All of these implementation branches are retained in `comparators.py`.

## Training and evaluation workflow

The main study used 23-region outer leave-one-region-out evaluation, training-region-only preprocessing, and five grouped inner folds. Six CMNN candidates combine widths 16/32 and L2 coefficients 0.0001/0.001/0.01. Three inner seeds select the candidate using mean region-equal Brier; the selected candidate's median inner best epoch determines the main outer refit budget. Ten outer seeds are listed separately in the configuration.

The main kernel uses float64 full-batch Adam, learning rate 0.003, optimizer weight decay zero, and binary cross-entropy plus explicit L2. Validation is checked every 10 epochs, with a 1e-5 improvement threshold, patience 100 and cap 1000. The main L2 includes network biases. See the actual kernels for the precise objective.

The fixed M0/R/C/A0 comparison uses H=32 and lambda=0.001 with shared initialization and archived matched epoch budgets. It must not be described as the same configuration as inner-selected M. The standalone kernels do not recreate the complete scheduling, split archives, statistical families, or historical matching protocols. The manuscript and supplementary methods remain the specification for those experiments.

## Support, query and path diagnostics

Build `Support` on outer training inputs: one instance in (z,s), another in (z,s,u), with k=5, quantile=0.95, minimum-neighbour count=5. Both checks are required. Intervals are in u and additionally use the original global bounds a in [0.01,1.5]. `Curve` receives reference-coordinate effective weights and the fixed first six coordinates. Its roots are calculated per network; an average of per-seed roots need not be the root of the mean probability curve. Plateau and root-state bookkeeping must retain the original numerical tolerances.

For audits, recompute all derived features after changing the physical input. Use only consecutive supported nodes, resetting at every support gap. `path_metrics.case_metrics` reports maximum adverse step, peak drawdown and negative-step sum. Use sign=+1 for increasing PGA/Mw and sign=-1 for increasing cone resistance. A grid audit is a numerical diagnostic, not a proof between grid nodes. Seed ranges describe training variability, not confidence intervals.

## Optional reference-Mw extension

When Mw enters FS only via MSF=(Mw/7.5)^(-2.56), define sstar=ln FS(a_ref, Mw_ref=7.5, other z), and m=ln[MSF(7.5)/MSF(Mw)]. Then ln FS=sstar-u-m. Constraining both u and m gives the two specified path guarantees. `reference_mw.py` retains the archived input map, sign mask and NumPy inference. The extension is a supplementary experiment, not a replacement for the original main model.
