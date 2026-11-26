CMNN-Liq — Supplementary Material

English supplementary methods, results and analyses for the CMNN-Liq study of reference-state representations for pathwise-monotonic CPT-based liquefaction probability modelling.

The study distinguishes monotonicity with respect to an individual input coordinate from monotonicity along an engineering scenario path, where PGA and the derived factor of safety change together. It examines reference-state representations, additional directional constraints, conditional PGA queries and an extension to magnitude paths.

## Files

| File | Description |
| --- | --- |
| [CMNN-Liq_Supplementary_Material_EN.pdf](CMNN-Liq_Supplementary_Material_EN.pdf) | PDF version for reading and review. |
| [CMNN-Liq_Supplementary_Material_EN.docx](CMNN-Liq_Supplementary_Material_EN.docx) | Editable Word version with native equations. |

Both files contain the same supplementary material. Upload this README and the two files into the same repository folder to preserve the links.

## Contents

### Supplementary methods

| Section | Content |
| --- | --- |
| SM1 | Activation functions, standardisation and path derivatives |
| SM2 | Baseline definitions and training settings |
| SM3 | Complete comparison design and statistical inference |
| SM4 | Input support, path audits and query criteria |

### Supplementary results and analyses

| Section | Content |
| --- | --- |
| S1 | Network implementation and effective-coefficient mapping |
| S2 | Complete effect sizes and statistical definitions |
| S3 | Regional heterogeneity and case-level loss decomposition |
| S4 | Support rules, root eligibility and ranges for four cases |
| S5 | Data selection and engineering factor-of-safety implementation |
| S6 | Tuned baselines and separate comparison families |
| S7 | Extrapolation, preprocessing and other-direction diagnostics |
| S8 | Complete path audits and threshold queries for stepwise models |
| S9 | Eight-cell implementation design and LightGBM representation comparison |
| S10 | Seed versions of fitted-model diagnostics and continuous-model queries |
| S11 | Matched additional-constraint effects and robustness results |
| S12 | Additional empirical baselines and supplementary evaluation |
| S13 | Development history, sensitivity analyses, multivariable path audits and probability calibration |
| S14 | Reference-magnitude extension, ablations and regional paired comparisons |

Supplementary Figs. S-A and S-B summarise regional paired differences and target-probability queries, respectively. Reference numbers follow the main manuscript; the cited references are listed at the end of the supplement.

## Interpretation of the results

- Seed ranges describe variability across repeated training runs; they are not confidence intervals.
- Statistical comparisons use the protocols and comparison families specified in SM3. Results from different seed versions or correction schemes should be interpreted under their stated protocols.
- A structural monotonicity guarantee applies to the scenario paths for which it is established. A grid audit describes responses at the evaluated nodes.
- Conditional PGA queries are model-based results within the stated input support. They are not measured triggering thresholds or design ground motions.

## Data and code availability

The source case-history database is **LIQ/7/2833, version 3**:

[https://doi.org/10.17632/rmjnghdrbv.3](https://doi.org/10.17632/rmjnghdrbv.3)

Core method implementations are available in the [CMNN-Liq repository](https://github.com/wuwangzsj/CMNN-Liq).

The repository provides the main implementations and methodological workflow. It is not a complete reproduction package and does not include all held-out predictions or trained model weights. Other data supporting the reported results are available from the authors on request.

## Version

English supplementary material prepared on **26 November 2026**. The translation retains the numerical results, equations, figures, tables and reference numbering of the corresponding Chinese supplement. Preparing this English version involved no model retraining or new experiments.
