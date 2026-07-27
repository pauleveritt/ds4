# Superpowers Roadmap Backlog

Future and partial work that is tracked but intentionally kept out of the
active execution view. Promote an item only after its scope, evidence, and
acceptance signal are concrete enough for a design and plan.

## Inference Sampling

| Status | Cycle | Source | Spec | Depends On | Acceptance Signal |
| --- | --- | --- | --- | --- | --- |
| future | S-MIN-P-DEFAULT — Make min-p 0.05 the default | Antirez, ds4 creator (2026-07-27): min-p at 0.05 recovered substantial Laguna S 2.1 quality without a meaningful loss of output richness; he recommends it as the default for most inference engines. | — | Sampling-default audit across CLI, agent, and server surfaces; explicit user-supplied sampling options must retain precedence. | All applicable inference surfaces default to `min-p=0.05`; explicit `--min-p`/API values win; Laguna S 2.1 regression and richness checks show the claimed quality recovery without a material output-length or diversity loss. |
| future | S-MIN-P-VS-TOP-P — Compare adaptive min-p with nucleus sampling | Antirez, ds4 creator (2026-07-27): investigate whether min-p's threshold relative to the top token is better than top-p's distribution-mass cutoff, because top-p can retain very poor tokens. | — | A representative, reproducible evaluation set covering Laguna S 2.1 and at least one other supported model; fixed temperature/seed and documented sampling grids. | Report compares min-p and top-p at matched richness/length targets, records quality and bad-token failures, and makes an evidence-backed default/recommendation decision for each supported inference engine. |

### Creator notes

> Hint: min-p 0.05 (never accept tokens < 5% score of top one) recovers a
> lot of quality in Laguna S2.1 without impacting output richness in any
> significant way. min-p should be the default for most inference engines.
>
> Did you compare min-p vs top-p (nucleus sampling), does it feel that having
> a threshold dependent on top score is a lot better to have it dynamically
> dependent on the distribution? top-p *is* the problem under selecting very
> bad tokens can happen indeed.
