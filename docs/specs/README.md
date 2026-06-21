# Specs

Where Zenkai's specs live: **PRDs, delivery plans, and their filled forms**, one folder per feature/epic.

## Layout

```
docs/specs/
  <feature-name>/
    prd.md                       # the PRD (from the prd-writing skill)
    plan.md                      # the delivery plan (from the plan-writing skill)
    context/
      prd-form.md                # filled PRD form (decision record)
      plan-form.md               # filled plan form (decision record)
      plan-critique.md           # appended plan critiques
    implementation-review.md     # filled per-chunk during execution
```

`<feature-name>` is kebab-case (e.g. `target-prop-scheduler`). Not every feature needs every file — small
changes may have only `plan.md`. The `prd-writing` and `plan-writing` skills read [the root
CLAUDE.md](../../CLAUDE.md) to find this location and the repo conventions, then write here.
