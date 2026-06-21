<!--
CONTEXT-FILE TEMPLATES for Zenkai.
Two templates below: the ROOT router and the SUB-PACKAGE map node. Copy the relevant block when adding a
new package, fill the {placeholders}, and delete the guidance comments. The guiding rule for every
CLAUDE.md in this repo: it is a MAP, not a MANUAL — scope + children + pointers, no code samples, no
`file.py#L42` line references (they go stale). Detailed/volatile content goes in docs/ instead.
Keep each file short (aim < ~150 lines). See docs/ai-readiness-setup.md for why.
-->

<!-- ========================= ROOT TEMPLATE (repo-level CLAUDE.md) ========================= -->

# {Project}

{One- or two-sentence description of what the project is and its distinguishing idea.}

This file is a **router**: where things are and where to read more. Keep it lean.

## Repository map

| Path | What it is |
|------|------------|
| {dir/} | {one-line scope; link to its CLAUDE.md if it has one} |
| {docs/specs/} | **Where specs live** — PRDs, plans, and their forms. |

## Commands

{The concrete install / test / one-test / lint / format / docs commands. Use the real tools — point to the
tooling registry for the full list.}

## Conventions

{2–6 essential rules inline; link to docs/conventions.md for the full set. Be specific and checkable.}

## Specs

{Where specs are stored and the per-feature layout, so prd-writing/plan-writing know where to put them.}

## Maintenance

{State when this file must be updated, so it stays self-describing as the repo grows.}


<!-- ===================== SUB-PACKAGE TEMPLATE (per-package CLAUDE.md) ===================== -->

# {package} — {one-line scope}

Scope: {what this package is responsible for, one or two lines}.

## Map (children)

<!-- Enumerate the key modules/sub-packages with a one-line role each. This is the local map. -->
| Child | Role |
|-------|------|
| `{_module.py}` | {role} |
| `{subpkg/}` | {role; link to its own CLAUDE.md} |

## Local conventions

<!-- ONLY what differs from the root conventions. Root conventions are inherited automatically (Claude Code
     concatenates ancestor CLAUDE.md files), so never restate them. Often this section is "nothing extra". -->
{package-specific conventions, or a note that there are none beyond root}

## See also

- Detailed guide: {../docs/guides/<package>.md, if one exists}
- Root router: {../CLAUDE.md}
- Conventions: {../docs/conventions.md}
