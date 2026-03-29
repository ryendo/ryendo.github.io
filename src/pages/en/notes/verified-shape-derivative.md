---
layout: ../../../layouts/NoteLayout.astro
title: A note on verified shape derivatives
description: A short Markdown + KaTeX example based on shape derivatives of Laplacian eigenvalues.
date: 2026-03-29
lang: en
---

This note demonstrates how mathematical writing works in Markdown with KaTeX. Both inline math such as `$ \lambda_1(\Omega) $` and display equations are supported.

$$
-\Delta u = \lambda u \quad \text{in } \Omega
$$

A simple way to think about verified shape derivatives is to study difference quotients along a perturbed family of domains $\Omega_t$:

$$
\frac{\lambda(\Omega_t) - \lambda(\Omega)}{t}.
$$

In practice, one tries to obtain rigorous bounds for such expressions with verified numerical computation.

## How to add a new note

Create a new Markdown file under `src/pages/en/notes/` and add frontmatter like this:

```md
---
layout: ../../../layouts/NoteLayout.astro
title: Your title
description: A short summary
date: 2026-03-29
lang: en
---
```

## A KaTeX note

KaTeX supports a wide range of TeX commands, but not every LaTeX environment works exactly the same way. For aligned displays, `aligned` is usually safer than `align`.
