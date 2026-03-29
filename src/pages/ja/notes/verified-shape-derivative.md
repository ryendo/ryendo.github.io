---
layout: ../../../layouts/NoteLayout.astro
title: 精度保証付き形状微分メモ
description: Laplacian 固有値の形状微分を題材にした Markdown + KaTeX の簡単な表示例です。
date: 2026-03-29
lang: ja
---

以下は数式表示のサンプルです。KaTeX により、インライン数式 `$ \lambda_1(\Omega) $` とディスプレイ数式の両方を扱えます。

$$
-\Delta u = \lambda u \quad \text{in } \Omega
$$

簡単な形状微分のイメージとして、領域摂動 $\Omega_t$ に対して固有値の差分商

$$
\frac{\lambda(\Omega_t) - \lambda(\Omega)}{t}
$$

を精度保証付きに評価していく、という見方ができます。

## 使い方

新しいノートを追加するには、`src/pages/ja/notes/` に Markdown ファイルを追加し、先頭に frontmatter を書いてください。

```md
---
layout: ../../../layouts/NoteLayout.astro
title: タイトル
description: 説明
date: 2026-03-29
lang: ja
---
```

## KaTeX での注意

KaTeX では一部の LaTeX 環境の扱いが異なるため、たとえば `align` ではなく `aligned` を使う方が安全です。
