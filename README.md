# Ryoki Endo academic site

Astro + Markdown + KaTeX + GitHub Pages

自分用の研究者個人サイト。
researchmap の公開情報をもとに、英語 / 日本語のページを分けて作っている。
講演は日本語と英語で分けてあり、タイトルは原文のまま入れている。

## 入っているもの

- `/en/` 英語ページ
- `/ja/` 日本語ページ
- `src/data/profile.ts` にプロフィール・業績データ
- `src/pages/en/notes/`, `src/pages/ja/notes/` に Markdown ノート
- KaTeX 数式対応
- GitHub Pages 用 workflow (`.github/workflows/deploy.yml`)

## 先に直すところ

### 1. 公開 URL

`astro.config.mjs` の以下を自分の GitHub に合わせる。

```js
site: 'https://YOUR_GITHUB_USERNAME.github.io',
base: '/YOUR_REPOSITORY_NAME',
```

- ユーザーサイト `username.github.io` の場合
  - `site: 'https://username.github.io'`
  - `base: ''`
- プロジェクトサイト `username.github.io/repo-name` の場合
  - `site: 'https://username.github.io'`
  - `base: '/repo-name'`

### 2. 内容

`src/data/profile.ts` を編集する。
研究キーワード、論文、講演、受賞、所属学会、研究課題などはここ。

### 3. 顔写真

仮画像は `public/avatar.svg`。
差し替えるなら同名で置き換えればよい。

## ノート追加

### 日本語

`src/pages/ja/notes/` に Markdown を追加。

### English

Add a Markdown file under `src/pages/en/notes/`.

frontmatter 例:

```md
---
layout: ../../../layouts/NoteLayout.astro
title: タイトル
description: 短い説明
date: 2026-03-29
lang: ja
---
```

英語なら `lang: en`。

## ローカル確認

```bash
npm install
npm run dev
```

ビルド確認:

```bash
npm run build
```

## GitHub Pages 公開

流れ:

1. GitHub で repository を作る
2. このプロジェクトを入れる
3. `astro.config.mjs` の `site` と `base` を直す
4. `main` に push
5. GitHub の **Settings → Pages** で **Source = GitHub Actions**
6. Actions が通れば公開

コマンド例:

```bash
git init
git branch -M main
git add .
git commit -m "Initial site"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

## 構造メモ

- 固定データ: `src/data/profile.ts`
- レイアウト: `src/layouts/`
- スタイル: `src/styles/`
- 日本語ページ: `src/pages/ja/`
- 英語ページ: `src/pages/en/`
- ノート: Markdown

## メモ

- WordPress は使わず、手で直しやすい構成にしている
- メールアドレスは必要なら自分で修正
- 講演タイトルは原文のまま
- 英語ページでも Japanese / English talks を分けている
