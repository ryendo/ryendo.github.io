# Ryoki Endo academic site starter
Astro + Markdown + KaTeX + GitHub Pages

このプロジェクトは、researchmap の公開情報をもとにした bilingual academic website のひな型です。  
英語 / 日本語のページを分け、**講演は日本語と英語で分離**してあります。  
講演タイトルは与えられた表記をそのまま使っています。

## 含まれているもの
- `/en/` 英語トップページ
- `/ja/` 日本語トップページ
- `src/data/profile.ts` に研究者プロフィールの構造化データ
- `src/pages/en/notes/` と `src/pages/ja/notes/` に Markdown ノート
- KaTeX 数式対応
- GitHub Pages 用 workflow (`.github/workflows/deploy.yml`)

## まず編集する場所
### 1. Astro の公開 URL 設定
`astro.config.mjs` を開いて、次の 2 行を自分の GitHub リポジトリに合わせて直してください。

```js
site: 'https://YOUR_GITHUB_USERNAME.github.io',
base: '/YOUR_REPOSITORY_NAME',
```

- **ユーザーサイト** (`username.github.io`) で公開するなら:
  - `site: 'https://username.github.io'`
  - `base: ''`
- **プロジェクトサイト** (`username.github.io/repo-name`) で公開するなら:
  - `site: 'https://username.github.io'`
  - `base: '/repo-name'`

### 2. プロフィール本文と業績
`src/data/profile.ts` を編集してください。  
研究キーワード、論文、講演、受賞、所属学会、研究課題などはここにまとまっています。

### 3. 顔写真
仮画像は `public/avatar.svg` です。  
自分の写真に差し替えるなら、同じ名前で置き換えるだけで反映できます。

## ノートの追加方法
### 日本語
`src/pages/ja/notes/` に Markdown ファイルを追加します。

### English
Add a Markdown file under `src/pages/en/notes/`.

使う frontmatter の例:

```md
---
layout: ../../../layouts/NoteLayout.astro
title: タイトル
description: 短い説明
date: 2026-03-29
lang: ja
---
```

英語ノートなら `lang: en` にしてください。

## ローカルで確認する
Node.js を入れた状態で、ターミナルでこのプロジェクトに移動して:

```bash
npm install
npm run dev
```

ビルド確認:

```bash
npm run build
```

## GitHub Pages へのデプロイ手順
### いちばん簡単な流れ
1. GitHub で新しい repository を作る  
2. このプロジェクト一式をその repository に入れる  
3. `astro.config.mjs` の `site` と `base` を直す  
4. 変更を `main` ブランチに push する  
5. GitHub の **Settings → Pages** で **Source = GitHub Actions** を選ぶ  
6. Actions が走って、サイトが公開される

### コマンド例
```bash
git init
git branch -M main
git add .
git commit -m "Initial academic site"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

## データ構造の考え方
- **固定情報**: `src/data/profile.ts`
- **見た目**: `src/layouts/` と `src/styles/`
- **日本語ページ**: `src/pages/ja/`
- **英語ページ**: `src/pages/en/`
- **記事 / ノート**: Markdown ファイル

WordPress のような CMS は使わず、**生成 AI が作ったコードをそのまま手で直しやすい構造**に寄せています。

## 補足
- メールアドレスは researchmap 表記をそのまま保持しています。必要なら自分で修正してください。
- 講演タイトルは指定どおり原文のままです。
- 英語ページでも「Talks in Japanese」と「Talks in English」を分けてあります。
