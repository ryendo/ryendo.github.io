import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Edit these two values before your first deploy.
//
// User / organization site:
//   site: 'https://YOUR_GITHUB_USERNAME.github.io'
//   base: ''
//
// Project site:
//   site: 'https://YOUR_GITHUB_USERNAME.github.io'
//   base: '/YOUR_REPOSITORY_NAME'
export default defineConfig({
  site: 'https://ryendo.github.io',
  base: '',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
