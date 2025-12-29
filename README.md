# Portfolio

Single-page React + Vite portfolio with client-side routing and a neon simulation-inspired aesthetic.

## Setup

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Deploy

- Static output is in `dist/` after build.
- Netlify: `_redirects` is included in `public/` for SPA fallback.
- Other hosts: configure an SPA fallback to `/index.html` for unknown routes.

### Subpath deployments

If deploying under a subpath (e.g. `https://example.com/portfolio/`), set:

```bash
BASE_PATH=/portfolio/ npm run build
```

This controls the Vite `base` setting in `vite.config.js`.

## Assets

- `public/profile-placeholder.svg`
- `public/resume.pdf`
- `public/paper-placeholder.pdf`
