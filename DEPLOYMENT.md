# Deployment Guide

This document describes how to deploy the WAVES Intelligence™ marketing site to Vercel.

## Overview

The Next.js marketing site is located in the `/site` directory of this repository. It is configured for deployment to Vercel with automatic preview and production deployments.

## Initial Setup

### Connect Repository to Vercel

1. **Create/Login to Vercel Account**
   - Visit [vercel.com](https://vercel.com)
   - Sign in with GitHub

2. **Import Project**
   - Click "Add New..." → "Project"
   - Select this repository: `jasonheldman-creator/Waves-Simple`
   - Vercel will automatically detect the `vercel.json` configuration

3. **Configure Project Settings**
   - **Root Directory**: `site` (auto-configured from vercel.json)
   - **Build Command**: `npm run build` (auto-configured from vercel.json)
   - **Output Directory**: `.next` (Next.js default)
   - **Install Command**: `npm ci` (default)
   - **Framework Preset**: Next.js (auto-detected)

4. **Deploy**
   - Click "Deploy"
   - Wait for the initial deployment to complete

## Deployment Types

### Production Deployments

Production deployments occur automatically when:
- Code is pushed to the `main` branch
- Pull requests are merged to `main`

**Finding Production URLs:**
- In Vercel Dashboard: Project → Deployments → Filter by "Production"
- Production URL format: `https://[project-name].vercel.app`
- Custom domains can be configured in Vercel Dashboard → Settings → Domains

### Preview Deployments

Preview deployments are automatically created for:
- Every push to a pull request
- Every commit to a branch with an open pull request

**Finding Preview URLs:**
1. **In GitHub Pull Request:**
   - Look for the "Vercel bot" comment
   - Check the PR "Checks" section for "Vercel – site"
   - Click "Details" to visit the preview deployment

2. **In Vercel Dashboard:**
   - Visit your project page
   - Click "Deployments" tab
   - Filter by branch name or PR number
   - Preview URLs are listed for each deployment

3. **Preview URL Format:**
   - Branch-based: `https://[project-name]-[branch-name]-[team].vercel.app`
   - PR-based: `https://[project-name]-git-[branch]-[team].vercel.app`

## Environment Variables

Currently, the marketing site does not require any environment variables. If needed in the future:

1. Go to Vercel Dashboard → Project → Settings → Environment Variables
2. Add variables for Production, Preview, or Development environments
3. Redeploy for changes to take effect

## Manual Deployments

To trigger a manual deployment:

1. **Via Vercel Dashboard:**
   - Go to Project → Deployments
   - Click "..." menu on any deployment
   - Select "Redeploy"

2. **Via Vercel CLI:**
   ```bash
   npm i -g vercel
   vercel --prod  # For production
   vercel         # For preview
   ```

## Build Configuration

The build is configured via:
- `vercel.json` - Vercel-specific settings
- `site/package.json` - Build scripts and dependencies
- `site/next.config.ts` - Next.js configuration

### vercel.json
```json
{
  "buildCommand": "npm run build",
  "rootDirectory": "site"
}
```

### Build Scripts
```json
{
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "lint": "eslint"
}
```

Note: The `lint` script uses ESLint's flat config which automatically discovers and lints files based on `eslint.config.mjs`.

## Local Development

To run the site locally:

```bash
cd site
npm install
npm run dev
```

The site will be available at `http://localhost:3000`

## CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/site-ci.yml`) automatically verifies builds on:
- Push to `main` branch
- Pull requests to `main` branch

The CI workflow ensures that the build succeeds before merging, preventing broken deployments.

## Troubleshooting

### Build Failures

1. **Check Build Logs:**
   - In Vercel Dashboard → Deployments → Click failed deployment
   - Scroll to "Build Logs" section

2. **Common Issues:**
   - Missing dependencies: Run `npm ci` locally to verify
   - TypeScript errors: Run `npm run build` locally
   - Linting errors: Run `npm run lint` locally

3. **Test Locally:**
   ```bash
   cd site
   npm ci
   npm run build
   ```

### Preview Deployment Not Showing

- Ensure the PR is from a branch in the same repository (not a fork)
- Check Vercel Dashboard → Project → Settings → Git
- Verify that "Automatic Deployments" is enabled for preview branches

### Production Deployment Failed

- Check that the `main` branch builds successfully locally
- Review Vercel build logs for specific errors
- Verify all environment variables are set (if needed)

## Monitoring

### Deployment Status

- **Vercel Dashboard**: Real-time deployment status and logs
- **GitHub PR Checks**: Build status badges
- **Vercel Bot**: Automated PR comments with deployment URLs

### Analytics

Vercel provides built-in analytics:
- Go to Vercel Dashboard → Project → Analytics
- View page views, visitors, and performance metrics

## Support

For deployment issues:
1. Check Vercel build logs
2. Review this documentation
3. Test builds locally with `npm run build`
4. Check Vercel status: [vercel-status.com](https://www.vercel-status.com)
