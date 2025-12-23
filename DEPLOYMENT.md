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

3. **Configure Project Settings**
   
   **CRITICAL:** During import or in Project Settings → General, configure:
   
   - **Framework Preset**: `Next.js` (should auto-detect)
   - **Root Directory**: `site` ⚠️ **REQUIRED** - The Next.js app is in the `/site` subdirectory
   - **Build Command**: Leave empty (uses default: `npm run build`)
   - **Output Directory**: Leave empty (uses default: `.next`)
   - **Install Command**: Leave empty (uses default: `npm install`)
   - **Node.js Version**: 20.x (recommended)

   **Note:** The `vercel.json` file handles domain redirects only. Build configuration must be set in the Vercel dashboard project settings.

4. **Deploy**
   - Click "Deploy"
   - Wait for the initial deployment to complete
   - Verify the build succeeds and the site is accessible

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

The marketing site requires the following environment variable for proper domain configuration:

### NEXT_PUBLIC_SITE_URL

- **Purpose**: Defines the canonical site URL for metadata, Open Graph tags, sitemap, and robots.txt
- **Production Value**: `https://www.wavesintelligence.app`
- **Required**: No (defaults to production URL if not set)
- **Environments**: Production, Preview (optional for preview environments)

To configure in Vercel:

1. Go to Vercel Dashboard → Project → Settings → Environment Variables
2. Add `NEXT_PUBLIC_SITE_URL` with value `https://www.wavesintelligence.app`
3. Select "Production" environment
4. Click "Save"
5. Redeploy for changes to take effect

For local development, create a `.env.local` file:
```bash
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

See `site/.env.example` for reference.

## Domain Configuration

The site is configured to use `www.wavesintelligence.app` as the canonical domain.

### DNS Setup in Vercel

1. **Go to Vercel Dashboard → Project → Settings → Domains**

2. **Add Primary Domain:**
   - Click "Add Domain"
   - Enter: `www.wavesintelligence.app`
   - Click "Add"
   - Follow DNS configuration instructions (if domain is managed elsewhere)
   - If domain is purchased via Vercel, DNS is automatically configured

3. **Add Non-WWW Domain:**
   - Click "Add Domain"
   - Enter: `wavesintelligence.app`
   - Click "Add"
   - Vercel will automatically redirect to www version (configured in vercel.json)

4. **Verify SSL Certificates:**
   - Both domains should show "Valid Certificate" status
   - SSL certificates are automatically provisioned by Vercel
   - HTTPS is enforced by default

### Redirect Configuration

The `vercel.json` file includes automatic redirects:
- `wavesintelligence.app` → `www.wavesintelligence.app` (HTTP 301 permanent redirect)
- All routes preserve their paths during redirect
- HTTPS is enforced on all routes

### Testing Domain Configuration

After DNS setup is complete:

1. **Test WWW Domain:**
   ```bash
   curl -I https://www.wavesintelligence.app
   # Should return HTTP/2 200
   ```

2. **Test Non-WWW Redirect:**
   ```bash
   curl -I https://wavesintelligence.app
   # Should return HTTP/2 301 with Location: https://www.wavesintelligence.app/
   ```

3. **Test Specific Routes:**
   - Home: `https://www.wavesintelligence.app/`
   - Product: `https://www.wavesintelligence.app/product`
   - Why: `https://www.wavesintelligence.app/why`
   - Demo: `https://www.wavesintelligence.app/demo`

4. **Verify SEO Configuration:**
   - Sitemap: `https://www.wavesintelligence.app/sitemap.xml`
   - Robots: `https://www.wavesintelligence.app/robots.txt`

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
- **Vercel Dashboard**: Project Settings → General (Root Directory, Build Command, Framework Preset)
- `vercel.json` - Domain redirect rules only
- `site/package.json` - Build scripts and dependencies
- `site/next.config.ts` - Next.js configuration

### vercel.json

The `vercel.json` file in the repository root contains **only** domain redirect configuration:

```json
{
  "redirects": [
    {
      "source": "/:path*",
      "has": [
        {
          "type": "host",
          "value": "wavesintelligence.app"
        }
      ],
      "destination": "https://www.wavesintelligence.app/:path*",
      "permanent": true
    }
  ]
}
```

This configuration:
- Redirects all traffic from non-www (`wavesintelligence.app`) to www (`www.wavesintelligence.app`) domain with HTTP 301 permanent redirect
- Preserves all URL paths during redirect
- Works for all routes (e.g., `/product`, `/platform`, etc.)

**Important:** Build settings (`buildCommand`, `rootDirectory`) are deprecated in `vercel.json` and must be configured in the Vercel Dashboard project settings instead.

### Vercel Dashboard Settings

Configure in Project Settings → General:
- **Root Directory**: `site` ⚠️ **CRITICAL** - Must be set manually
- **Build Command**: (leave empty, uses default `npm run build`)
- **Output Directory**: (leave empty, uses default `.next`)
- **Framework Preset**: Next.js (should auto-detect)

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
   
   **"Error: No framework detected"** or **"No package.json found"**
   - **Cause**: Root Directory is not set to `site`
   - **Solution**: 
     - Go to Vercel Dashboard → Project → Settings → General
     - Set "Root Directory" to `site`
     - Redeploy
   
   **"Missing dependencies" or "Module not found"**
   - **Cause**: Dependencies not installed correctly
   - **Solution**: Run `npm ci` locally in `/site` to verify package-lock.json is valid
   
   **TypeScript errors**
   - **Cause**: Type errors in code
   - **Solution**: Run `npm run build` locally in `/site` to identify errors
   
   **Linting errors**
   - **Cause**: ESLint violations
   - **Solution**: Run `npm run lint` locally in `/site` and fix issues

3. **Test Locally:**
   ```bash
   cd site
   npm ci
   npm run build
   ```

### Framework Not Detected

**Symptom:** Vercel cannot detect Next.js framework or builds fail with "No framework detected"

**Solution:**
1. Verify Root Directory is set to `site` in Vercel Dashboard → Settings → General
2. Ensure Framework Preset is set to "Next.js" (should auto-detect)
3. Verify `site/package.json` exists and has `next` as a dependency
4. Redeploy after making changes

### Redirect Not Working

**Symptom:** `wavesintelligence.app` does not redirect to `www.wavesintelligence.app`

**Solution:**
1. Verify both domains are added in Vercel Dashboard → Settings → Domains
2. Ensure `vercel.json` redirect configuration is in the repository root
3. Check that latest deployment includes the redirect configuration
4. Wait for DNS propagation (5-60 minutes after adding domains)
5. Test redirect: `curl -I https://wavesintelligence.app` (should return 301)

### Preview Deployment Not Showing

- Ensure the PR is from a branch in the same repository (not a fork)
- Check Vercel Dashboard → Project → Settings → Git
- Verify that "Automatic Deployments" is enabled for preview branches

### Production Deployment Failed

- Check that the `main` branch builds successfully locally
- Review Vercel build logs for specific errors
- Verify all environment variables are set (if needed)
- Ensure Root Directory is set to `site` in project settings

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
