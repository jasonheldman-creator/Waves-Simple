# Deploy to Vercel - Quick Start Guide

## 🚨 CRITICAL: Root Directory Must Be Set

The WAVES Intelligence™ marketing site is a Next.js application located in the `/site` subdirectory of this repository. **You MUST configure Vercel to use `site` as the Root Directory**, otherwise you will get a 404 NOT_FOUND error.

## Quick Deploy Steps

### 1. Import Project to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click "Add New..." → "Project"
3. Select this repository: `jasonheldman-creator/Waves-Simple`

### 2. Configure Build Settings ⚠️ CRITICAL

In the "Configure Project" screen:

| Setting | Value | Required? |
|---------|-------|-----------|
| **Root Directory** | `site` | ✅ **YES - MUST BE SET** |
| **Framework Preset** | Next.js | ✅ (auto-detected) |
| **Build Command** | (leave empty) | No (uses default) |
| **Output Directory** | (leave empty) | No (uses default) |
| **Install Command** | (leave empty) | No (uses default) |

**To set Root Directory:**
- Click "Edit" next to "Root Directory"
- Enter: `site`
- Click "Continue"

### 3. Deploy

Click "Deploy" and wait for the build to complete (2-5 minutes).

## Why This Is Required

The repository structure is:

```
Waves-Simple/
├── site/              ← Next.js app is HERE
│   ├── package.json
│   ├── next.config.ts
│   └── src/
├── app.py            ← Python Streamlit app
├── requirements.txt
└── vercel.json       ← Redirect configuration
```

Without setting Root Directory to `site`:
- Vercel looks for `package.json` in the root directory
- Cannot find Next.js configuration
- Returns "No framework detected" or 404 error
- Deployment fails

## Verification

After deployment:

1. Go to Project → Settings → General
2. Verify "Root Directory" shows `site`
3. Verify "Framework Preset" shows `Next.js`

## Common Issues

### 404 NOT_FOUND Error

**Problem:** Vercel shows 404 on all routes

**Solution:**
1. Go to Project Settings → General
2. Set "Root Directory" to `site`
3. Click "Save"
4. Redeploy the project

### "No Framework Detected" Error

**Problem:** Vercel cannot find Next.js

**Solution:**
1. Verify "Root Directory" is set to `site`
2. Verify `site/package.json` exists in the repository
3. Redeploy

## Additional Configuration

### Environment Variables (Optional)

For production deployment, set:

- `NEXT_PUBLIC_SITE_URL`: Your production URL (e.g., `https://www.wavesintelligence.app`)

### Custom Domain (Optional)

1. Go to Project Settings → Domains
2. Add your domain (e.g., `www.wavesintelligence.app`)
3. Follow Vercel's DNS configuration instructions (DNS propagation typically takes 24-48 hours)
4. The `vercel.json` file handles www/non-www redirects

## Complete Documentation

For detailed deployment instructions, see:
- [VERCEL_SETUP.md](./VERCEL_SETUP.md) - Full deployment guide
- [site/README.md](./site/README.md) - Next.js app documentation

## Validation

To validate your configuration before deploying:

```bash
python3 validate_vercel_config.py
```

This checks:
- ✅ No deprecated properties in vercel.json
- ✅ Redirect rules won't cause loops
- ✅ Configuration follows Vercel best practices

---

**Need Help?** See [VERCEL_SETUP.md](./VERCEL_SETUP.md) for step-by-step instructions with screenshots and troubleshooting.
