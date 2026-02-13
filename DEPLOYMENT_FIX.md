# Fix for Vercel 404 NOT_FOUND Error

## Problem
The WAVES Intelligence™ marketing site shows a 404 NOT_FOUND error on Vercel because the Next.js application is located in the `site/` subdirectory, but Vercel is not configured to use this as the root directory.

## Solution

The Next.js marketing site lives in the `/site` subdirectory of this repository. To fix the 404 error, you need to configure Vercel to use `site` as the Root Directory.

### Step 1: Configure Root Directory in Vercel Dashboard

1. Go to your Vercel project dashboard
2. Navigate to **Settings** → **General**
3. Find the **Root Directory** setting
4. Click **Edit** next to Root Directory
5. Enter: `site`
6. Click **Save**

### Step 2: Verify Framework Detection

1. In the same Settings → General page, verify:
   - **Framework Preset** is set to `Next.js` (should auto-detect)
   - **Build Command** is empty or `npm run build`
   - **Output Directory** is empty or `.next`
   - **Install Command** is empty or `npm install`

### Step 3: Redeploy

1. Go to the **Deployments** tab
2. Click the three dots (...) on the latest deployment
3. Click **Redeploy**
4. Wait for the deployment to complete

## Verification

After redeployment, the site should:
- ✅ Load successfully at your Vercel URL (no 404 error)
- ✅ Show the WAVES Intelligence™ homepage
- ✅ Have working navigation to all pages

## Why This Fix Works

The repository structure is:
```
Waves-Simple/
├── site/              ← Next.js marketing site is HERE
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx      ← Root page
│   │       └── layout.tsx    ← Root layout
│   ├── package.json
│   ├── next.config.ts
│   └── ...
├── app.py             ← Python Streamlit app (separate)
├── vercel.json        ← Domain redirects only
└── ...
```

Without setting Root Directory to `site`, Vercel looks for the Next.js app in the repository root and doesn't find it, resulting in a 404 error.

## Alternative Solution (Not Recommended)

If you cannot access the Vercel Dashboard, you could move the entire `site/` directory contents to the repository root. However, this is **not recommended** because:
- It would mix the Next.js marketing site with the Python Streamlit application
- It would require updating many file paths and configurations
- It would make the repository structure less organized

## Reference

For complete deployment instructions, see [VERCEL_SETUP.md](./VERCEL_SETUP.md).
