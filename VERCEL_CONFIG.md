# Vercel Configuration Quick Reference

## ⚠️ CRITICAL CONFIGURATION

To fix 404 NOT_FOUND errors, set these values in Vercel Dashboard:

### Project Settings → General

```
Root Directory:     site           ← MUST BE SET
Framework Preset:   Next.js        ← Auto-detected
Build Command:      (empty)        ← Uses default
Output Directory:   (empty)        ← Uses default
Install Command:    (empty)        ← Uses default
Node.js Version:    20.x           ← Recommended
```

## ✅ Configuration Validation Checklist

Before deploying, verify:

- [ ] Repository imported to Vercel from GitHub
- [ ] Root Directory set to `site` in Project Settings
- [ ] Framework Preset shows "Next.js"
- [ ] No custom build commands (unless needed)
- [ ] Deployment triggered and completed successfully

## 🔍 How to Set Root Directory

1. Open Vercel Dashboard
2. Select your project
3. Go to **Settings** tab
4. Click **General** in the left sidebar
5. Scroll to **Root Directory** section
6. Click **Edit** button
7. Enter: `site`
8. Click **Save**
9. Go to **Deployments** tab
10. Click **Redeploy** on latest deployment

## 📁 Repository Structure (for reference)

```
Waves-Simple/
├── site/                    ← Root Directory points here
│   ├── package.json         ← Next.js dependencies
│   ├── next.config.ts       ← Next.js config
│   ├── src/
│   │   └── app/
│   │       ├── layout.tsx   ← Root layout (required)
│   │       ├── page.tsx     ← Home page (required)
│   │       └── ...          ← Other pages
│   └── ...
├── vercel.json              ← Redirects only (no build configuration)
├── app.py                   ← Python app (not deployed to Vercel)
└── requirements.txt         ← Python deps (not used by Vercel)
```

## 🚨 Common Mistakes

### ❌ DON'T

- Leave Root Directory empty
- Set Root Directory to `/`
- Add build commands to vercel.json (deprecated)
- Try to deploy the Python app to Vercel

### ✅ DO

- Set Root Directory to exactly `site`
- Let Vercel auto-detect Next.js
- Use default build commands
- Configure in Dashboard, not vercel.json

## 🎯 Expected Results

After correct configuration:

- ✅ Home page loads: `https://your-project.vercel.app/`
- ✅ All routes work: `/platform`, `/product`, `/contact`, etc.
- ✅ No 404 errors
- ✅ Framework detected as "Next.js" in Settings
- ✅ Build logs show successful Next.js build

## 🆘 Still Getting 404?

### Double-check:

1. **Root Directory** is set to `site` (NOT `./site`, NOT `/site`, just `site`)
2. **Framework Preset** shows "Next.js"
3. You clicked **Save** after changing Root Directory
4. You **Redeployed** after changing settings (not just Deploy)
5. Clear browser cache / try incognito mode

### If still failing:

1. Check deployment logs for errors
2. Verify `site/package.json` exists in your repo
3. Ensure latest code is pushed to GitHub
4. Try deleting project and re-importing with correct settings from start

## 📚 Full Documentation

- **Quick Start:** [DEPLOY_TO_VERCEL.md](./DEPLOY_TO_VERCEL.md)
- **Complete Guide:** [VERCEL_SETUP.md](./VERCEL_SETUP.md)
- **Troubleshooting:** See "Troubleshooting" section in VERCEL_SETUP.md

## 🎓 Why This Configuration?

This is a **monorepo** with multiple applications:
- `/site` - Next.js marketing website (deployed to Vercel)
- Root directory - Python Streamlit app (deployed elsewhere)

Vercel needs to know which application to deploy, hence the Root Directory setting.

---

**Ready to deploy?** Follow [DEPLOY_TO_VERCEL.md](./DEPLOY_TO_VERCEL.md) step-by-step.
