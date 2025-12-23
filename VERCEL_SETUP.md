# Vercel Project Configuration Guide

This guide provides the exact steps to configure the WAVES Intelligence™ marketing site deployment on Vercel.

## Prerequisites

- GitHub repository: `jasonheldman-creator/Waves-Simple`
- Vercel account (sign up at [vercel.com](https://vercel.com) using GitHub)
- Domain: `wavesintelligence.app` (optional, can configure later)

## Step 1: Import Project to Vercel

1. **Log in to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Login" and sign in with GitHub

2. **Create New Project**
   - Click "Add New..." → "Project"
   - Select "Import from Git"
   - Choose the repository: `jasonheldman-creator/Waves-Simple`

3. **Configure Build Settings** ⚠️ **CRITICAL STEP**
   
   In the "Configure Project" screen, set the following:
   
   | Setting | Value | Notes |
   |---------|-------|-------|
   | **Framework Preset** | `Next.js` | Should auto-detect; if not, select manually |
   | **Root Directory** | `site` | ⚠️ **MUST BE SET** - Click "Edit" and enter `site` |
   | **Build Command** | (leave empty) | Uses default: `npm run build` |
   | **Output Directory** | (leave empty) | Uses default: `.next` |
   | **Install Command** | (leave empty) | Uses default: `npm install` |
   | **Node.js Version** | `20.x` | Recommended version |

4. **Deploy**
   - Click "Deploy" button
   - Wait for the initial deployment to complete (2-5 minutes)
   - Verify deployment succeeds with "Congratulations!" message

## Step 2: Verify Framework Detection

After first deployment:

1. Go to Project → Settings → General
2. Verify "Framework Preset" shows "Next.js"
3. Verify "Root Directory" shows `site`
4. If either is incorrect, update and redeploy

## Step 3: Configure Custom Domain (Optional)

Skip this step if deploying to default Vercel domain (`*.vercel.app`).

### Add Domains

1. **Go to Domain Settings**
   - In Vercel Dashboard → Your Project → Settings → Domains

2. **Add WWW Domain (Primary)**
   - Click "Add Domain"
   - Enter: `www.wavesintelligence.app`
   - Click "Add"
   - Status will show "Pending" or "Invalid Configuration" initially

3. **Add Non-WWW Domain (Redirect)**
   - Click "Add Domain" again
   - Enter: `wavesintelligence.app`
   - Click "Add"
   - This domain will automatically redirect to the www version (configured in `vercel.json`)

### Configure DNS

**If domain is purchased through Vercel:**
- DNS is automatically configured ✓
- Skip to "Verify SSL Certificates"

**If domain is managed elsewhere (e.g., Namecheap, GoDaddy, Cloudflare):**

1. Vercel will display DNS configuration instructions
2. Copy the DNS records shown
3. Add them to your domain registrar's DNS settings:

   ```
   Type: A
   Name: @ (or leave blank for root domain)
   Value: 76.76.21.21 (or IP shown in Vercel)
   TTL: 3600 (or automatic)

   Type: CNAME
   Name: www
   Value: cname.vercel-dns.com
   TTL: 3600 (or automatic)
   ```

4. Save changes at your domain registrar
5. Wait for DNS propagation (5-60 minutes, up to 48 hours)

### Verify SSL Certificates

1. In Settings → Domains, wait for SSL status to change
2. Both domains should show "Valid Certificate" ✓
3. If "Pending" for > 30 minutes, check DNS configuration
4. HTTPS is automatically enforced

## Step 4: Set Environment Variables

Required for proper metadata and SEO configuration.

1. **Navigate to Environment Variables**
   - In Vercel Dashboard → Project → Settings → Environment Variables

2. **Add Production URL Variable**
   - Click "Add New"
   - Name: `NEXT_PUBLIC_SITE_URL`
   - Value: `https://www.wavesintelligence.app`
   - Environment: Select "Production" only
   - Click "Save"

3. **Trigger Redeploy**
   - Go to Deployments tab
   - Click the "..." menu on the latest production deployment
   - Select "Redeploy"
   - Wait for deployment to complete

## Step 5: Verify Deployment

### Test Site Access

1. **Visit Production URL**
   - Default Vercel URL: `https://[project-name].vercel.app`
   - Custom domain: `https://www.wavesintelligence.app`

2. **Test Key Routes**
   - Home: `/`
   - Platform: `/platform`
   - Product: `/product`
   - Contact: `/contact`
   - All routes should load without errors

### Test Domain Redirects (if custom domain configured)

```bash
# Test WWW domain returns 200 OK
curl -I https://www.wavesintelligence.app

# Test non-WWW redirects to WWW (should return 301)
curl -I https://wavesintelligence.app

# Test redirect preserves paths
curl -I https://wavesintelligence.app/platform
# Should redirect to: https://www.wavesintelligence.app/platform
```

### Test SEO Configuration

1. **Check Sitemap**
   - URL: `https://www.wavesintelligence.app/sitemap.xml`
   - Should list all site routes
   - All URLs should use www.wavesintelligence.app

2. **Check Robots.txt**
   - URL: `https://www.wavesintelligence.app/robots.txt`
   - Should reference sitemap URL

3. **Check Page Metadata**
   - Open any page
   - View page source (right-click → View Page Source)
   - Verify `<meta property="og:url" content="https://www.wavesintelligence.app...">` is present
   - Verify canonical URL matches www domain

## Step 6: Configure Automatic Deployments

Ensure automatic deployments are enabled for continuous deployment.

1. **Go to Git Settings**
   - In Vercel Dashboard → Project → Settings → Git

2. **Verify Production Branch**
   - Production Branch: `main`
   - ✓ Enabled

3. **Verify Preview Deployments**
   - ✓ Automatic Deployments enabled
   - All branches with PRs will get preview deployments

## Troubleshooting

### Build Fails with "No Framework Detected"

**Problem:** Vercel cannot find the Next.js application

**Solution:**
1. Go to Settings → General
2. Set "Root Directory" to `site`
3. Ensure Framework Preset is "Next.js"
4. Redeploy

### Build Fails with "Module Not Found" or "Cannot Find Package"

**Problem:** Dependencies not installed correctly

**Solution:**
1. Verify `site/package.json` and `site/package-lock.json` exist in repository
2. Try local build: `cd site && npm ci && npm run build`
3. If local build succeeds, redeploy on Vercel
4. Check build logs for specific missing packages

### Redirect Not Working

**Problem:** Non-www domain doesn't redirect to www

**Solution:**
1. Verify both domains are added in Settings → Domains
2. Check that `vercel.json` is in repository root with redirect configuration
3. Ensure latest deployment includes the vercel.json file
4. Wait for DNS propagation if domain was just added
5. Clear browser cache and test in incognito mode

### SSL Certificate Pending

**Problem:** SSL certificate shows "Pending" status

**Solution:**
1. Wait 5-10 minutes and refresh the page
2. Verify DNS records are configured correctly at your registrar
3. Check domain resolves: `nslookup www.wavesintelligence.app`
4. If still pending after 30 minutes, check [Vercel Status](https://www.vercel-status.com)

### Environment Variable Not Working

**Problem:** Site doesn't use the configured environment variable

**Solution:**
1. Verify variable name is exactly: `NEXT_PUBLIC_SITE_URL`
2. Ensure it's set for "Production" environment
3. Check the value has no extra spaces: `https://www.wavesintelligence.app`
4. Trigger a NEW deployment (not redeploy) for env var changes to take effect
5. Check build logs to see if variable is being used

## Configuration Summary

✅ **Completed when all of the following are true:**

- [ ] Project imported to Vercel from GitHub
- [ ] Root Directory set to `site` in project settings
- [ ] Framework Preset shows "Next.js"
- [ ] Initial deployment succeeds
- [ ] Custom domains added (if applicable)
- [ ] DNS configured at domain registrar (if applicable)
- [ ] SSL certificates showing "Valid" (if using custom domain)
- [ ] Environment variable `NEXT_PUBLIC_SITE_URL` set
- [ ] All test routes accessible
- [ ] Non-www to www redirect working (if using custom domain)
- [ ] Sitemap and robots.txt accessible
- [ ] Preview deployments working for PRs

## Support Resources

- **Vercel Documentation**: https://vercel.com/docs
- **Next.js on Vercel**: https://vercel.com/docs/frameworks/nextjs
- **Vercel Status**: https://www.vercel-status.com
- **DNS Checker**: https://dnschecker.org
- **SSL Checker**: https://www.ssllabs.com/ssltest/

## Additional Notes

### Why Root Directory Must Be Set

The repository structure is:
```
Waves-Simple/
├── site/              ← Next.js app is here
│   ├── package.json
│   ├── next.config.ts
│   └── src/
├── app.py            ← Python Streamlit app
├── requirements.txt
└── vercel.json       ← Redirect configuration
```

Vercel needs to know the Next.js app is in the `/site` subdirectory, not the repository root. Without setting Root Directory to `site`, Vercel will:
- Look for `package.json` in the root (won't find it)
- Fail with "No framework detected"
- Not be able to build the site

### Why vercel.json is Minimal

The `vercel.json` file contains **only** redirect rules for domain management:
- Non-www → www redirect
- No build configuration

Build settings (`buildCommand`, `rootDirectory`) were previously in `vercel.json` but are now **deprecated**. They must be configured in the Vercel Dashboard → Project Settings → General instead.

This ensures:
- ✅ Compatibility with latest Vercel platform
- ✅ Better separation of concerns
- ✅ Easier project portability
- ✅ No deprecated property warnings
