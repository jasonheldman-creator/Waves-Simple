# Domain Migration Checklist for wavesintelligence.app

This checklist guides you through completing the domain migration to `www.wavesintelligence.app`.

## ✅ Code Changes (Completed)

All code changes have been completed and committed to this branch:

- [x] Updated `site/src/app/layout.tsx` with canonical URL metadata
- [x] Created `site/src/app/sitemap.ts` for dynamic sitemap generation
- [x] Created `site/src/app/robots.ts` for SEO crawler configuration
- [x] Created `site/.env.example` documenting environment variables
- [x] Updated `vercel.json` with non-www to www redirect
- [x] Updated `.gitignore` to allow `.env.example`
- [x] Updated documentation (DEPLOYMENT.md and site/README.md)
- [x] Build verification completed successfully
- [x] Security scan completed with no issues

## 🔧 Vercel Dashboard Configuration (Manual Steps Required)

Complete these steps in the Vercel dashboard after merging this PR:

### Step 1: Add Domains

1. **Navigate to Project Settings:**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Select your WAVES Intelligence project
   - Click "Settings" → "Domains"

2. **Add Primary WWW Domain:**
   - Click "Add Domain"
   - Enter: `www.wavesintelligence.app`
   - Click "Add"
   - Status: Wait for SSL certificate provisioning (usually automatic)
   - Expected result: Shows "Valid Certificate" ✓

3. **Add Non-WWW Domain:**
   - Click "Add Domain" again
   - Enter: `wavesintelligence.app`
   - Click "Add"
   - Status: Will show as redirect domain
   - Expected result: Automatically redirects to www version ✓

### Step 2: Configure DNS

**If domain is purchased through Vercel:**
- DNS is automatically configured ✓
- Skip to Step 3

**If domain is managed elsewhere:**
1. In Vercel, you'll see DNS configuration instructions
2. Copy the provided DNS records
3. Add them to your domain registrar:
   ```
   Type: A
   Name: @
   Value: [Vercel's IP address - shown in dashboard]

   Type: CNAME
   Name: www
   Value: cname.vercel-dns.com
   ```
4. Wait for DNS propagation (typically 5-60 minutes, can take up to 48 hours)

### Step 3: Set Environment Variables

1. **Navigate to Environment Variables:**
   - In Vercel Dashboard → Project → Settings
   - Click "Environment Variables"

2. **Add NEXT_PUBLIC_SITE_URL:**
   - Variable Name: `NEXT_PUBLIC_SITE_URL`
   - Value: `https://www.wavesintelligence.app`
   - Environment: Select "Production"
   - Click "Save"

3. **Trigger Redeploy:**
   - Go to "Deployments" tab
   - Click the "..." menu on the latest production deployment
   - Select "Redeploy"
   - Verify the new deployment succeeds

### Step 4: Verify SSL Certificates

1. **Check Certificate Status:**
   - In Settings → Domains
   - Both domains should show "Valid Certificate"
   - If showing "Pending" or "Error", wait a few minutes and refresh

2. **Force HTTPS:**
   - Should be enabled by default
   - Verify all HTTP requests redirect to HTTPS

## 🧪 Testing & Verification

After completing Vercel configuration, test the following:

### Domain and Redirect Tests

```bash
# Test WWW domain loads correctly
curl -I https://www.wavesintelligence.app
# Expected: HTTP/2 200 OK

# Test non-WWW redirects to WWW
curl -I https://wavesintelligence.app
# Expected: HTTP/2 301 Moved Permanently
# Expected: Location: https://www.wavesintelligence.app/

# Test HTTP redirects to HTTPS
curl -I http://www.wavesintelligence.app
# Expected: HTTP/2 301 or 308 with Location: https://...

# Test redirect preserves paths
curl -I https://wavesintelligence.app/product
# Expected: HTTP/2 301 with Location: https://www.wavesintelligence.app/product
```

### Route Verification

Test all main routes load correctly:

- [ ] Home: https://www.wavesintelligence.app/
- [ ] Product: https://www.wavesintelligence.app/product
- [ ] Why: https://www.wavesintelligence.app/why
- [ ] Demo: https://www.wavesintelligence.app/demo
- [ ] Platform: https://www.wavesintelligence.app/platform
- [ ] Architecture: https://www.wavesintelligence.app/architecture
- [ ] Security: https://www.wavesintelligence.app/security
- [ ] Governance: https://www.wavesintelligence.app/governance
- [ ] Console: https://www.wavesintelligence.app/console
- [ ] Company: https://www.wavesintelligence.app/company
- [ ] Contact: https://www.wavesintelligence.app/contact
- [ ] Press: https://www.wavesintelligence.app/press

### SEO Configuration

- [ ] Sitemap accessible: https://www.wavesintelligence.app/sitemap.xml
- [ ] Robots.txt accessible: https://www.wavesintelligence.app/robots.txt
- [ ] Sitemap contains all expected routes
- [ ] All sitemap URLs use www.wavesintelligence.app

### Metadata Verification

Open any page and check the HTML source (View → Developer → View Source):

- [ ] `<meta property="og:url" content="https://www.wavesintelligence.app...">` present
- [ ] `<meta property="og:site_name" content="WAVES Intelligence">` present
- [ ] Canonical URL in `<link rel="canonical">` uses www.wavesintelligence.app
- [ ] No references to old domains in metadata

### Cross-Browser Testing

Test in multiple browsers:

- [ ] Chrome (Desktop)
- [ ] Firefox (Desktop)
- [ ] Safari (Desktop)
- [ ] Safari (iOS/Mobile)
- [ ] Chrome (Android/Mobile)

### Performance & Security

- [ ] SSL certificate is valid (green lock icon in browser)
- [ ] No mixed content warnings
- [ ] Pages load quickly (< 3 seconds)
- [ ] Mobile responsive design works correctly

## 🔍 Troubleshooting

### DNS not propagating
- **Problem:** Domain doesn't resolve after adding to Vercel
- **Solution:** 
  - Wait 5-60 minutes for DNS propagation
  - Check DNS with: `nslookup www.wavesintelligence.app`
  - Verify DNS records at your registrar match Vercel's instructions

### SSL Certificate Pending
- **Problem:** Certificate shows "Pending" status
- **Solution:**
  - Wait 5-10 minutes and refresh
  - Ensure DNS is correctly configured
  - If persists > 30 min, check Vercel Status page

### Redirect Not Working
- **Problem:** Non-www doesn't redirect to www
- **Solution:**
  - Verify `vercel.json` is in repository root
  - Ensure latest deployment includes the vercel.json changes
  - Check deployment logs for any errors
  - May need to redeploy after DNS propagation completes

### Environment Variable Not Applied
- **Problem:** Site still uses default URL
- **Solution:**
  - Verify environment variable is set for "Production" environment
  - Trigger a new deployment (redeploy doesn't always pick up env changes)
  - Check deployment logs to confirm variable is being used

### Old Domain Still Showing
- **Problem:** Metadata or sitemap shows old domain
- **Solution:**
  - Clear browser cache
  - Verify NEXT_PUBLIC_SITE_URL is set correctly
  - Check build logs to ensure environment variable is being used
  - May need to do a fresh deployment (not redeploy)

## 📞 Support Resources

- **Vercel Documentation:** https://vercel.com/docs
- **Vercel Status:** https://www.vercel-status.com
- **DNS Checker:** https://dnschecker.org
- **SSL Checker:** https://www.ssllabs.com/ssltest/

## ✨ Post-Migration Tasks (Optional)

After successful migration:

- [ ] Update Google Search Console with new domain
- [ ] Update any Google Analytics properties
- [ ] Submit new sitemap to Google Search Console
- [ ] Update any external links pointing to the site
- [ ] Update social media profiles with new domain
- [ ] Notify stakeholders of new domain
- [ ] Monitor analytics for traffic patterns
- [ ] Set up 301 redirects from old domain (if applicable)

## 🎉 Migration Complete!

Once all tests pass and the domain is live:

1. Mark this checklist as complete
2. Archive this document for reference
3. Update any internal documentation with the new domain
4. Celebrate! 🎊

---

**Migration Date:** _____________  
**Completed By:** _____________  
**Verification Date:** _____________  
**Verified By:** _____________
