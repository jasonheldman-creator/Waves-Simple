# WAVES Intelligence Marketing Site

This is the official marketing website for WAVES Intelligence, built with Next.js 14+, TypeScript, and Tailwind CSS.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Git

### Local Development

1. **Navigate to the site directory:**
   ```bash
   cd site
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000) to view the site.

The page auto-updates as you edit files in the `src` directory.

## 📁 Project Structure

```
site/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx         # Root layout with Navbar & Footer
│   │   ├── page.tsx           # Home page
│   │   ├── sitemap.ts         # Dynamic sitemap generation
│   │   ├── robots.ts          # Robots.txt configuration
│   │   ├── platform/          # Platform page
│   │   ├── console/           # Console page
│   │   ├── waves/             # Waves page
│   │   ├── architecture/      # Architecture page
│   │   ├── security/          # Security page
│   │   ├── company/           # Company page
│   │   ├── press/             # Press page
│   │   ├── contact/           # Contact page
│   │   └── api/
│   │       └── contact/       # Contact form API endpoint
│   │           └── route.ts
│   ├── components/            # Reusable UI components
│   │   ├── Navbar.tsx
│   │   ├── Footer.tsx
│   │   ├── Hero.tsx
│   │   ├── FeatureGrid.tsx
│   │   ├── WaveCards.tsx
│   │   ├── ScreenshotGallery.tsx
│   │   ├── ArchitectureDiagram.tsx
│   │   ├── CallToAction.tsx
│   │   └── ContactForm.tsx
│   └── content/               # Content management
│       └── siteContent.ts     # Site copy and text content
├── public/                    # Static assets
├── .env.example               # Environment variable template
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── README.md
```

## 🎨 Features

### Pages
- **Home (/)**: Hero section, features grid, and call-to-action
- **/platform**: Platform overview with features and screenshots
- **/console**: Console access information
- **/waves**: Investment waves showcase with 15 placeholder cards
- **/architecture**: System architecture diagram and details
- **/security**: Security features and compliance information
- **/company**: Company information and values
- **/press**: Press resources and news
- **/contact**: Contact form with validation

### Components
- **Navbar**: Sticky navigation with mobile menu and "Launch Console" CTA
- **Footer**: Site-wide footer with links and information
- **Hero**: Customizable hero sections with gradient backgrounds
- **FeatureGrid**: Responsive grid for displaying features
- **WaveCards**: Investment wave cards (15 placeholders by default)
- **ScreenshotGallery**: Platform screenshot showcase with captions
- **ArchitectureDiagram**: Inline SVG system architecture diagram
- **CallToAction**: Conversion-focused CTA sections
- **ContactForm**: Validated contact form with API integration

### Design System
- **Theme**: Dark institutional design with charcoal/black background
- **Accents**: Cyan (#00ffff) and green (#00ff88) neon highlights
- **Typography**: Premium, legible fonts (Geist Sans & Geist Mono)
- **Responsive**: Mobile-first design with Tailwind CSS
- **SEO**: Optimized metadata on all pages

## 🛠 Development

### Available Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run ESLint
npm run lint

# Format code with Prettier
npm run format

# Check code formatting
npm run format:check
```

### API Routes

#### POST /api/contact
Contact form submission endpoint.

**Request Body:**
```json
{
  "name": "string",
  "email": "string",
  "company": "string",
  "message": "string"
}
```

**Validation:**
- All fields are required
- Email must be valid format
- Name: 2-100 characters
- Company: 2-100 characters
- Message: 10-5000 characters

**Response:**
- Success: `{ "success": true, "message": "..." }`
- Error: `{ "error": "error message" }`

Submissions are logged server-side (no external email integration).

## 🚢 Deployment

### Deploy to Vercel (Recommended)

1. **Push your code to GitHub**

2. **Import to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository: `jasonheldman-creator/Waves-Simple`
   
3. **Configure Project Settings (CRITICAL):**
   - **Root Directory**: Set to `site` ⚠️ **REQUIRED**
   - **Framework Preset**: Next.js (should auto-detect)
   - **Build Command**: Leave empty (uses default `npm run build`)
   - **Output Directory**: Leave empty (uses default `.next`)
   - Click "Deploy"

4. **Configure Domain:**
   - In Vercel project settings, go to "Domains"
   - Add your custom domain: `www.wavesintelligence.app`
   - Add non-www redirect: `wavesintelligence.app` (will auto-redirect to www via `vercel.json`)
   - Follow Vercel's instructions to configure DNS

5. **Set Environment Variables:**
   - Go to Vercel project settings → Environment Variables
   - Add: `NEXT_PUBLIC_SITE_URL=https://www.wavesintelligence.app`
   - Select "Production" environment
   - Save and redeploy

### Environment Variables

The site uses the following environment variables:

- **NEXT_PUBLIC_SITE_URL**: Canonical site URL (default: `https://www.wavesintelligence.app`)
  - Used for metadata, Open Graph tags, sitemap, and robots.txt
  - Set in Vercel for production deployments
  - For local development, create `.env.local` with `NEXT_PUBLIC_SITE_URL=http://localhost:3000`

- **NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL**: External URL for live performance CSV data (optional)
  - If set, the site will fetch performance data from this URL instead of generating from local files
  - The CSV must follow the format: `wave_id,wave_name,status,performance_1d,performance_30d,performance_ytd,last_updated`
  - If not set or if fetch fails, falls back to generating from local `wave_history.csv`
  - Set in Vercel project settings → Environment Variables for production deployments
  - Example: `NEXT_PUBLIC_LIVE_SNAPSHOT_CSV_URL=https://data.example.com/waves/live_snapshot.csv`

See `.env.example` for reference.

## 🌐 DNS Configuration

### Marketing Site
The site uses `www.wavesintelligence.app` as the canonical domain with automatic redirect from non-www.

**Primary Domain:** `www.wavesintelligence.app`
**Secondary Domain:** `wavesintelligence.app` (redirects to www)

**DNS Settings in Vercel:**
1. Add both domains in Vercel Dashboard → Project → Settings → Domains
2. Vercel automatically provisions SSL certificates
3. The `vercel.json` configuration handles the redirect from non-www to www

**Automatic Redirect:**
- `wavesintelligence.app/*` → `www.wavesintelligence.app/*` (HTTP 301 permanent)
- All routes preserve their paths during redirect
- HTTPS is enforced on all routes

### Console Subdomain
If you have a separate console application:

**Domain:** `console.wavesintelligence.app`
**DNS Settings:**
- Type: `CNAME`
- Value: Your console hosting URL

### Example DNS Configuration

```
# Marketing Site (Next.js on Vercel)
www.wavesintelligence.app      CNAME  cname.vercel-dns.com
wavesintelligence.app          A      76.76.21.21  # Vercel's IP (auto-configured)

# Console (if hosted separately)
console.wavesintelligence.app  CNAME  your-console-app.example.com
```

**Note:** If your domain is purchased through Vercel, DNS is automatically configured. Otherwise, follow Vercel's DNS configuration instructions.

## 📝 Content Management

All site copy is managed in `src/content/siteContent.ts`. Edit this file to update:
- Page titles and descriptions
- Hero section content
- Feature lists
- Call-to-action text
- Contact information

## 🎯 SEO Optimization

Each page includes:
- Custom title and description meta tags
- Open Graph tags for social sharing
- Canonical URL configuration via `metadataBase`
- Semantic HTML structure
- Mobile-responsive design
- Fast page load times

**SEO Files:**
- `/sitemap.xml`: Automatically generated sitemap with all routes
- `/robots.txt`: Crawler configuration with sitemap reference
- All URLs point to canonical domain: `www.wavesintelligence.app`

## 🔒 Security

- Form validation on both client and server
- CSRF protection via Next.js
- No sensitive data in client-side code
- Server-side logging only (no external data transmission)

## 📚 Tech Stack

- **Framework:** Next.js 14+ (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS 4
- **Fonts:** Geist Sans & Geist Mono
- **Linting:** ESLint
- **Formatting:** Prettier

## 🤝 Contributing

1. Make changes in the `site` directory
2. Test locally with `npm run dev`
3. Build to verify: `npm run build`
4. Format code: `npm run format`
5. Commit and push changes

## 📞 Support

For questions or issues:
- Technical: Review this README
- Content: Edit `src/content/siteContent.ts`
- Components: Check `src/components/`
- Deployment: See Vercel documentation

## 📄 License

Proprietary - WAVES Intelligence

---

Built with ❤️ using Next.js 14+
