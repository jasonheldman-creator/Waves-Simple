# WAVES Intelligence Marketing Site

This is the official marketing website for WAVES Intelligence, built with Next.js 14+, TypeScript, and Tailwind CSS.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Git
- GitHub Personal Access Token (for snapshot rebuild functionality)

### Local Development

1. **Navigate to the site directory:**
   ```bash
   cd site
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env.local
   ```
   
   Edit `.env.local` and set:
   - `NEXT_PUBLIC_SITE_URL=http://localhost:3000`
   - `GITHUB_TOKEN=your_github_personal_access_token`
   - `GITHUB_REPO=jasonheldman-creator/Waves-Simple`
   - `GITHUB_BRANCH=main`

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000) to view the site.

The page auto-updates as you edit files in the `src` directory.

## 📁 Project Structure

```
site/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── layout.tsx         # Root layout with Navbar & Footer
│   │   ├── page.tsx           # Snapshot Console (main page)
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
│   │       ├── snapshot/      # GET /api/snapshot
│   │       │   └── route.ts
│   │       ├── rebuild-snapshot/  # POST /api/rebuild-snapshot
│   │       │   └── route.ts
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
- **Home (/)**: WAVES Intelligence™ Snapshot Console - Live market data for 28 canonical waves
- **/platform**: Platform overview with features and screenshots
- **/console**: Console access information
- **/waves**: Investment waves showcase with 15 placeholder cards
- **/architecture**: System architecture diagram and details
- **/security**: Security features and compliance information
- **/company**: Company information and values
- **/press**: Press resources and news
- **/contact**: Contact form with validation

### Snapshot Console Features
- **Live Data Display**: Real-time view of 28 canonical waves
- **Market Returns**: 1D, 30D, 60D, and 365D performance metrics
- **Rebuild Functionality**: Fetch live market data and commit to GitHub
- **Status Tracking**: Visual indicators for data quality and missing tickers
- **Auto-refresh**: Manual and automatic snapshot updates

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

#### GET /api/snapshot
Fetch the current live snapshot from `data/live_snapshot.csv`.

**Query Parameters:**
- `format=csv` (optional): Download CSV file instead of JSON

**Response (JSON):**
```json
{
  "count": 28,
  "timestamp": "2026-01-02T10:00:00.000Z",
  "data": [
    {
      "Wave_ID": "ai_cloud_megacap_wave",
      "Wave": "AI & Cloud MegaCap Wave",
      "Return_1D": "0.015000",
      "Return_30D": "0.125000",
      "Return_60D": "0.240000",
      "Return_365D": "0.580000",
      "AsOfUTC": "2026-01-02T10:00:00.000Z",
      "DataStatus": "OK",
      "MissingTickers": ""
    }
  ]
}
```

**Response (CSV):**
Returns the raw CSV file for download.

**Validation:**
- Returns HTTP 500 if snapshot doesn't contain exactly 28 waves

#### POST /api/rebuild-snapshot
Rebuild the live snapshot by fetching fresh market data and committing to GitHub.

**Process:**
1. Load canonical 28 waves from `wave_weights.csv`
2. Fetch market data for all tickers:
   - Crypto (ending in `-USD`): CoinGecko API
   - Equities: Stooq CSV endpoint
3. Compute weighted wave returns for 1D/30D/60D/365D periods
4. Validate exactly 28 waves with complete data
5. Commit to `data/live_snapshot.csv` via GitHub API (only if valid)

**Response (Success):**
```json
{
  "success": true,
  "message": "Snapshot rebuilt and committed successfully",
  "timestamp": "2026-01-02T10:00:00.000Z",
  "waveCount": 28,
  "commit": "Live snapshot update: 28 waves @ 2026-01-02T10:00:00.000Z",
  "summary": {
    "totalWaves": 28,
    "validWaves": 28,
    "tickersFetched": 180,
    "totalTickers": 185
  }
}
```

**Response (Failure):**
```json
{
  "success": false,
  "error": "Validation failed",
  "message": "Only 25/28 waves have valid data",
  "failedWaves": ["Wave A", "Wave B", "Wave C"],
  "details": [
    {
      "wave": "Wave A",
      "status": "FAILED",
      "missingTickers": "TICKER1;TICKER2"
    }
  ]
}
```

**Validation Rules:**
- Must have exactly 28 unique waves from `wave_weights.csv`
- All waves must have at least one valid return period
- GitHub commit only happens if validation passes
- No changes to repository if validation fails

**Required CSV Columns:**
| Column | Description |
|--------|-------------|
| `Wave_ID` | Slugified wave name (e.g., `ai_cloud_megacap_wave`) |
| `Wave` | Canonical wave name (e.g., `AI & Cloud MegaCap Wave`) |
| `Return_1D` | 1-day return as decimal (0.05 = 5%) |
| `Return_30D` | 30-day return as decimal |
| `Return_60D` | 60-day return as decimal |
| `Return_365D` | 365-day return as decimal |
| `AsOfUTC` | ISO timestamp of data snapshot |
| `DataStatus` | `OK` or `FAILED` |
| `MissingTickers` | Semicolon-separated list of failed tickers |

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
   - Add the following variables:
     - `NEXT_PUBLIC_SITE_URL=https://www.wavesintelligence.app`
     - `GITHUB_TOKEN=your_github_personal_access_token`
     - `GITHUB_REPO=jasonheldman-creator/Waves-Simple`
     - `GITHUB_BRANCH=main`
   - Select "Production" environment
   - Save and redeploy

### Environment Variables

The site requires the following environment variables:

#### Required for All Deployments
- **NEXT_PUBLIC_SITE_URL**: Canonical site URL (default: `https://www.wavesintelligence.app`)
  - Used for metadata, Open Graph tags, sitemap, and robots.txt
  - Set in Vercel for production deployments
  - For local development, create `.env.local` with `NEXT_PUBLIC_SITE_URL=http://localhost:3000`

#### Required for Snapshot Rebuild
- **GITHUB_TOKEN**: GitHub Personal Access Token with `repo` scope
  - Create at: https://github.com/settings/tokens
  - Required permissions: `repo` (full control of private repositories)
  - Used to commit snapshot updates to `data/live_snapshot.csv`
  
- **GITHUB_REPO**: Repository in format `owner/repo` (default: `jasonheldman-creator/Waves-Simple`)
  - The repository where snapshots are committed
  
- **GITHUB_BRANCH**: Branch name for commits (default: `main`)
  - Target branch for snapshot commits

See `.env.example` for reference.

### GitHub Token Setup

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: `WAVES Snapshot Console`
4. Expiration: Choose appropriate duration
5. Scopes: Select `repo` (full control of private repositories)
6. Click "Generate token"
7. Copy the token immediately (you won't see it again!)
8. Add to Vercel environment variables as `GITHUB_TOKEN`

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
- GitHub token stored securely in environment variables (server-side only)
- No sensitive data in client-side code
- Server-side logging only (no external data transmission)

## 📊 Market Data Sources

### Cryptocurrency Data
- **Provider**: CoinGecko API (Free tier)
- **Endpoint**: `https://api.coingecko.com/api/v3/coins/{id}/market_chart`
- **Rate Limits**: ~50 calls/minute (free tier)
- **Tickers**: All tickers ending in `-USD` (e.g., `BTC-USD`, `ETH-USD`)

### Equity Data
- **Provider**: Stooq
- **Endpoint**: `https://stooq.com/q/d/l/?s={ticker}.US&i=d`
- **Format**: CSV (Date, Open, High, Low, Close, Volume)
- **Tickers**: All standard equity tickers (e.g., `AAPL`, `MSFT`, `TSLA`)

### Error Handling
- Ticker-level failures are logged and tracked
- Wave-level failures occur only when ALL tickers fail
- Missing ticker data is reported in the `MissingTickers` column
- Snapshot commit only happens when ALL 28 waves succeed

## 📚 Tech Stack

- **Framework:** Next.js 14+ (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS 4
- **Fonts:** Geist Sans & Geist Mono
- **GitHub API:** @octokit/rest
- **CSV Parsing:** papaparse
- **Linting:** ESLint
- **Formatting:** Prettier

## 🧪 Testing

### Manual Testing Checklist

1. **GET /api/snapshot**
   - [ ] Returns JSON with 28 waves
   - [ ] Returns CSV when `?format=csv` is used
   - [ ] Returns error if snapshot is invalid

2. **POST /api/rebuild-snapshot**
   - [ ] Fetches market data for all tickers
   - [ ] Computes returns for all periods
   - [ ] Validates exactly 28 waves
   - [ ] Commits to GitHub on success
   - [ ] Returns error report on failure
   - [ ] Leaves GitHub file unchanged on failure

3. **UI (Snapshot Console)**
   - [ ] Displays 28 waves in table
   - [ ] Shows return percentages with color coding
   - [ ] "Rebuild Snapshot Now" button works
   - [ ] Status messages display correctly
   - [ ] Timestamp updates after rebuild

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

The site requires the following environment variable:

- **NEXT_PUBLIC_SITE_URL**: Canonical site URL (default: `https://www.wavesintelligence.app`)
  - Used for metadata, Open Graph tags, sitemap, and robots.txt
  - Set in Vercel for production deployments
  - For local development, create `.env.local` with `NEXT_PUBLIC_SITE_URL=http://localhost:3000`

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
