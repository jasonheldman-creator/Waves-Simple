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
   - Import your GitHub repository
   - Select the `site` directory as the root directory
   - Click "Deploy"

3. **Configure Domain:**
   - In Vercel project settings, go to "Domains"
   - Add your custom domain: `wavesintelligence.com`
   - Follow Vercel's instructions to configure DNS

### Environment Variables

No environment variables are required for basic operation. If you add external services (email, analytics, etc.), configure them in Vercel's project settings.

## 🌐 DNS Configuration

### Marketing Site

Point your domain to the Next.js site:

**Domain:** `wavesintelligence.com`
**DNS Settings:**

- Type: `A` or `CNAME`
- Value: Your Vercel deployment URL

### Console Subdomain

Point the console subdomain to your existing Streamlit app:

**Domain:** `console.wavesintelligence.com`
**DNS Settings:**

- Type: `CNAME`
- Value: Your Streamlit hosting URL

### Example DNS Configuration

```
# Marketing Site (Next.js)
wavesintelligence.com          A      76.76.21.21
www.wavesintelligence.com      CNAME  wavesintelligence.com

# Console (Streamlit)
console.wavesintelligence.com  CNAME  your-streamlit-app.streamlit.app
```

**Note:** Replace the IPs/URLs with your actual deployment endpoints.

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
- Semantic HTML structure
- Mobile-responsive design
- Fast page load times

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
