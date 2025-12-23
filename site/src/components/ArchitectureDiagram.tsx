export default function ArchitectureDiagram() {
  return (
    <section className="bg-gradient-to-b from-gray-900 to-black py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            System <span className="text-cyan-400">Architecture</span>
          </h2>
          <p className="mt-4 text-lg text-gray-400">
            Enterprise-grade infrastructure designed for institutional performance
          </p>
        </div>

        <div className="mx-auto max-w-4xl">
          <svg
            viewBox="0 0 800 600"
            className="w-full h-auto"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Background */}
            <rect width="800" height="600" fill="#0a0a0a" />

            {/* Data Layer */}
            <g>
              <rect
                x="50"
                y="50"
                width="700"
                height="100"
                fill="#1a1a1a"
                stroke="#00ffff"
                strokeWidth="2"
                rx="8"
              />
              <text x="400" y="90" textAnchor="middle" fill="#00ffff" fontSize="18" fontWeight="bold">
                Data Layer
              </text>
              <text x="400" y="115" textAnchor="middle" fill="#ededed" fontSize="14">
                Market Data • Wave History • Portfolio Holdings
              </text>
            </g>

            {/* Processing Layer */}
            <g>
              <rect
                x="50"
                y="200"
                width="220"
                height="120"
                fill="#1a1a1a"
                stroke="#00ff88"
                strokeWidth="2"
                rx="8"
              />
              <text x="160" y="240" textAnchor="middle" fill="#00ff88" fontSize="16" fontWeight="bold">
                Analytics Engine
              </text>
              <text x="160" y="265" textAnchor="middle" fill="#ededed" fontSize="12">
                Performance Metrics
              </text>
              <text x="160" y="285" textAnchor="middle" fill="#ededed" fontSize="12">
                Alpha Attribution
              </text>
              <text x="160" y="305" textAnchor="middle" fill="#ededed" fontSize="12">
                Risk Analysis
              </text>
            </g>

            <g>
              <rect
                x="290"
                y="200"
                width="220"
                height="120"
                fill="#1a1a1a"
                stroke="#00ff88"
                strokeWidth="2"
                rx="8"
              />
              <text x="400" y="240" textAnchor="middle" fill="#00ff88" fontSize="16" fontWeight="bold">
                Decision Engine
              </text>
              <text x="400" y="265" textAnchor="middle" fill="#ededed" fontSize="12">
                Portfolio Optimization
              </text>
              <text x="400" y="285" textAnchor="middle" fill="#ededed" fontSize="12">
                Regime Detection
              </text>
              <text x="400" y="305" textAnchor="middle" fill="#ededed" fontSize="12">
                Signal Generation
              </text>
            </g>

            <g>
              <rect
                x="530"
                y="200"
                width="220"
                height="120"
                fill="#1a1a1a"
                stroke="#00ff88"
                strokeWidth="2"
                rx="8"
              />
              <text x="640" y="240" textAnchor="middle" fill="#00ff88" fontSize="16" fontWeight="bold">
                Reporting Layer
              </text>
              <text x="640" y="265" textAnchor="middle" fill="#ededed" fontSize="12">
                Board Packs
              </text>
              <text x="640" y="285" textAnchor="middle" fill="#ededed" fontSize="12">
                Custom Reports
              </text>
              <text x="640" y="305" textAnchor="middle" fill="#ededed" fontSize="12">
                Real-time Alerts
              </text>
            </g>

            {/* Presentation Layer */}
            <g>
              <rect
                x="50"
                y="370"
                width="700"
                height="100"
                fill="#1a1a1a"
                stroke="#00ffff"
                strokeWidth="2"
                rx="8"
              />
              <text x="400" y="410" textAnchor="middle" fill="#00ffff" fontSize="18" fontWeight="bold">
                Presentation Layer
              </text>
              <text x="400" y="435" textAnchor="middle" fill="#ededed" fontSize="14">
                Web Console • API Gateway • Mobile Interface
              </text>
            </g>

            {/* Security Layer */}
            <g>
              <rect
                x="50"
                y="520"
                width="700"
                height="60"
                fill="#1a1a1a"
                stroke="#ff6b6b"
                strokeWidth="2"
                rx="8"
              />
              <text x="400" y="555" textAnchor="middle" fill="#ff6b6b" fontSize="16" fontWeight="bold">
                Security & Compliance Layer
              </text>
            </g>

            {/* Connection lines */}
            <line x1="400" y1="150" x2="400" y2="200" stroke="#00ffff" strokeWidth="2" />
            <line x1="160" y1="320" x2="160" y2="370" stroke="#00ff88" strokeWidth="2" />
            <line x1="400" y1="320" x2="400" y2="370" stroke="#00ff88" strokeWidth="2" />
            <line x1="640" y1="320" x2="640" y2="370" stroke="#00ff88" strokeWidth="2" />
            <line x1="400" y1="470" x2="400" y2="520" stroke="#00ffff" strokeWidth="2" />
          </svg>
        </div>
      </div>
    </section>
  );
}
