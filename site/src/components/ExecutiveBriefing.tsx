"use client";

import { useState } from "react";

interface ExecutiveBriefingProps {
  variant?: "primary" | "secondary";
  showDownload?: boolean;
}

export default function ExecutiveBriefing({
  variant = "primary",
  showDownload = true,
}: ExecutiveBriefingProps) {
  const [isVideoOpen, setIsVideoOpen] = useState(false);

  const handleWatchClick = () => {
    setIsVideoOpen(true);
  };

  const handleCloseVideo = () => {
    setIsVideoOpen(false);
  };

  if (variant === "secondary") {
    // Smaller reference section for Platform page
    return (
      <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6">
        <h3 className="text-xl font-semibold text-white mb-3">Executive Briefing</h3>
        <p className="text-sm text-gray-300 mb-4">
          A narrated system-level walkthrough of the WAVES Intelligence™ Framework.
        </p>
        <button
          onClick={handleWatchClick}
          className="inline-flex items-center rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-6 py-3 text-sm font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
        >
          <span className="mr-2">▶</span>
          Watch Executive Briefing
        </button>
        
        {/* Video Modal */}
        {isVideoOpen && (
          <VideoModal onClose={handleCloseVideo} showDownload={showDownload} />
        )}
      </div>
    );
  }

  // Primary placement for homepage
  return (
    <section className="bg-gradient-to-b from-black via-gray-900 to-black py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            Executive Briefing
          </h2>
          <p className="mt-4 text-lg text-gray-400 max-w-3xl mx-auto">
            A narrated overview of the WAVES Intelligence™ decision framework (approximately 8 minutes).
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-8 text-center">
            <button
              onClick={handleWatchClick}
              className="inline-block rounded-md bg-gradient-to-r from-cyan-500 to-green-500 px-8 py-4 text-base font-semibold text-black transition-all hover:from-cyan-400 hover:to-green-400 hover:shadow-lg hover:shadow-cyan-500/50"
            >
              <span className="mr-2">▶</span>
              Watch Executive Briefing
            </button>
            
            {showDownload && (
              <div className="mt-6">
                <a
                  href="/briefing/WAVES_Intelligence_Executive_Briefing.pptx"
                  className="text-sm text-gray-400 hover:text-cyan-400 transition-colors"
                  download
                >
                  Download Slides (PPTX)
                </a>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Video Modal */}
      {isVideoOpen && (
        <VideoModal onClose={handleCloseVideo} showDownload={showDownload} />
      )}
    </section>
  );
}

interface VideoModalProps {
  onClose: () => void;
  showDownload?: boolean;
}

function VideoModal({ onClose, showDownload = true }: VideoModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-6xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute -top-12 right-0 text-white hover:text-cyan-400 text-2xl font-bold transition-colors"
          aria-label="Close video"
        >
          ✕
        </button>

        {/* Video container */}
        <div className="relative w-full bg-black rounded-lg overflow-hidden" style={{ paddingBottom: "56.25%" }}>
          <video
            className="absolute top-0 left-0 w-full h-full"
            controls
            autoPlay
            preload="metadata"
            controlsList="nodownload"
          >
            <source src="/briefing/executive-briefing.mp4" type="video/mp4" />
            <p className="text-white p-8">
              Your browser does not support the video tag. Please download the presentation or view it in a modern browser.
            </p>
          </video>
        </div>

        {/* Download link below video */}
        {showDownload && (
          <div className="mt-4 text-center">
            <a
              href="/briefing/WAVES_Intelligence_Executive_Briefing.pptx"
              className="text-sm text-gray-400 hover:text-cyan-400 transition-colors"
              download
            >
              Download Slides (PPTX)
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
