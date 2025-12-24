"use client";

import { useState, useRef, useEffect } from "react";

interface ExecutiveBriefingProps {
  compact?: boolean;
}

export default function ExecutiveBriefing({ compact = false }: ExecutiveBriefingProps) {
  const [videoError, setVideoError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);

    return () => {
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
    };
  }, []);

  const handleVideoError = () => {
    setVideoError(true);
  };

  const videoSrc = "/briefing/executive-briefing.mp4";

  if (compact) {
    return (
      <div className="rounded-lg border border-cyan-500/20 bg-gradient-to-br from-gray-900/80 to-gray-800/50 p-6">
        <div className="flex items-start">
          <div className="text-3xl mr-4">🎥</div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white mb-2">Executive Briefing</h3>
            <p className="text-sm text-gray-400 mb-4">
              Watch a comprehensive overview of the WAVES Intelligence™ decision framework
              (approximately 8 minutes).
            </p>
            <a
              href="/briefing"
              className="inline-flex items-center text-sm font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              View Briefing →
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="relative rounded-lg overflow-hidden border border-cyan-500/20 bg-gray-900">
        {videoError ? (
          <div className="aspect-video flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
            <div className="text-center p-8">
              <div className="text-6xl mb-4">🎥</div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Executive Briefing Video
              </h3>
              <p className="text-gray-400 max-w-md mx-auto">
                Video content will be available here. This is an informational placeholder
                ensuring the player interface is ready for deployment.
              </p>
            </div>
          </div>
        ) : (
          <video
            ref={videoRef}
            className="w-full aspect-video bg-black"
            controls
            preload="metadata"
            onError={handleVideoError}
            controlsList="nodownload"
            playsInline
          >
            <source src={videoSrc} type="video/mp4" />
            <p className="text-gray-400 p-8">
              Your browser does not support the video tag. Please use a modern browser to
              view this content.
            </p>
          </video>
        )}
        
        {/* Video overlay for additional context when not playing */}
        {!isPlaying && !videoError && (
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent pointer-events-none flex items-end">
            <div className="p-6 w-full">
              <p className="text-xs text-gray-300">
                WAVES Intelligence™ Executive Briefing • Approximately 8 minutes
              </p>
            </div>
          </div>
        )}
      </div>
      
      {/* Governance note */}
      <div className="mt-4 rounded-md bg-gray-900/50 border border-gray-800 p-4">
        <p className="text-xs text-gray-400 leading-relaxed">
          <span className="font-semibold text-gray-300">Governance Note:</span> This
          briefing is an informational system overview provided for educational purposes. It
          does not constitute investment advice, and viewers should conduct their own due
          diligence.
        </p>
      </div>
    </div>
  );
}
