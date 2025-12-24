#!/usr/bin/env python3
"""
WAVES Intelligence™ Executive Briefing - ElevenLabs Audio Generation Script

This script generates voiceover audio files for the 13-slide executive briefing
using ElevenLabs API with Vector voice settings.

Features:
- Reads narration text files from /presentation/narration/
- Generates high-quality audio files using ElevenLabs API
- Creates placeholder silent audio if API key is missing
- Computes duration metrics and validates against target (7-9 minutes)
- Generates NARRATION_MANIFEST.json for PowerPoint integration

Usage:
    # Set ElevenLabs API key as environment variable
    export ELEVENLABS_API_KEY="your_api_key_here"
    
    # Run the script
    python generate_audio.py
    
    # Or with custom output directory
    python generate_audio.py --output-dir /path/to/output

Requirements:
    pip install elevenlabs pydub
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import ElevenLabs SDK
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️  Warning: elevenlabs package not installed. Install with: pip install elevenlabs")

# Try to import pydub for audio manipulation
try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️  Warning: pydub package not installed. Install with: pip install pydub")


class AudioGenerator:
    """Generates voiceover audio files for WAVES Executive Briefing."""
    
    def __init__(self, config_path: str = None, output_dir: str = None):
        """Initialize the audio generator.
        
        Args:
            config_path: Path to elevenlabs_config.json
            output_dir: Directory to save generated audio files
        """
        # Set paths
        self.script_dir = Path(__file__).parent
        self.presentation_dir = self.script_dir.parent
        self.narration_dir = self.presentation_dir / "narration"
        
        # Configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.script_dir / "elevenlabs_config.json"
        
        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.script_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize ElevenLabs client
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        
        if self.api_key and ELEVENLABS_AVAILABLE:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
                print("✓ ElevenLabs API client initialized successfully")
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize ElevenLabs client: {e}")
                self.client = None
        else:
            if not self.api_key:
                print("⚠️  Warning: ELEVENLABS_API_KEY environment variable not set")
            print("→ Will generate silent placeholder audio files instead")
        
        # Slide metadata
        self.slides = self._get_slide_metadata()
        
        # Results
        self.results = []
    
    def _load_config(self) -> Dict:
        """Load ElevenLabs configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config from {self.config_path}: {e}")
            # Return default configuration
            return {
                "voice_id": "pNInz6obpgDQGcFmaJgB",
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.65,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "output_format": "mp3_44100_128"
            }
    
    def _get_slide_metadata(self) -> List[Dict]:
        """Get metadata for all 13 slides."""
        slides = []
        for i in range(1, 14):
            slide_num = f"{i:02d}"
            
            # Find narration file
            narration_files = list(self.narration_dir.glob(f"slide_{slide_num}_*.txt"))
            
            if narration_files:
                narration_file = narration_files[0]
            else:
                print(f"⚠️  Warning: Narration file not found for slide {slide_num}")
                narration_file = None
            
            slides.append({
                "slide_number": i,
                "slide_id": slide_num,
                "narration_file": narration_file,
                "audio_file": self.output_dir / f"slide_{slide_num}.mp3"
            })
        
        return slides
    
    def _estimate_text_duration(self, text: str) -> float:
        """Estimate audio duration based on text length.
        
        Uses average speaking rate of 150 words per minute for professional narration.
        
        Args:
            text: The narration text
            
        Returns:
            Estimated duration in seconds
        """
        words = len(text.split())
        # Professional narration: ~150 words per minute = 2.5 words per second
        duration_seconds = words / 2.5
        return duration_seconds
    
    def _create_silent_audio(self, duration_seconds: float, output_path: Path) -> bool:
        """Create a silent audio file as placeholder.
        
        Args:
            duration_seconds: Duration of silent audio
            output_path: Path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not PYDUB_AVAILABLE:
            print(f"⚠️  Cannot create silent audio: pydub not available")
            # Create a marker file instead
            marker_path = output_path.with_suffix('.txt')
            try:
                with open(marker_path, 'w') as f:
                    f.write(f"Placeholder for {output_path.name}\n")
                    f.write(f"Duration: {duration_seconds:.1f} seconds\n")
                print(f"   → Created marker file: {marker_path.name}")
                return False
            except Exception as e:
                print(f"⚠️  Error creating marker file: {e}")
                return False
        
        try:
            # Create silent audio
            silent = AudioSegment.silent(duration=int(duration_seconds * 1000))
            
            # Export as MP3
            silent.export(output_path, format="mp3", bitrate="128k")
            
            return True
        except FileNotFoundError as e:
            if 'ffmpeg' in str(e).lower() or 'avconv' in str(e).lower():
                print(f"⚠️  ffmpeg/avconv not found - required for audio creation")
                print(f"   → Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
                # Create a marker file instead
                marker_path = output_path.with_suffix('.txt')
                try:
                    with open(marker_path, 'w') as f:
                        f.write(f"Placeholder for {output_path.name}\n")
                        f.write(f"Duration: {duration_seconds:.1f} seconds\n")
                        f.write(f"NOTE: Install ffmpeg to generate actual silent audio files\n")
                    print(f"   → Created marker file instead: {marker_path.name}")
                except Exception as marker_error:
                    print(f"⚠️  Error creating marker file: {marker_error}")
                return False
            else:
                raise
        except Exception as e:
            print(f"⚠️  Error creating silent audio: {e}")
            return False
    
    def _generate_audio_elevenlabs(self, text: str, output_path: Path) -> Tuple[bool, float]:
        """Generate audio using ElevenLabs API.
        
        Args:
            text: The narration text
            output_path: Path to save the audio file
            
        Returns:
            Tuple of (success: bool, duration: float)
        """
        if not self.client:
            return False, 0.0
        
        try:
            # Get voice settings from config
            voice_settings = VoiceSettings(
                stability=self.config["voice_settings"]["stability"],
                similarity_boost=self.config["voice_settings"]["similarity_boost"],
                style=self.config["voice_settings"].get("style", 0.0),
                use_speaker_boost=self.config["voice_settings"].get("use_speaker_boost", True)
            )
            
            # Generate audio
            print(f"   → Generating with ElevenLabs API...", end=" ")
            
            audio_generator = self.client.generate(
                text=text,
                voice=self.config["voice_id"],
                model=self.config["model_id"],
                voice_settings=voice_settings
            )
            
            # Save audio
            with open(output_path, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)
            
            print("✓")
            
            # Get duration using pydub
            if PYDUB_AVAILABLE:
                audio = AudioSegment.from_mp3(output_path)
                duration = len(audio) / 1000.0  # Convert to seconds
            else:
                # Estimate duration
                duration = self._estimate_text_duration(text)
            
            return True, duration
            
        except Exception as e:
            print(f"✗")
            print(f"⚠️  Error generating audio with ElevenLabs: {e}")
            return False, 0.0
    
    def generate_slide_audio(self, slide: Dict) -> Dict:
        """Generate audio for a single slide.
        
        Args:
            slide: Slide metadata dictionary
            
        Returns:
            Result dictionary with generation details
        """
        slide_id = slide["slide_id"]
        print(f"\n📍 Slide {slide_id}:")
        
        result = {
            "slide_number": slide["slide_number"],
            "slide_id": slide_id,
            "narration_file": str(slide["narration_file"]) if slide["narration_file"] else None,
            "audio_file": str(slide["audio_file"]),
            "success": False,
            "duration": 0.0,
            "estimated_duration": 0.0,
            "method": None,
            "error": None
        }
        
        # Read narration text
        if not slide["narration_file"] or not slide["narration_file"].exists():
            result["error"] = "Narration file not found"
            print(f"   ✗ Narration file not found")
            return result
        
        try:
            with open(slide["narration_file"], 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                result["error"] = "Narration file is empty"
                print(f"   ✗ Narration file is empty")
                return result
            
            # Show text preview
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"   → Text: {preview}")
            
            # Estimate duration
            estimated_duration = self._estimate_text_duration(text)
            result["estimated_duration"] = estimated_duration
            print(f"   → Estimated duration: {estimated_duration:.1f}s ({estimated_duration/60:.1f}min)")
            
            # Try ElevenLabs first
            if self.client:
                success, duration = self._generate_audio_elevenlabs(text, slide["audio_file"])
                
                if success:
                    result["success"] = True
                    result["duration"] = duration
                    result["method"] = "elevenlabs"
                    print(f"   ✓ Audio generated: {slide['audio_file'].name} ({duration:.1f}s)")
                    return result
            
            # Fallback to silent audio (or marker file)
            print(f"   → Creating silent placeholder...")
            success = self._create_silent_audio(estimated_duration, slide["audio_file"])
            
            if success:
                result["success"] = True
                result["duration"] = estimated_duration
                result["method"] = "silent_placeholder"
                print(f"   ✓ Silent placeholder created: {slide['audio_file'].name} ({estimated_duration:.1f}s)")
            else:
                # Even if we couldn't create the file, use estimated duration for planning
                result["duration"] = estimated_duration
                result["error"] = "Failed to create placeholder audio (marker file created)"
                print(f"   → Using estimated duration for planning: {estimated_duration:.1f}s")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"   ✗ Error: {e}")
        
        return result
    
    def generate_all_audio(self) -> List[Dict]:
        """Generate audio for all 13 slides.
        
        Returns:
            List of result dictionaries
        """
        print("\n" + "=" * 80)
        print("WAVES Intelligence™ Executive Briefing - Audio Generation")
        print("=" * 80)
        
        print(f"\nConfiguration:")
        print(f"  • Voice: {self.config.get('voice_name', 'Unknown')}")
        print(f"  • Model: {self.config.get('model_id', 'Unknown')}")
        print(f"  • Output directory: {self.output_dir}")
        print(f"  • Narration directory: {self.narration_dir}")
        
        if self.client:
            print(f"  • Method: ElevenLabs API")
        else:
            print(f"  • Method: Silent placeholders (no API key)")
        
        # Generate audio for each slide
        results = []
        for slide in self.slides:
            result = self.generate_slide_audio(slide)
            results.append(result)
            
            # Rate limiting - be respectful to API
            if self.client and result["success"] and result["method"] == "elevenlabs":
                time.sleep(0.5)  # Small delay between API calls
        
        self.results = results
        return results
    
    def analyze_durations(self) -> Dict:
        """Analyze durations and flag issues.
        
        Returns:
            Analysis dictionary with statistics and warnings
        """
        print("\n" + "=" * 80)
        print("Duration Analysis")
        print("=" * 80)
        
        # Calculate statistics - use all results with durations (estimated or actual)
        results_with_duration = [r for r in self.results if r["duration"] > 0]
        durations = [r["duration"] for r in results_with_duration]
        
        if not durations:
            print("⚠️  No durations available (no text files processed)")
            return {
                "total_slides": len(self.results),
                "successful_slides": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "target_min": 420,
                "target_max": 540,
                "within_target": False,
                "warnings": ["No durations available"]
            }
        
        successful_audio = [r for r in self.results if r["success"]]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Target: 7-9 minutes (420-540 seconds)
        target_min = 420
        target_max = 540
        
        print(f"\nStatistics:")
        print(f"  • Total slides: {len(self.results)}")
        print(f"  • Audio files generated: {len(successful_audio)}")
        print(f"  • Durations available: {len(results_with_duration)}")
        print(f"  • Total duration: {total_duration:.1f}s ({total_duration/60:.2f} minutes)")
        print(f"  • Average per slide: {avg_duration:.1f}s")
        print(f"  • Min slide duration: {min_duration:.1f}s")
        print(f"  • Max slide duration: {max_duration:.1f}s")
        
        if len(successful_audio) < len(results_with_duration):
            print(f"  • Note: Some durations are estimates (audio files not created)")
        
        # Check if within target range
        print(f"\nTarget Range: {target_min}s - {target_max}s ({target_min/60:.1f} - {target_max/60:.1f} minutes)")
        
        warnings = []
        if total_duration < target_min:
            diff = target_min - total_duration
            print(f"⚠️  WARNING: Total duration is {diff:.1f}s ({diff/60:.1f} min) UNDER target minimum")
            warnings.append(f"Duration {diff:.1f}s under target minimum")
        elif total_duration > target_max:
            diff = total_duration - target_max
            print(f"⚠️  WARNING: Total duration is {diff:.1f}s ({diff/60:.1f} min) OVER target maximum")
            warnings.append(f"Duration {diff:.1f}s over target maximum")
        else:
            print(f"✓ Total duration is within target range")
        
        # Flag slides with unusual durations
        print(f"\nPer-Slide Analysis:")
        
        # Expected duration per slide: 7-9 minutes / 13 slides = ~32-42 seconds per slide
        expected_min = 20  # Allow slides as short as 20s (title, closing)
        expected_max = 60  # Flag slides longer than 60s
        
        for result in results_with_duration:
            duration = result["duration"]
            slide_id = result["slide_id"]
            
            status = "✓"
            note = ""
            
            if not result["success"]:
                status = "ℹ️ "
                note = " (estimated)"
            
            if duration < expected_min:
                status = "⚠️ "
                note += " (unusually short - may need more content)"
                warnings.append(f"Slide {slide_id}: {duration:.1f}s is very short")
            elif duration > expected_max:
                status = "⚠️ "
                note += " (unusually long - may need editing)"
                warnings.append(f"Slide {slide_id}: {duration:.1f}s is very long")
            
            print(f"  {status} Slide {slide_id}: {duration:5.1f}s{note}")
        
        return {
            "total_slides": len(self.results),
            "successful_slides": len(successful_audio),
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "target_min": target_min,
            "target_max": target_max,
            "within_target": target_min <= total_duration <= target_max,
            "warnings": warnings
        }
    
    def generate_manifest(self) -> str:
        """Generate NARRATION_MANIFEST.json file.
        
        Returns:
            Path to the generated manifest file
        """
        print("\n" + "=" * 80)
        print("Generating Manifest")
        print("=" * 80)
        
        # Calculate cumulative start times
        cumulative_time = 0.0
        manifest_data = {
            "metadata": {
                "title": "WAVES Intelligence™ Executive Briefing",
                "version": "1.0.0",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_slides": len(self.results),
                "voice": self.config.get("voice_name", "ElevenLabs Vector"),
                "voice_id": self.config.get("voice_id", ""),
                "model": self.config.get("model_id", "")
            },
            "slides": []
        }
        
        for result in self.results:
            slide_entry = {
                "slide_number": result["slide_number"],
                "slide_id": result["slide_id"],
                "narration_file": Path(result["narration_file"]).name if result["narration_file"] else None,
                "audio_file": Path(result["audio_file"]).name,
                "duration_seconds": round(result["duration"], 2),
                "duration_formatted": self._format_duration(result["duration"]),
                "start_time_seconds": round(cumulative_time, 2),
                "start_time_formatted": self._format_duration(cumulative_time),
                "success": result["success"],
                "method": result["method"],
                "error": result["error"]
            }
            
            manifest_data["slides"].append(slide_entry)
            cumulative_time += result["duration"]
        
        # Add summary
        analysis = self.analyze_durations()
        manifest_data["summary"] = {
            "total_duration_seconds": round(analysis["total_duration"], 2),
            "total_duration_formatted": self._format_duration(analysis["total_duration"]),
            "successful_slides": analysis["successful_slides"],
            "within_target_range": analysis["within_target"],
            "warnings": analysis["warnings"]
        }
        
        # Save manifest
        manifest_path = self.output_dir / "NARRATION_MANIFEST.json"
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2)
        
        print(f"\n✓ Manifest generated: {manifest_path}")
        
        return str(manifest_path)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration as MM:SS.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted string
        """
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"
    
    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        
        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        
        print(f"\n✓ Generation complete!")
        print(f"  • Total slides: {len(self.results)}")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {failed}")
        
        if self.client:
            print(f"  • Method: ElevenLabs API")
        else:
            print(f"  • Method: Silent placeholders")
        
        print(f"\nOutput files:")
        print(f"  • Audio files: {self.output_dir}")
        print(f"  • Manifest: {self.output_dir / 'NARRATION_MANIFEST.json'}")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate voiceover audio for WAVES Executive Briefing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for audio files (default: voiceover/)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to elevenlabs_config.json (default: voiceover/elevenlabs_config.json)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = AudioGenerator(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Generate all audio
    generator.generate_all_audio()
    
    # Analyze durations
    generator.analyze_durations()
    
    # Generate manifest
    generator.generate_manifest()
    
    # Print summary
    generator.print_summary()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
