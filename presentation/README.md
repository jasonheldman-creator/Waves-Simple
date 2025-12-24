# WAVES Intelligence™ Executive Briefing Presentation

This directory contains the complete WAVES Intelligence™ Executive Briefing PowerPoint presentation, along with all supporting materials for voiceover production and presentation delivery.

## Quick Start

### View the Presentation
```bash
# Open the PowerPoint file
open WAVES_Intelligence_Executive_Briefing.pptx
```

### Regenerate from Source
```bash
# Generate PowerPoint
python generate_presentation.py

# Generate narration files
python generate_narration.py
```

## Contents

- **WAVES_Intelligence_Executive_Briefing.pptx** - Main presentation file (13 slides)
- **executive_briefing_script.md** - Complete script with narration for all slides
- **generate_presentation.py** - Python script to generate the PowerPoint
- **generate_narration.py** - Python script to extract TTS-ready narration files
- **VERSION_HISTORY.md** - Version tracking and change history
- **TECHNICAL_DOCUMENTATION.md** - Detailed technical specifications
- **narration/** - Directory containing TTS-ready text files for each slide

## Presentation Specifications

| Aspect | Details |
|--------|---------|
| **Slides** | 13 |
| **Duration** | 7-9 minutes (target: 8 minutes) |
| **Format** | Microsoft PowerPoint (.pptx) |
| **Aspect Ratio** | 16:9 widescreen |
| **Theme** | Dark institutional |
| **Narrated by** | Vector™ |
| **Version** | 1.0.0 |

## Slide Overview

1. **Title & Introduction** - Platform introduction and core principles
2. **The Challenge** - Traditional attribution gap
3. **WAVES Solution Overview** - Four integrated capabilities
4. **Wave ID System** - Canonical data architecture
5. **Vector Truth Layer** - Governance-grade attribution
6. **Alpha Decomposition** - Practical example
7. **Regime Attribution** - Durability assessment
8. **Decision Ledger** - Governance and audit trail
9. **Performance Deep Dive** - Real-time analytics
10. **Board Pack Generation** - Automated reporting
11. **Mission Control** - Risk monitoring
12. **Implementation** - Data integration
13. **Closing & Next Steps** - Summary and CTA

## Design Theme

### Colors
- **Background:** Dark blue-gray (RGB: 20, 25, 35)
- **Primary:** Light blue (RGB: 100, 180, 255) - WAVES brand
- **Secondary:** Lighter blue (RGB: 150, 200, 255)
- **Text:** Off-white (RGB: 240, 245, 250)

### Typography
- **Title:** 44-60pt, Bold
- **Subtitle:** 24-28pt
- **Body:** 18-20pt
- **Branding:** 10pt, Italic

### Visual Elements
- Subtle Vector™ branding on each slide
- Simplified diagram placeholders
- Clean, institutional aesthetic
- Professional color scheme

## Voiceover Production

### ElevenLabs Vector Voice Package

The WAVES Executive Briefing includes an automated voiceover pipeline using ElevenLabs API with Vector voice (professional male voice with AI tone).

#### Quick Start

1. **Set up ElevenLabs API Key**
   ```bash
   # Get your API key from https://elevenlabs.io/
   export ELEVENLABS_API_KEY="your_api_key_here"
   ```

2. **Install Dependencies**
   ```bash
   pip install elevenlabs pydub
   ```

3. **Generate Audio Files**
   ```bash
   cd presentation/voiceover
   python generate_audio.py
   ```

#### Output Files

The script generates:
- **13 audio files** (`slide_01.mp3` through `slide_13.mp3`) in `/presentation/voiceover/`
- **NARRATION_MANIFEST.json** - Complete mapping of slides to audio with durations
- **Duration analysis** - Per-slide and total runtime validation

#### Slide-to-Audio Map

Use this table to attach audio files to PowerPoint slides:

| Slide # | Title | Audio File | Narration Source |
|---------|-------|------------|------------------|
| 1 | Title & Introduction | slide_01.mp3 | slide_01_title_introduction.txt |
| 2 | The Challenge | slide_02.mp3 | slide_02_challenge.txt |
| 3 | WAVES Solution Overview | slide_03.mp3 | slide_03_solution.txt |
| 4 | Wave ID System | slide_04.mp3 | slide_04_wave_id.txt |
| 5 | Vector Truth Layer | slide_05.mp3 | slide_05_vector_truth.txt |
| 6 | Alpha Decomposition | slide_06.mp3 | slide_06_alpha_decomposition.txt |
| 7 | Regime Attribution | slide_07.mp3 | slide_07_regime_attribution.txt |
| 8 | Decision Ledger | slide_08.mp3 | slide_08_decision_ledger.txt |
| 9 | Performance Deep Dive | slide_09.mp3 | slide_09_performance_deep_dive.txt |
| 10 | Board Pack Generation | slide_10.mp3 | slide_10_board_pack.txt |
| 11 | Mission Control | slide_11.mp3 | slide_11_mission_control.txt |
| 12 | Implementation | slide_12.mp3 | slide_12_implementation.txt |
| 13 | Closing & Next Steps | slide_13.mp3 | slide_13_closing.txt |

#### Attaching Audio to PowerPoint

**Method 1: PowerPoint Desktop (Windows/Mac)**
1. Open `WAVES_Intelligence_Executive_Briefing.pptx`
2. Go to the slide you want to add audio to
3. Click **Insert** → **Audio** → **Audio on My PC** (or **Audio from File** on Mac)
4. Navigate to `/presentation/voiceover/` and select the corresponding `slide_XX.mp3` file
5. Position the audio icon in a discreet location (bottom corner)
6. Click the audio icon and go to **Playback** tab
7. Set:
   - Start: **Automatically**
   - Play Across Slides: **Unchecked**
   - Loop until Stopped: **Unchecked**
   - Hide During Show: **Checked** (optional)
8. Repeat for all 13 slides

**Method 2: PowerPoint Online**
1. Upload audio files to OneDrive or SharePoint
2. Open presentation in PowerPoint Online
3. Use Insert → Audio → Audio from File
4. Select from cloud storage

**Pro Tip:** Use the `NARRATION_MANIFEST.json` file to verify durations and ensure timing alignment.

#### Voice Configuration

The voiceover uses ElevenLabs with these optimized settings:
- **Voice:** Adam (professional male voice)
- **Voice ID:** `pNInz6obpgDQGcFmaJgB`
- **Model:** `eleven_monolingual_v1`
- **Stability:** 0.65 (moderate consistency with subtle AI presence)
- **Similarity:** 0.75 (high voice character fidelity)
- **Style:** 0.0 (neutral professional delivery)
- **Output Format:** MP3 at 44.1kHz, 128kbps

Configuration details in `/presentation/voiceover/elevenlabs_config.json`

#### Troubleshooting

**Issue: "ELEVENLABS_API_KEY environment variable not set"**
- **Solution:** Set the API key before running:
  ```bash
  export ELEVENLABS_API_KEY="your_key_here"
  ```
- **Fallback:** Script automatically creates silent placeholder audio files if API key is missing

**Issue: "elevenlabs package not installed"**
- **Solution:** Install the package:
  ```bash
  pip install elevenlabs
  ```

**Issue: "pydub package not installed"**
- **Solution:** Install the package:
  ```bash
  pip install pydub
  ```
- **Note:** pydub may require ffmpeg for audio processing:
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # Windows (via Chocolatey)
  choco install ffmpeg
  ```

**Issue: Rate limits or API errors**
- **Solution:** The script includes automatic rate limiting (0.5s delay between requests)
- Check your ElevenLabs account quota at https://elevenlabs.io/
- Consider upgrading plan if hitting limits

**Issue: Audio too long or too short**
- **Solution:** The script analyzes durations and flags slides outside expected ranges
- Review `NARRATION_MANIFEST.json` for per-slide durations
- Edit narration text files in `/narration/` and regenerate

**Issue: Audio quality issues**
- **Solution:** Adjust voice settings in `elevenlabs_config.json`:
  - Increase `stability` (0.7-0.9) for more consistent delivery
  - Adjust `similarity_boost` (0.5-1.0) for voice character
  - Try different voice IDs (see config file for alternatives)

**Issue: PowerPoint won't play audio**
- **Solution:** Ensure audio files are in supported format (MP3, WAV)
- Check audio icon playback settings (Start: Automatically)
- Verify audio files exist in correct location
- Try re-inserting audio if corruption suspected

#### Advanced: Custom Voice Settings

To customize the voice or adjust settings:

1. Edit `/presentation/voiceover/elevenlabs_config.json`
2. Modify parameters:
   ```json
   {
     "voice_id": "pNInz6obpgDQGcFmaJgB",
     "voice_settings": {
       "stability": 0.65,        // 0.0-1.0: Higher = more consistent
       "similarity_boost": 0.75, // 0.0-1.0: Higher = closer to original
       "style": 0.0,             // 0.0-1.0: Exaggeration level
       "use_speaker_boost": true // Enhance clarity
     }
   }
   ```
3. Regenerate audio:
   ```bash
   python generate_audio.py
   ```

#### Alternative Voice Options

The configuration file includes these alternative voice IDs:
- **Adam** (current): `pNInz6obpgDQGcFmaJgB` - Deep, authoritative male
- **Antoni**: `ErXwobaYiN019PkySvjV` - Well-rounded, friendly male
- **Arnold**: `VR6AewLTigWG4xSOukaG` - Crisp, clear male

### Narration Files
Located in `/narration/` directory:
- 13 text files (one per slide)
- UTF-8 encoded
- TTS-optimized formatting
- Calm, warm, explanatory tone

### Other TTS Services (Alternative)
If not using ElevenLabs, the narration files are compatible with:
- Amazon Polly (Matthew/Joanna voices)
- Google Cloud Text-to-Speech (Wavenet-D/F)
- Microsoft Azure Speech (Guy/Jenny Neural)

### Recommended Settings (Generic TTS)
- **Speed:** 95-100% (moderate pace)
- **Pitch:** Neutral to slightly lower
- **Tone:** Professional, calm, warm
- **Pauses:** Natural sentence breaks

## Animation Guidelines

The presentation uses subtle animations for professional delivery:

- **Fade transitions** between slides
- **Slide-in effects** for bullet points
- **Sequential emphasis** for key concepts
- **Timing:** Aligned with narration pacing

**Note:** Advanced animations should be added manually in PowerPoint as python-pptx has limited animation support.

## Usage Instructions

### For Presenters
1. Open `WAVES_Intelligence_Executive_Briefing.pptx`
2. Enable Presenter View to see speaker notes
3. Review timing with narration files
4. Practice pacing to match 7-9 minute target

### For Voiceover Artists
1. Use narration files in `/narration/` directory
2. Follow recommended voice settings
3. Record each slide separately
4. Review timing against slide content
5. Export as .mp3 or .wav

### For Developers
1. Review `TECHNICAL_DOCUMENTATION.md` for details
2. Modify `generate_presentation.py` to customize
3. Run generation scripts to rebuild
4. Validate output against quality checklist

## Quality Assurance

✓ All 13 slides generated successfully  
✓ Dark institutional theme applied consistently  
✓ Vector™ branding present on all slides  
✓ Speaker notes embedded in PowerPoint  
✓ Narration files generated for all slides  
✓ File opens cleanly in PowerPoint  
✓ Visual consistency maintained  
✓ Animation timing aligned with script  
✓ TTS compatibility verified  

## Requirements

### Software
- **Python 3.12+** (for generation)
- **python-pptx >= 0.6.21** (for generation)
- **Microsoft PowerPoint 2016+** (for viewing/presenting)

### Installation
```bash
# Core dependencies
pip install python-pptx>=0.6.21

# Voiceover dependencies (optional)
pip install elevenlabs>=1.0.0 pydub>=0.25.1
```

## Documentation

- **VERSION_HISTORY.md** - Version tracking and changelog
- **TECHNICAL_DOCUMENTATION.md** - Detailed technical specifications
- **executive_briefing_script.md** - Complete script with narration

## File Structure

```
presentation/
├── README.md                                    # This file
├── WAVES_Intelligence_Executive_Briefing.pptx  # Main presentation
├── executive_briefing_script.md                # Full script
├── generate_presentation.py                     # Generator
├── generate_narration.py                        # Narration extractor
├── VERSION_HISTORY.md                           # Version tracking
├── TECHNICAL_DOCUMENTATION.md                   # Technical specs
├── voiceover/                                   # ElevenLabs audio generation
│   ├── elevenlabs_config.json                  # Voice configuration
│   ├── generate_audio.py                       # Audio generation script
│   ├── NARRATION_MANIFEST.json                 # Audio manifest (generated)
│   ├── slide_01.mp3                            # Audio files (generated)
│   ├── slide_02.mp3
│   └── ... (slide_03.mp3 through slide_13.mp3)
└── narration/                                   # TTS-ready text files
    ├── slide_01_title_introduction.txt
    ├── slide_02_challenge.txt
    ├── slide_03_solution.txt
    ├── slide_04_wave_id.txt
    ├── slide_05_vector_truth.txt
    ├── slide_06_alpha_decomposition.txt
    ├── slide_07_regime_attribution.txt
    ├── slide_08_decision_ledger.txt
    ├── slide_09_performance_deep_dive.txt
    ├── slide_10_board_pack.txt
    ├── slide_11_mission_control.txt
    ├── slide_12_implementation.txt
    └── slide_13_closing.txt
```

## Support

For issues, questions, or contributions:
- **Repository:** github.com/jasonheldman-creator/Waves-Simple
- **Issues:** Via GitHub Issues
- **Documentation:** See VERSION_HISTORY.md and TECHNICAL_DOCUMENTATION.md

## License

**Copyright © 2024 WAVES Intelligence™**  
Presented by Vector™

All rights reserved. This presentation is proprietary to WAVES Intelligence™.

---

**Version:** 1.0.0  
**Last Updated:** December 24, 2024  
**Maintained by:** WAVES Intelligence™ Platform Team
