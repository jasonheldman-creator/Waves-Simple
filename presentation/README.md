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

### Narration Files
Located in `/narration/` directory:
- 13 text files (one per slide)
- UTF-8 encoded
- TTS-optimized formatting
- Calm, warm, explanatory tone

### TTS Services
Compatible with:
- Amazon Polly (Matthew/Joanna voices)
- Google Cloud Text-to-Speech (Wavenet-D/F)
- Microsoft Azure Speech (Guy/Jenny Neural)
- ElevenLabs (Professional settings)

### Recommended Settings
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
pip install python-pptx>=0.6.21
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
└── narration/                                   # TTS-ready files
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
