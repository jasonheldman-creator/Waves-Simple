# WAVES Intelligence™ Presentation - Technical Documentation

## Overview
This document provides technical details about the WAVES Intelligence™ Executive Briefing PowerPoint presentation, including generation methodology, design specifications, animation implementation, and usage instructions.

## System Requirements

### Software Requirements
- **Python:** 3.12+ (tested on 3.12.3)
- **PowerPoint:** Microsoft PowerPoint 2016+ or compatible viewer
- **Libraries:**
  - python-pptx >= 0.6.21
  - Standard Python library (os, datetime)

### Installation
```bash
pip install python-pptx>=0.6.21
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Architecture

### File Organization
```
presentation/
├── WAVES_Intelligence_Executive_Briefing.pptx  # Generated presentation
├── executive_briefing_script.md                # Source script (13 slides)
├── generate_presentation.py                     # Main generator
├── generate_narration.py                        # Narration extractor
├── VERSION_HISTORY.md                           # Version tracking
├── TECHNICAL_DOCUMENTATION.md                   # This file
└── narration/                                   # TTS-ready files
    ├── slide_01_title_introduction.txt
    ├── slide_02_challenge.txt
    └── ... (13 files total)
```

### Code Structure

#### WavesPresentation Class
The main presentation generator class with the following responsibilities:

**Color Management:**
```python
COLORS = {
    'background': RGBColor(20, 25, 35),      # Dark blue-gray
    'primary': RGBColor(100, 180, 255),      # WAVES brand blue
    'secondary': RGBColor(150, 200, 255),    # Lighter blue
    'accent': RGBColor(255, 200, 100),       # Warm accent
    'text': RGBColor(240, 245, 250),         # Off-white
    'text_dim': RGBColor(180, 190, 200),     # Dimmed text
    'success': RGBColor(100, 220, 150),      # Success green
    'warning': RGBColor(255, 180, 100),      # Warning orange
}
```

**Slide Dimensions:**
- Width: 10 inches (16:9 widescreen)
- Height: 5.625 inches
- Resolution: Standard PowerPoint (96 DPI)

**Typography:**
- Title: 44-60pt, Bold, Primary color
- Subtitle: 24-28pt, Secondary color
- Body text: 18-20pt, Regular, Text color
- Bullet points: 20pt, Regular, Text color
- Footer/Branding: 10pt, Italic, Dimmed text

### Slide Generation Methods

Each slide has a dedicated method following the naming convention:
```python
def slide_N_description(self):
    """Generates slide N with specific content."""
```

Example methods:
- `slide_1_title()` - Title and introduction
- `slide_2_challenge()` - Problem statement
- `slide_5_vector_truth()` - Vector Truth Layer explanation
- etc.

### Design Patterns

#### Dark Institutional Theme
All slides use a consistent dark background to:
- Reduce visual fatigue during presentations
- Create professional, institutional aesthetic
- Provide high contrast for text readability
- Match WAVES Intelligence™ brand guidelines

#### Vector™ Branding
Subtle branding applied via `add_vector_branding()`:
- Position: Bottom-right corner by default
- Font: 10pt, italic
- Color: Dimmed text (RGB: 180, 190, 200)
- Text: "Vector™"

#### Diagram Placeholders
Simplified diagram placeholders created with:
```python
def add_diagram_placeholder(slide, diagram_type, left, top, width, height):
    # Creates rounded rectangle with border
    # Labels diagram type
    # Uses institutional color scheme
```

## Animation Specifications

### Animation Philosophy
Animations are intentionally subtle to maintain professional tone and avoid distraction. The presentation follows institutional presentation best practices.

### Animation Types Used

1. **Fade Transitions**
   - Between slides
   - For smooth, professional flow
   - Duration: 0.5-0.7 seconds

2. **Slide-In Effects**
   - For bullet points
   - Sequential reveal (top to bottom)
   - Duration: 0.3-0.5 seconds per item

3. **Sequential Emphasis**
   - For key concepts
   - Subtle highlighting without movement
   - Color-based emphasis where appropriate

### Animation Timing
- **Slide 1:** Title fade-in (1.5s), subtitle delay (0.5s)
- **Slides 2-12:** Content reveal aligned with narration pacing
- **Slide 13:** Sequential text reveal for closing points
- **Total Duration:** 7-9 minutes (target: 8 minutes)

**Note:** Programmatic animation support in python-pptx is limited. The current implementation provides static slides with speaker notes. Advanced animations can be added manually in PowerPoint using the Animation Pane.

## Speaker Notes Implementation

Each slide includes detailed speaker notes embedded in the PowerPoint file:

```python
def add_notes(slide, notes_text):
    """Add speaker notes to slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text
```

Notes contain:
- Full narration text for that slide
- Pacing guidance
- Key talking points
- Transition cues

## Narration Files

### Format
- **Encoding:** UTF-8
- **Line Endings:** Unix (LF)
- **Extension:** .txt (plain text)
- **Naming:** `slide_NN_description.txt`

### TTS Compatibility
Files are optimized for Text-to-Speech processing:
- Clear sentence structure
- Proper punctuation for natural pauses
- No special formatting or markup
- Consistent paragraph breaks
- Professional, conversational tone

### Recommended TTS Services

1. **Amazon Polly**
   - Voice: Matthew (male) or Joanna (female)
   - Engine: Neural
   - Speed: 95-100%

2. **Google Cloud Text-to-Speech**
   - Voice: en-US-Wavenet-D (male) or en-US-Wavenet-F (female)
   - Speaking rate: 0.95-1.0
   - Pitch: -2.0 to 0.0

3. **Microsoft Azure Speech**
   - Voice: en-US-GuyNeural (male) or en-US-JennyNeural (female)
   - Rate: 0.95-1.0
   - Pitch: Default

4. **ElevenLabs**
   - Voice: Professional settings
   - Stability: 60-75%
   - Clarity: 75-85%

## Usage Instructions

### Generating the Presentation

```bash
# Navigate to presentation directory
cd /home/runner/work/Waves-Simple/Waves-Simple/presentation

# Generate PowerPoint file
python generate_presentation.py

# Generate narration files
python generate_narration.py
```

### Customizing the Presentation

To modify slide content, edit the respective slide method in `generate_presentation.py`:

```python
def slide_N_custom(self):
    slide = self.create_blank_slide()
    self.add_title(slide, "Custom Title", "Custom Subtitle")
    self.add_body_text(slide, "Custom content here")
    self.add_vector_branding(slide)
    self.add_notes(slide, "Custom narration text")
```

### Updating Colors

Modify the `COLORS` dictionary in the `WavesPresentation` class:

```python
COLORS = {
    'background': RGBColor(R, G, B),
    'primary': RGBColor(R, G, B),
    # ... etc
}
```

### Adding New Slides

1. Create a new slide method:
   ```python
   def slide_14_new_content(self):
       slide = self.create_blank_slide()
       # ... add content
   ```

2. Add to the `generate()` method:
   ```python
   slides = [
       # ... existing slides
       ("Slide 14: New Content", self.slide_14_new_content),
   ]
   ```

3. Create corresponding narration file:
   ```python
   # In generate_narration.py
   "slide_14_new_content.txt": """Narration text here..."""
   ```

## Validation & Quality Assurance

### Pre-Delivery Checklist
- [ ] Presentation opens without errors in PowerPoint
- [ ] All 13 slides render correctly
- [ ] Dark theme applied consistently
- [ ] Vector™ branding visible on all slides
- [ ] Speaker notes embedded properly
- [ ] Narration files generated successfully
- [ ] File size reasonable (<100 KB)
- [ ] No missing fonts or layout issues

### Testing Procedures

1. **Visual Inspection:**
   ```bash
   # Open in PowerPoint
   open WAVES_Intelligence_Executive_Briefing.pptx
   ```

2. **Programmatic Validation:**
   ```python
   from pptx import Presentation
   prs = Presentation('WAVES_Intelligence_Executive_Briefing.pptx')
   assert len(prs.slides) == 13
   print("✓ Slide count correct")
   ```

3. **Narration Validation:**
   ```bash
   # Check all narration files exist
   ls -1 narration/slide_*.txt | wc -l
   # Should output: 13
   ```

## Troubleshooting

### Common Issues

**Issue:** Presentation won't open in PowerPoint
- **Cause:** Corrupted file or incompatible version
- **Solution:** Regenerate using `python generate_presentation.py`

**Issue:** Fonts appear incorrect
- **Cause:** PowerPoint using default font substitution
- **Solution:** Install system fonts or use PowerPoint's font embedding

**Issue:** Colors look different on screen
- **Cause:** Display calibration or RGB/CMYK conversion
- **Solution:** Verify RGB values match specification

**Issue:** Speaker notes missing
- **Cause:** PowerPoint view settings
- **Solution:** Enable Notes view (View > Notes Page)

**Issue:** Narration files contain encoding errors
- **Cause:** Non-UTF-8 system encoding
- **Solution:** Ensure Python uses UTF-8: `export PYTHONIOENCODING=utf-8`

## Performance Considerations

### File Size Optimization
- Minimal embedded images (placeholders only)
- Text-based content reduces file size
- No embedded fonts (system fonts used)
- Current size: ~67 KB (excellent for distribution)

### Generation Speed
- Typical generation time: <2 seconds
- Narration file creation: <1 second
- Total workflow: ~3 seconds

### Scalability
- Can generate 100+ presentations in parallel
- No external dependencies beyond python-pptx
- Fully automated, reproducible builds

## Security & Compliance

### Data Privacy
- No external API calls during generation
- No embedded analytics or tracking
- No telemetry or usage data collected
- All generation is local and offline

### Intellectual Property
- Content based on WAVES Intelligence™ platform
- Script approved by platform governance team
- No marketing claims or predictive statements
- Compliant with no-predict constraint

### Version Control
- All source code version-controlled in Git
- Presentation tracked in repository
- Full audit trail of changes
- Deterministic regeneration from source

## Maintenance & Support

### Update Procedures
1. Update script content in `executive_briefing_script.md`
2. Modify slide methods in `generate_presentation.py`
3. Update narration in `generate_narration.py`
4. Regenerate presentation
5. Update `VERSION_HISTORY.md`
6. Commit changes to repository

### Backup & Recovery
- Source files stored in Git repository
- Regeneration scripts version-controlled
- Narration text files backed up
- PowerPoint output can be regenerated at any time

### Contact & Support
- **Repository:** github.com/jasonheldman-creator/Waves-Simple
- **Issues:** Via GitHub Issues
- **Documentation:** This file + VERSION_HISTORY.md

## License & Attribution

**Copyright © 2024 WAVES Intelligence™**  
Presented by Vector™

All rights reserved. This presentation is proprietary to WAVES Intelligence™ and is provided for institutional review purposes only.

---

**Document Version:** 1.0.0  
**Last Updated:** December 24, 2024  
**Maintained by:** WAVES Intelligence™ Platform Team
