# Executive Briefing Assets

This directory contains the Executive Briefing assets for the WAVES Intelligence™ website.

## Required Files

### Primary Asset (Required)
- **executive-briefing.mp4** - The narrated WAVES Intelligence™ Executive Briefing video (approximately 8 minutes)
  - This file should be exported from the PowerPoint presentation with narration
  - Place the MP4 file in this directory as `executive-briefing.mp4`

### Optional Asset
- **WAVES_Intelligence_Executive_Briefing.pptx** - Optional downloadable slides
  - Copy from `/presentation/WAVES_Intelligence_Executive_Briefing.pptx` if offering downloads

## Video Export Instructions

To create the `executive-briefing.mp4` file from the PowerPoint presentation:

1. Open `/presentation/WAVES_Intelligence_Executive_Briefing.pptx`
2. Attach audio files from `/presentation/voiceover/` to each slide (see `/presentation/README.md`)
3. Export to video:
   - **File** → **Export** → **Create a Video**
   - Select **Full HD (1080p)** quality
   - Set seconds per slide to match narration duration
   - Click **Create Video**
   - Save as `executive-briefing.mp4` in this directory

## Notes
- The MP4 file is not included in the repository due to size constraints
- The video player on the website will gracefully handle the missing file until it's added
- Once the MP4 is added, it will be automatically served by Next.js from the public folder
