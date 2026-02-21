---
title: SkinProAI
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SkinProAI

AI-assisted dermoscopic lesion analysis for clinical decision support.

## Features

- **Patient management** — create and select patient profiles
- **Image analysis** — upload dermoscopic images for automated assessment via MedGemma visual examination, MONET feature extraction, and ConvNeXt classification
- **Temporal comparison** — sequential images are automatically compared to detect change over time
- **Grad-CAM visualisation** — attention maps highlight regions driving the classification
- **Persistent chat history** — full analysis cascade is stored and replayed on reload

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript (Vite) |
| Backend | FastAPI + uvicorn |
| Vision-language model | MedGemma (Google) via Hugging Face |
| Classifier | ConvNeXt fine-tuned on ISIC HAM10000 |
| Feature extraction | MONET skin concept probes |
| Explainability | Grad-CAM |

## Usage

1. Open the app and create a patient record
2. Click the patient card to open the chat
3. Attach a dermoscopic image and send — analysis runs automatically
4. Upload further images for the same patient to trigger temporal comparison
5. Ask follow-up questions in text to query the AI about the findings

## Disclaimer

SkinProAI is a research prototype intended for educational and investigational use only. It is **not** a certified medical device and must not be used as a substitute for professional clinical judgement.
