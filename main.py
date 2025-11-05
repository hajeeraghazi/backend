from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from datetime import datetime

app = FastAPI(title="AI Makeup Tutorial API (Simplified)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MakeupStep(BaseModel):
    area: str
    instruction: str
    products: List[str]
    tips: str

class MakeupLook(BaseModel):
    id: str
    name: str
    occasion: str
    difficulty: str
    duration: str
    description: str
    steps: List[MakeupStep]

class FeedbackResponse(BaseModel):
    area: str
    score: int
    positive: str
    improvement: str
    tip: str
    color_analysis: dict
    symmetry_score: float

MAKEUP_LOOKS = [
    {
        "id": "natural",
        "name": "Natural Day Look",
        "occasion": "Everyday, Work, Casual",
        "difficulty": "Beginner",
        "duration": "15 mins",
        "description": "Fresh, minimal makeup for a polished everyday appearance",
        "steps": [
            {
                "area": "Skin Prep",
                "instruction": "Apply moisturizer and primer all over face",
                "products": ["Moisturizer", "Primer"],
                "tips": "Wait 2-3 minutes for products to absorb before moving to next step"
            },
            {
                "area": "Foundation",
                "instruction": "Apply light coverage foundation or BB cream",
                "products": ["Foundation/BB Cream", "Beauty Sponge"],
                "tips": "Blend from center of face outward for natural finish"
            },
            {
                "area": "Concealer",
                "instruction": "Apply concealer under eyes in triangle shape and on blemishes",
                "products": ["Concealer", "Small Brush"],
                "tips": "Use one shade lighter than skin tone for brightening effect"
            },
            {
                "area": "Eyebrows",
                "instruction": "Fill in eyebrows lightly with pencil or powder",
                "products": ["Brow Pencil/Powder", "Spoolie"],
                "tips": "Follow natural brow shape, brush hairs up and out"
            },
            {
                "area": "Eyes",
                "instruction": "Apply neutral eyeshadow on lids, add light brown in crease",
                "products": ["Neutral Eyeshadow Palette", "Blending Brush"],
                "tips": "Blend well for seamless transition between colors"
            },
            {
                "area": "Mascara",
                "instruction": "Apply one coat of mascara to upper lashes",
                "products": ["Black/Brown Mascara"],
                "tips": "Wiggle wand at base of lashes, then sweep upward"
            },
            {
                "area": "Cheeks",
                "instruction": "Apply soft blush to apples of cheeks",
                "products": ["Pink/Peach Blush", "Blush Brush"],
                "tips": "Smile and apply to the rounded part, blend toward temples"
            },
            {
                "area": "Lips",
                "instruction": "Apply nude or pink lip gloss or tinted balm",
                "products": ["Lip Gloss/Tinted Balm"],
                "tips": "Choose shade close to natural lip color for subtle enhancement"
            }
        ]
    },
    {
        "id": "glam",
        "name": "Evening Glam",
        "occasion": "Party, Dinner, Night Out",
        "difficulty": "Intermediate",
        "duration": "35 mins",
        "description": "Bold and glamorous look for special evening events",
        "steps": [
            {
                "area": "Skin Prep",
                "instruction": "Apply moisturizer, primer, and illuminating base",
                "products": ["Moisturizer", "Primer", "Illuminator"],
                "tips": "Focus illuminator on high points: cheekbones, brow bones, cupids bow"
            },
            {
                "area": "Foundation",
                "instruction": "Apply full coverage foundation with brush or sponge",
                "products": ["Full Coverage Foundation", "Brush/Sponge"],
                "tips": "Build coverage gradually, blend down to neck"
            },
            {
                "area": "Contour & Highlight",
                "instruction": "Contour cheekbones, jawline, nose. Highlight high points",
                "products": ["Contour Powder", "Highlighter", "Brushes"],
                "tips": "Blend thoroughly - no harsh lines"
            },
            {
                "area": "Eyes - Dramatic",
                "instruction": "Apply dramatic eyeshadow, liner, and false lashes",
                "products": ["Eyeshadow Palette", "Eyeliner", "False Lashes", "Mascara"],
                "tips": "Build intensity gradually with shimmer and dark shades"
            },
            {
                "area": "Cheeks",
                "instruction": "Apply blush and additional highlighter",
                "products": ["Blush", "Highlighter"],
                "tips": "Build color intensity gradually for balanced look"
            },
            {
                "area": "Lips",
                "instruction": "Line lips, apply bold lipstick, add gloss",
                "products": ["Lip Liner", "Lipstick", "Lip Gloss"],
                "tips": "Blot after first layer, reapply for long-lasting color"
            }
        ]
    },
    {
        "id": "bridal",
        "name": "Bridal Elegance",
        "occasion": "Wedding, Formal Events",
        "difficulty": "Advanced",
        "duration": "60 mins",
        "description": "Timeless, photograph-ready makeup for your special day",
        "steps": [
            {
                "area": "Skin Prep & Prime",
                "instruction": "Double cleanse, apply serum, moisturizer, and primer",
                "products": ["Cleanser", "Serum", "Moisturizer", "Primer"],
                "tips": "Well-prepped skin ensures flawless makeup that lasts all day"
            },
            {
                "area": "Color Correction & Foundation",
                "instruction": "Apply color corrector and HD foundation",
                "products": ["Color Corrector", "HD Foundation"],
                "tips": "Use transfer-resistant formula for long wear"
            },
            {
                "area": "Concealing & Setting",
                "instruction": "Layer concealer and bake with powder",
                "products": ["Concealer", "Translucent Powder"],
                "tips": "Let powder sit while doing eyes for crease-free finish"
            },
            {
                "area": "Contour & Bronze",
                "instruction": "Subtly contour and add warmth",
                "products": ["Contour", "Bronzer"],
                "tips": "Keep it soft - photos amplify everything"
            },
            {
                "area": "Eyes - Romantic",
                "instruction": "Create romantic eye look with soft shimmer",
                "products": ["Eye Primer", "Eyeshadow Palette", "Brushes"],
                "tips": "Use champagne and rose gold tones for timeless photos"
            },
            {
                "area": "Lashes & Definition",
                "instruction": "Apply false lashes and waterproof mascara",
                "products": ["False Lashes", "Waterproof Mascara"],
                "tips": "Natural-looking lashes photograph beautifully"
            },
            {
                "area": "Highlight & Blush",
                "instruction": "Layer cream and powder blush with luminous highlight",
                "products": ["Cream Blush", "Powder Blush", "Highlighter"],
                "tips": "Layering ensures color lasts through tears and kisses"
            },
            {
                "area": "Lips & Setting",
                "instruction": "Perfect lips with liner, lipstick, and set everything",
                "products": ["Lip Liner", "Lipstick", "Setting Spray"],
                "tips": "Choose classic tones for timeless wedding photos"
            }
        ]
    }
]


@app.get("/")
async def root():
    return {"message":"AI Makeup Tutorial API (Simplified)","version":"1.0.0"}

@app.get("/api/looks", response_model=List[MakeupLook])
async def get_makeup_looks():
    return MAKEUP_LOOKS

@app.get("/api/looks/{look_id}", response_model=MakeupLook)
async def get_makeup_look(look_id: str):
    look = next((l for l in MAKEUP_LOOKS if l["id"]==look_id), None)
    if not look:
        raise HTTPException(status_code=404, detail="Makeup look not found")
    return look

@app.post("/api/compare-makeup", response_model=FeedbackResponse)
async def compare_makeup(original_file: UploadFile = File(...), current_file: UploadFile = File(...), step_index: int = 0, look_id: str = "natural"):
    # This simplified endpoint does not process images. It returns deterministic feedback for demo.
    step_area = "Unknown"
    look = next((l for l in MAKEUP_LOOKS if l["id"]==look_id), None)
    if look and step_index < len(look["steps"]):
        step_area = look["steps"][step_index]["area"]
    score = 85
    return FeedbackResponse(
        area=step_area,
        score=score,
        positive=f"Nice job on your {step_area}!",
        improvement="Blend a bit more at the edges.",
        tip="Check your work in natural light.",
        color_analysis={"coverage_percentage": 5.0, "well_applied": True},
        symmetry_score=92.5
    )

@app.get("/health")
async def health_check():
    return {"status":"healthy","timestamp": datetime.now().isoformat()}
