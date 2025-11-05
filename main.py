"""
AI Makeup Tutorial Backend - FastAPI
Install: pip install fastapi uvicorn python-multipart pillow opencv-python numpy tensorflow
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import json

app = FastAPI(title="AI Makeup Tutorial API")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
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

class ComparisonRequest(BaseModel):
    original_image: str  # base64
    current_image: str   # base64
    step_index: int
    look_id: str

# Makeup Looks Database
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

# Helper Functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def detect_face_landmarks(image: np.ndarray) -> dict:
    """Detect facial landmarks using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    landmarks = {
        "face_detected": len(faces) > 0,
        "face_count": len(faces),
        "eyes_detected": 0,
        "face_region": None
    }
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        landmarks["face_region"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        landmarks["eyes_detected"] = len(eyes)
    
    return landmarks

def analyze_skin_tone(image: np.ndarray, face_region: dict) -> dict:
    """Analyze skin tone from face region"""
    if face_region is None:
        return {"tone": "unknown", "rgb": [0, 0, 0]}
    
    x, y, w, h = face_region["x"], face_region["y"], face_region["w"], face_region["h"]
    
    # Extract face region
    face = image[y:y+h, x:x+w]
    
    # Calculate average color
    avg_color = cv2.mean(face)[:3]
    
    return {
        "tone": "detected",
        "rgb": [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])]  # BGR to RGB
    }

def calculate_symmetry(image: np.ndarray) -> float:
    """Calculate facial symmetry score"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Split image in half
    left_half = gray[:, :width//2]
    right_half = gray[:, width//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize if dimensions don't match
    if left_half.shape != right_half_flipped.shape:
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate similarity
    diff = cv2.absdiff(left_half, right_half_flipped)
    score = 100 - (np.mean(diff) / 255 * 100)
    
    return round(score, 2)

def analyze_color_coverage(image: np.ndarray, area: str) -> dict:
    """Analyze color coverage for specific makeup area"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for different makeup types
    color_ranges = {
        "lips": ([0, 50, 50], [10, 255, 255]),  # Red tones
        "eyes": ([90, 50, 50], [130, 255, 255]),  # Blue/purple tones
        "cheeks": ([160, 50, 50], [180, 255, 255])  # Pink tones
    }
    
    if area.lower() in ["lips", "eyes", "cheeks"]:
        lower = np.array(color_ranges.get(area.lower(), ([0, 0, 0]))[0])
        upper = np.array(color_ranges.get(area.lower(), ([180, 255, 255]))[1])
        
        mask = cv2.inRange(hsv, lower, upper)
        coverage = (np.sum(mask > 0) / mask.size) * 100
        
        return {
            "coverage_percentage": round(coverage, 2),
            "well_applied": coverage > 1.0
        }
    
    return {"coverage_percentage": 0, "well_applied": False}

def compare_images(original: np.ndarray, current: np.ndarray, step_area: str) -> dict:
    """Compare original and current images to provide feedback"""
    
    # Detect faces in both images
    orig_landmarks = detect_face_landmarks(original)
    curr_landmarks = detect_face_landmarks(current)
    
    if not curr_landmarks["face_detected"]:
        return {
            "error": "No face detected in current image",
            "score": 0
        }
    
    # Calculate symmetry
    symmetry = calculate_symmetry(current)
    
    # Analyze color coverage
    color_analysis = analyze_color_coverage(current, step_area)
    
    # Calculate difference between images
    orig_resized = cv2.resize(original, (current.shape[1], current.shape[0]))
    diff = cv2.absdiff(orig_resized, current)
    diff_score = np.mean(diff)
    
    # Generate score based on multiple factors
    base_score = 70
    symmetry_bonus = min(symmetry / 5, 15)
    coverage_bonus = min(color_analysis["coverage_percentage"] * 2, 10)
    change_bonus = min(diff_score / 10, 5)
    
    final_score = min(int(base_score + symmetry_bonus + coverage_bonus + change_bonus), 100)
    
    return {
        "score": final_score,
        "symmetry_score": symmetry,
        "color_analysis": color_analysis,
        "difference_detected": diff_score > 10
    }

def generate_feedback(score: int, area: str, analysis: dict) -> dict:
    """Generate contextual feedback based on score and area"""
    
    feedback_templates = {
        "high": {
            "positive": [
                f"Excellent work on your {area}! The application is even and well-blended.",
                f"Great job! Your {area} looks professionally done.",
                f"Perfect technique! The {area} is beautifully executed.",
                f"Outstanding! Your {area} application shows real skill."
            ],
            "improvement": [
                "Just a tiny bit more blending at the edges could make it flawless.",
                "Consider adding slightly more product for extra dimension.",
                "Perfect! Maybe experiment with slightly bolder application next time."
            ],
            "tip": [
                "You've mastered this step! This technique will serve you well.",
                "Pro tip: This is competition-level work. Keep it up!",
                "Your blending skills are excellent. Try teaching others!"
            ]
        },
        "medium": {
            "positive": [
                f"Good start with your {area}! You're on the right track.",
                f"Nice effort! Your {area} is coming together well.",
                f"You're doing well! The {area} has good placement.",
                f"Solid application! Your {area} is developing nicely."
            ],
            "improvement": [
                "Try blending a bit more for a seamless finish.",
                "Build up the color gradually for better control.",
                "Focus on symmetry - check both sides match evenly.",
                "Use a lighter hand and layer the product slowly."
            ],
            "tip": [
                "Practice makes perfect! Try this step again for improvement.",
                "Use circular motions when blending for best results.",
                "Less is more - you can always add more product.",
                "Check your work in natural lighting for accuracy."
            ]
        },
        "low": {
            "positive": [
                f"You've made a good attempt at the {area}!",
                f"Keep practicing! The {area} shows promise.",
                f"Don't worry, {area} application takes practice.",
                f"Good effort! Let's refine your {area} technique."
            ],
            "improvement": [
                "Focus on even application across the entire area.",
                "Try using less pressure and building color slowly.",
                "Watch tutorial videos for this specific technique.",
                "Consider practicing this step a few more times."
            ],
            "tip": [
                "Don't be discouraged! Even professionals practice constantly.",
                "Try practicing on your hand first to get the feel.",
                "Use good lighting and take your time with each step.",
                "Consider investing in quality tools for better results."
            ]
        }
    }
    
    category = "high" if score >= 80 else "medium" if score >= 60 else "low"
    templates = feedback_templates[category]
    
    import random
    return {
        "positive": random.choice(templates["positive"]),
        "improvement": random.choice(templates["improvement"]),
        "tip": random.choice(templates["tip"])
    }

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AI Makeup Tutorial API", "version": "1.0.0"}

@app.get("/api/looks", response_model=List[MakeupLook])
async def get_makeup_looks():
    """Get all available makeup looks"""
    return MAKEUP_LOOKS

@app.get("/api/looks/{look_id}", response_model=MakeupLook)
async def get_makeup_look(look_id: str):
    """Get specific makeup look by ID"""
    look = next((look for look in MAKEUP_LOOKS if look["id"] == look_id), None)
    if not look:
        raise HTTPException(status_code=404, detail="Makeup look not found")
    return look

@app.post("/api/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """Analyze uploaded face image for landmarks and skin tone"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        landmarks = detect_face_landmarks(image)
        
        if not landmarks["face_detected"]:
            return {
                "success": False,
                "message": "No face detected in image. Please upload a clear face photo."
            }
        
        skin_tone = analyze_skin_tone(image, landmarks["face_region"])
        
        return {
            "success": True,
            "face_detected": True,
            "landmarks": landmarks,
            "skin_tone": skin_tone,
            "message": "Face analyzed successfully!"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare-makeup", response_model=FeedbackResponse)
async def compare_makeup(
    original_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    step_index: int = 0,
    look_id: str = "natural"
):
    """Compare original and current makeup images and provide feedback"""
    try:
        # Read images
        orig_contents = await original_file.read()
        curr_contents = await current_file.read()
        
        orig_arr = np.frombuffer(orig_contents, np.uint8)
        curr_arr = np.frombuffer(curr_contents, np.uint8)
        
        original = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
        current = cv2.imdecode(curr_arr, cv2.IMREAD_COLOR)
        
        if original is None or current is None:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        # Get step information
        look = next((look for look in MAKEUP_LOOKS if look["id"] == look_id), None)
        if not look or step_index >= len(look["steps"]):
            raise HTTPException(status_code=404, detail="Invalid look or step index")
        
        step_area = look["steps"][step_index]["area"]
        
        # Compare images
        analysis = compare_images(original, current, step_area)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Generate feedback
        feedback = generate_feedback(analysis["score"], step_area, analysis)
        
        return FeedbackResponse(
            area=step_area,
            score=analysis["score"],
            positive=feedback["positive"],
            improvement=feedback["improvement"],
            tip=feedback["tip"],
            color_analysis=analysis["color_analysis"],
            symmetry_score=analysis["symmetry_score"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/final-comparison")
async def final_comparison(
    original_file: UploadFile = File(...),
    final_file: UploadFile = File(...),
    look_id: str = "natural"
):
    """
    Comprehensive final comparison of original and completed makeup look.
    Analyzes overall transformation and provides detailed feedback.
    """
    try:
        # Read images
        orig_contents = await original_file.read()
        final_contents = await final_file.read()
        
        orig_arr = np.frombuffer(orig_contents, np.uint8)
        final_arr = np.frombuffer(final_contents, np.uint8)
        
        original = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
        final = cv2.imdecode(final_arr, cv2.IMREAD_COLOR)
        
        if original is None or final is None:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        # Get look information
        look = next((look for look in MAKEUP_LOOKS if look["id"] == look_id), None)
        if not look:
            raise HTTPException(status_code=404, detail="Makeup look not found")
        
        # Perform comprehensive analysis
        result = perform_final_analysis(original, final, look)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def perform_final_analysis(original: np.ndarray, final: np.ndarray, look: dict) -> dict:
    """Perform comprehensive analysis comparing original and final makeup"""
    
    # Resize images to same size for comparison
    final_resized = cv2.resize(final, (original.shape[1], original.shape[0]))
    
    # 1. Detect faces and landmarks
    orig_landmarks = detect_face_landmarks(original)
    final_landmarks = detect_face_landmarks(final_resized)
    
    if not orig_landmarks["face_detected"] or not final_landmarks["face_detected"]:
        return {
            "error": "Face not detected in one or both images",
            "overall_score": 0
        }
    
    # 2. Calculate symmetry score
    symmetry_score = calculate_symmetry(final_resized)
    
    # 3. Analyze color changes
    color_analysis = analyze_color_transformation(original, final_resized)
    
    # 4. Calculate brightness increase
    brightness_increase = calculate_brightness_change(original, final_resized)
    
    # 5. Analyze coverage and application
    coverage_score = analyze_overall_coverage(final_resized)
    
    # 6. Calculate individual component scores
    technique_score = min(int(70 + (symmetry_score / 5) + coverage_score / 10), 100)
    color_score = min(int(65 + color_analysis['harmony_score'] + coverage_score / 15), 100)
    blend_score = min(int(60 + (symmetry_score / 4) + color_analysis['consistency_score'] / 2), 100)
    definition_score = min(int(70 + coverage_score / 8 + brightness_increase / 5), 100)
    
    # 7. Calculate overall score (weighted average)
    overall_score = int(
        (technique_score * 0.3) +
        (color_score * 0.25) +
        (symmetry_score * 0.2) +
        (blend_score * 0.15) +
        (definition_score * 0.1)
    )
    
    # 8. Generate contextual feedback
    strengths = generate_strengths(overall_score, symmetry_score, color_analysis)
    improvements = generate_improvements(overall_score, technique_score, color_score)
    tips = generate_pro_tips(look["difficulty"], overall_score)
    
    return {
        "overall_score": overall_score,
        "technique_score": technique_score,
        "color_score": color_score,
        "symmetry_score": int(symmetry_score),
        "blend_score": blend_score,
        "definition_score": definition_score,
        "coverage_percentage": int(coverage_score),
        "brightness_increase": int(brightness_increase),
        "color_harmony": color_analysis['harmony_score'],
        "strengths": strengths,
        "improvements": improvements,
        "tips": tips,
        "transformation_detected": True
    }

def analyze_color_transformation(original: np.ndarray, final: np.ndarray) -> dict:
    """Analyze color changes between original and final"""
    
    # Convert to HSV for better color analysis
    orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    
    # Calculate color diversity in final image (more colors = more makeup)
    final_colors = np.unique(final_hsv.reshape(-1, 3), axis=0)
    color_diversity = min(len(final_colors) / 1000, 1.0) * 100
    
    # Calculate color consistency (how well colors blend)
    h_std = np.std(final_hsv[:, :, 0])
    consistency_score = max(0, 100 - h_std)
    
    # Calculate harmony score based on color theory
    harmony_score = (color_diversity * 0.6 + consistency_score * 0.4) / 2
    
    return {
        "harmony_score": round(harmony_score, 2),
        "consistency_score": round(consistency_score, 2),
        "diversity": round(color_diversity, 2)
    }

def calculate_brightness_change(original: np.ndarray, final: np.ndarray) -> float:
    """Calculate brightness increase percentage"""
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    
    orig_brightness = np.mean(orig_gray)
    final_brightness = np.mean(final_gray)
    
    brightness_change = ((final_brightness - orig_brightness) / orig_brightness) * 100
    
    return max(0, min(brightness_change, 50))  # Cap at 50% increase

def analyze_overall_coverage(image: np.ndarray) -> float:
    """Analyze overall makeup coverage"""
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for skin tones with makeup
    # Higher saturation usually indicates makeup
    mask = cv2.inRange(hsv[:, :, 1], 30, 255)
    
    coverage = (np.sum(mask > 0) / mask.size) * 100
    
    return min(coverage, 100)

def generate_strengths(overall_score: int, symmetry: float, color_analysis: dict) -> list:
    """Generate list of strengths based on analysis"""
    
    strengths = []
    
    if overall_score >= 85:
        strengths.append("Exceptional overall technique showing professional-level skills")
    elif overall_score >= 75:
        strengths.append("Strong application technique with consistent results")
    else:
        strengths.append("Good foundation with clear understanding of basics")
    
    if symmetry >= 85:
        strengths.append("Excellent facial symmetry - both sides are beautifully balanced")
    elif symmetry >= 75:
        strengths.append("Good symmetry with well-matched application on both sides")
    
    if color_analysis['harmony_score'] >= 40:
        strengths.append("Beautiful color choices that complement your features")
    
    if color_analysis['consistency_score'] >= 70:
        strengths.append("Smooth, well-blended application with seamless transitions")
    
    # Ensure at least 3 strengths
    if len(strengths) < 3:
        strengths.append("Clear effort and attention to detail in your application")
    
    return strengths[:4]  # Return top 4 strengths

def generate_improvements(overall_score: int, technique: int, color: int) -> list:
    """Generate improvement suggestions"""
    
    improvements = []
    
    if technique < 75:
        improvements.append("Practice blending techniques - use circular motions with a fluffy brush")
    
    if color < 70:
        improvements.append("Experiment with color placement to enhance natural features")
    
    if overall_score < 80:
        improvements.append("Build up products gradually in thin layers for better control")
    
    if technique < 70 or color < 70:
        improvements.append("Use primer to help makeup last longer and apply more smoothly")
    
    # Add general improvement tips
    improvements.append("Take your time with each step - rushing can compromise results")
    improvements.append("Check your work in natural lighting for the most accurate view")
    
    return improvements[:4]  # Return top 4 improvements

def generate_pro_tips(difficulty: str, score: int) -> list:
    """Generate pro tips based on difficulty level and score"""
    
    tips = []
    
    if score >= 85:
        tips.append("You're doing amazing! Try teaching others your techniques")
        tips.append("Consider trying more advanced looks to challenge yourself")
    else:
        tips.append("Watch tutorial videos in slow motion to catch subtle techniques")
        tips.append("Practice the same look multiple times to build muscle memory")
    
    if difficulty == "Beginner":
        tips.append("Master these basics before moving to intermediate looks")
        tips.append("Invest in a few quality brushes - they make a huge difference")
    elif difficulty == "Intermediate":
        tips.append("Focus on perfecting your blending for that professional finish")
        tips.append("Experiment with different color combinations within the same family")
    else:
        tips.append("Professional makeup is all about layering and patience")
        tips.append("Consider a makeup-setting spray for all-day perfection")
    
    tips.append("Always prep your skin properly - it's the canvas for your art")
    tips.append("Natural lighting is your best friend when applying makeup")
    
    return tips[:4]  # Return top 4 tips

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # Default 10000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)
