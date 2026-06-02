# ============================================================
# PROJECT 4: Image or Text Recognition (Basic)
# DecodeLabs Industrial Training Kit | Batch 2026
# ============================================================
# Goal      : Implement basic recognition using pre-trained models
# Track A   : TEXT Recognition — Named Entity Recognition (NER)
#             using spaCy (no GPU, no training needed)
# Track B   : IMAGE Classification — using a pre-trained
#             MobileNetV2 model via TensorFlow/Keras (optional)
# ============================================================
# Run Track A only if you don't have TF installed.
# Run Track B if you have TensorFlow installed.
# ============================================================

# ─────────────────────────────────────────────────────────────────────────────
# TRACK A: TEXT RECOGNITION — Named Entity Recognition (NER)
#
# Uses spaCy's pre-trained English model (en_core_web_sm).
# Install: pip install spacy && python -m spacy download en_core_web_sm
#
# What it does: Identifies and labels named entities in raw text —
# people, organisations, locations, dates, money, etc.
# ─────────────────────────────────────────────────────────────────────────────

def run_text_recognition():
    """Track A: NER-based text recognition using spaCy."""
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy")
        print("Then: python -m spacy download en_core_web_sm")
        return

    print("\n" + "=" * 60)
    print("  TRACK A: Text Recognition — Named Entity Recognition")
    print("=" * 60)

    # Load pre-trained model (no training required — key requirement)
    print("\n⏳ Loading pre-trained spaCy model (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model not found. Run: python -m spacy download en_core_web_sm")
        return
    print("✅ Model loaded successfully.\n")

    # Sample texts to demonstrate recognition
    sample_texts = [
        "Elon Musk founded SpaceX in 2002 in Hawthorne, California.",
        "Google was founded by Larry Page and Sergey Brin in September 1998.",
        "Apple's revenue in Q4 2023 was $89.5 billion, surpassing analyst expectations.",
        "The United Nations headquarters is located in New York City, USA.",
        "Amazon acquired Whole Foods Market for $13.7 billion in 2017.",
    ]

    for i, text in enumerate(sample_texts, 1):
        print(f"📄 Sample {i}: {text}")
        doc = nlp(text)

        if doc.ents:
            print("   🔍 Entities Found:")
            for ent in doc.ents:
                print(f"      [{ent.label_:>12}]  →  '{ent.text}'")
        else:
            print("   ⚠️  No named entities detected.")
        print()

    # ── Interactive Mode ──────────────────────────────────────
    print("-" * 60)
    print("🖊️  Enter your own text for NER (or press Enter to skip):")
    user_text = input("   Your text: ").strip()

    if user_text:
        doc = nlp(user_text)
        print("\n   🔍 Entities in your text:")
        if doc.ents:
            for ent in doc.ents:
                print(f"      [{ent.label_:>12}]  →  '{ent.text}'  ({spacy.explain(ent.label_)})")
        else:
            print("   No named entities detected.")

    print("\n✅ Track A complete!")


# ─────────────────────────────────────────────────────────────────────────────
# TRACK B: IMAGE RECOGNITION — Pre-trained MobileNetV2 (ImageNet)
#
# Uses TensorFlow/Keras and a pre-trained MobileNetV2 model.
# Install: pip install tensorflow pillow requests
#
# What it does: Classifies any image into 1000 ImageNet categories
# using a production-grade pre-trained CNN — no training needed.
# ─────────────────────────────────────────────────────────────────────────────

def run_image_recognition():
    """Track B: Image classification using pre-trained MobileNetV2."""
    try:
        import numpy as np
        import requests
        from PIL import Image
        from io import BytesIO
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import (
            preprocess_input,
            decode_predictions,
        )
    except ImportError as e:
        print(f"\n⚠️  Missing dependency: {e}")
        print("Run: pip install tensorflow pillow requests")
        return

    print("\n" + "=" * 60)
    print("  TRACK B: Image Recognition — MobileNetV2 (ImageNet)")
    print("=" * 60)

    # Load pre-trained model (weights download on first run ~14MB)
    print("\n⏳ Loading pre-trained MobileNetV2 model...")
    model = MobileNetV2(weights="imagenet")
    print("✅ Model loaded. 1000 ImageNet classes available.\n")

    # Sample image URLs to classify
    sample_images = [
        ("Cat",      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"),
        ("Elephant", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/1200px-African_Bush_Elephant.jpg"),
        ("Car",      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2014_ferrari_458_spider.jpg/1200px-2014_ferrari_458_spider.jpg"),
    ]

    def classify_image_from_url(url: str, label: str):
        """Download image from URL and classify with MobileNetV2."""
        print(f"📷 Classifying: {label}")
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))                   # MobileNetV2 input size
            arr = np.array(img)
            arr = np.expand_dims(arr, axis=0)              # Shape: (1, 224, 224, 3)
            arr = preprocess_input(arr)                    # Normalize to [-1, 1]

            preds = model.predict(arr, verbose=0)
            top3  = decode_predictions(preds, top=3)[0]

            print("   🔍 Top-3 Predictions:")
            for rank, (_, class_name, confidence) in enumerate(top3, 1):
                bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                print(f"      #{rank} {class_name.replace('_', ' '):25} {confidence*100:5.1f}%  [{bar}]")
        except Exception as e:
            print(f"   ⚠️  Could not process image: {e}")
        print()

    for label, url in sample_images:
        classify_image_from_url(url, label)

    # ── Interactive: classify a local image file ──────────────
    print("-" * 60)
    print("🖼️  Enter path to a local image file (or press Enter to skip):")
    path = input("   File path: ").strip()

    if path:
        try:
            import numpy as np
            img = Image.open(path).convert("RGB").resize((224, 224))
            arr = preprocess_input(np.expand_dims(np.array(img), 0))
            preds = model.predict(arr, verbose=0)
            top3  = decode_predictions(preds, top=3)[0]
            print("\n   🔍 Top-3 Predictions for your image:")
            for rank, (_, class_name, confidence) in enumerate(top3, 1):
                print(f"      #{rank} {class_name.replace('_', ' '):25} {confidence*100:.1f}%")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

    print("\n✅ Track B complete!")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Menu to choose which track to run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DecodeLabs | Project 4: Image & Text Recognition")
    print("  Building the Machine's Optic Nerve")
    print("=" * 60)
    print("\nSelect recognition track:")
    print("  [1] Text Recognition  — Named Entity Recognition (NER) via spaCy")
    print("  [2] Image Recognition — MobileNetV2 pre-trained classifier")
    print("  [3] Run Both Tracks")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        run_text_recognition()
    elif choice == "2":
        run_image_recognition()
    elif choice == "3":
        run_text_recognition()
        run_image_recognition()
    else:
        print("Invalid choice. Defaulting to Track A (Text Recognition).")
        run_text_recognition()

    print("\n" + "=" * 60)
    print("  Project 4 Complete!")
    print("  You have successfully implemented basic AI recognition.")
    print("=" * 60)


if __name__ == "__main__":
    main()
