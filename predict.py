import torch
from PIL import Image
from torchvision import transforms

from approach2_end_to_end import EndToEndDLModel
from config import (
    DATASETS_CONFIG,
    MODELS_CONFIG,
    TRAINING_CONFIG,
    AUGMENTATION_CONFIG,
    ML_CLASSIFIER_CONFIG
)

# =========================
# CONFIG (مهم جدًا يكون مطابق للتدريب)
# =========================
config = {
    'DATASETS_CONFIG': DATASETS_CONFIG[0],   # 👈 Brain MRI (4 classes)
    'MODEL_CONFIG': MODELS_CONFIG[0],        # 👈 نفس الموديل اللي اتدرب
    'TRAINING_CONFIG': TRAINING_CONFIG,
    'AUGMENTATION_CONFIG': AUGMENTATION_CONFIG,
    'ML_CLASSIFIER_CONFIG': ML_CLASSIFIER_CONFIG
}

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EndToEndDLModel(config)
model.build_model()

# ===== load weights =====
weights_path = "/home/yossef/deep_learning/models/Brain-MRI_EfficientNet-B0/end_to_end_model.pth"


state_dict = torch.load(weights_path, map_location=device)
model.model.load_state_dict(state_dict)

model.model.to(device)
model.model.eval()

# =========================
# TRANSFORM (لازم نفس التدريب)
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(                      # 👈 أهم سطر
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# CLASS NAMES
# =========================
class_names = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print("\nRAW OUTPUT:", output)
    print("PROBABILITIES:", probs)

    return pred

# =========================
# TEST
# =========================
image_path = "tee.jpeg"  # 👈 حط صورتك هنا

result = predict_image(image_path)

print("\n======================")
print("PREDICTION RESULT:")
print("======================")
print("➡", class_names.get(result, "Unknown Class"))