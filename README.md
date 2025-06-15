# Scene Manipulation via Text-Controlled Object Relocation and Relighting

Objective: To design a pipeline that takes an input scene image and a natural language instruction that combines **object detection**, **segmentation**, and **relighting**.

## Implemented:
- [x] Text Instruction Parsing 
- [x] Object Detection
- [x] Object Segmentation
- [x] Object Relocation

## Not Yet Implemented:
- [ ] Relighting

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/mahaqj/Scene-Manipulation.git
cd Scene-Manipulation
```

---

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4. Download Model Weights

Download manually from [Segment Anything’s Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)  
and place it here:

```
segment-anything/sam_vit_b_01ec64.pth
```

Or, if Git LFS is set up:

```bash
git lfs pull
```

---

## How to Run

```bash
python scene_manip.py
```

You’ll be prompted to:

- Select an image  
- Type a natural language instruction (e.g., "Move the dog to the left and add sunset light")  

---

## How it Works

1. `spaCy` extracts the object, action, direction, and lighting from the user's instruction
2. DETR detects the object in the image
3. SAM generates a precise mask of the object
4. The object is cut out using the selected mask
5. Stable Diffusion inpaints the region where the object was removed
6. It is then repositioned based on the instruction and pasted onto the updated image
7. Results are displayed side-by-side

---

## Example Outputs

![Figure_1_updated](https://github.com/user-attachments/assets/e781719c-ac80-4bbe-97a0-4bad75f66a08)
![Figure_12](https://github.com/user-attachments/assets/4bcfcd6d-ba6c-4774-9b92-caafffadb9c5)
![Figure_13](https://github.com/user-attachments/assets/de5830c1-4a34-4352-89a5-ce6f6bce289e)

---

## Tech Stack

- Python  
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)  
- [DETR](https://github.com/facebookresearch/detectron2)  
- OpenCV, PIL, spaCy

---

## Credits

- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta  
- [DETR](https://github.com/facebookresearch/detectron2)  

---

## Notes

- `segment-anything/` contains only necessary files from the original repository by Meta AI
