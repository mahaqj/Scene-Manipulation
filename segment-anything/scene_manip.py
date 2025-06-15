import spacy
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# -------------------------------------------------------------------------------------
# step 1 -> text instruction parsing
# -------------------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm") # small english model
print("-------------------------------------------------------------------------------")
print("enter an instruction like: 'move the dog to the right and add sunset lighting)")
print("possible directions: left, right, up, down, center")
print("possible lightings: sunrise, sunset, nighttime, daytime")
print("-------------------------------------------------------------------------------")
instruction = input("instruction: ")
print("-------------------------------------------------------------------------------")
x = nlp(instruction) # apply spacy nlp to the user's instruction

nouns = [] # get nouns
for token in x: # token selects 1 word at a time
    if token.pos_ == "NOUN":
        nouns.append(token.lemma_)
target_object = nouns[0] if nouns else None

verbs = [] # get verbs
for token in x:
    if token.pos_ == "VERB":
        verbs.append(token.lemma_)
action = verbs[0] if verbs else None

directions = [] # get direction
for token in x:
    if token.text in ["left", "right", "up", "down", "center"]:
        directions.append(token.text)
direction = directions[0] if directions else None

possible_lightings = ["sunset", "sunrise", "nighttime", "daytime"]
lighting = None # get lighting
for word in instruction.split():
    if word in possible_lightings:
        lighting = word
        break

print(f"parsed: object-{target_object}, action-{action}, direction-{direction}, lighting-{lighting}") # select the first everything for now

# -------------------------------------------------------------------------------------
# step 2 -> load image
# -------------------------------------------------------------------------------------
image_path = "dog1.jpg"
image_pil = Image.open(image_path)
image_cv2 = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

# -------------------------------------------------------------------------------------
# step 3 -> detr object detection
# -------------------------------------------------------------------------------------
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50") # resizes, normalizes, scales etc. to convert image to correct format
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50") # scans image and returns boxes and labels for detected objects
inputs = processor(images=image_pil, return_tensors="pt") # converts image to tensor
outputs = model(**inputs) # runs detection and gives boxes and labels

image_size = image_pil.size[::-1] # (height, width)
results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=[image_size])[0] # get first result from the list

box = None
for i in range(len(results["labels"])):
    label_id = results["labels"][i].item() # get label id
    label_name = model.config.id2label[label_id] # convert id to text
    if label_name.lower() == target_object: # check if it's our target object
        box = results["boxes"][i].tolist() # get its box
        box_label = label_name # save the name
        break

if not box:
    print("object not found in image :(")
    exit()

x1, y1, x2, y2 = box
x_center = int((x1 + x2)/2)
y_center = int((y1 + y2)/2)

# -------------------------------------------------------------------------------------
# step 4 -> sam for mask prediction
# -------------------------------------------------------------------------------------
input_point = np.array([[x_center, y_center]]) # target object is most likely here (middle of the box)
input_label = np.array([1]) # 1 means the input_point is on the object we want

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

# now select best mask out of 3 based on confidence score * area covered
areas = [] # calculate area covered for each mask
for mask in masks:
    area = np.sum(mask)
    areas.append(area)

combined_scores = [] # multiply each score by its corresponding area
for i in range(len(scores)):
    score = scores[i]
    area = areas[i]
    combined = score * area
    combined_scores.append(combined)

best_index = np.argmax(combined_scores)
selected_mask = masks[best_index]

# -------------------------------------------------------------------------------------
# step 5 -> cut out object and mask it
# -------------------------------------------------------------------------------------
height, width = selected_mask.shape

ys = []
xs = []
for row in range(height):
    for col in range(width):
        if selected_mask[row, col]: # check if object pixel is here (1)
            ys.append(row)
            xs.append(col)

# find the smallest and largest rows/cols with object
up = min(ys)
down = max(ys)
left = min(xs)
right = max(xs)

# create a new blank/black image of same size
cutout = np.zeros_like(image_rgb)

# copy all the object pixels from original image to the blank image
for row in range(height):
    for col in range(width):
        if selected_mask[row, col]:
            cutout[row, col] = image_rgb[row, col]

crop = cutout[up:down, left:right] # cropped rgb image of target object only
mask_crop = selected_mask[up:down, left:right] # mask part of cropped picture

# -------------------------------------------------------------------------------------
# step 6 -> inpaint the original image to remove the object
# -------------------------------------------------------------------------------------
inpaint_mask = (selected_mask * 255).astype(np.uint8)
inpainted_image = cv2.inpaint(image_rgb, inpaint_mask, 3, cv2.INPAINT_TELEA)

# -------------------------------------------------------------------------------------
# step 7 -> calculate new location to move the object
# -------------------------------------------------------------------------------------
offset = 100
paste_h, paste_w = crop.shape[:2]
image_h, image_w = image_rgb.shape[:2]

new_top = up
new_left = left

if direction == "left":
    new_left = max(0, left - offset)
elif direction == "right":
    new_left = min(image_w - (right - left), left + offset)
elif direction == "up":
    new_top = max(0, up - offset)
elif direction == "down":
    new_top = min(image_h - (down - up), up + offset)
elif direction == "center":
    new_top = (image_h - paste_h) // 2
    new_left = (image_w - paste_w) // 2

# -------------------------------------------------------------------------------------
# step 8 -> paste the object on the new location
# -------------------------------------------------------------------------------------
relocated_image = inpainted_image.copy()

if new_top + paste_h > image_h or new_left + paste_w > image_w:
    print("exceeds image bounds.")
else:
    for channel in range(3): # loop through rbg channels (0, 1, 2)
        target_area = relocated_image[new_top:new_top+paste_h, new_left:new_left+paste_w, channel] # get the part of the big image where to paste
        crop_channel = crop[..., channel] # get the channel from the crop image
        pasted = np.where(mask_crop, crop_channel, target_area) # if mask is True, use crop pixel else, keep the original
        relocated_image[new_top:new_top+paste_h, new_left:new_left+paste_w, channel] = pasted # replace the area in the big image

# -------------------------------------------------------------------------------------
# step 9 -> final visualization
# -------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(25, 6))

# original image with box
axes[0].imshow(image_rgb)
rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="red", facecolor="none")
axes[0].add_patch(rect)
axes[0].scatter(x_center, y_center, color='yellow', s=100)
axes[0].text(x1, y1 - 10, f"{box_label}", color="red", fontsize=12, weight='bold')
axes[0].set_title("original with detr box")
axes[0].axis("off")

# best sam mask
axes[1].imshow(image_rgb)
axes[1].imshow(selected_mask, alpha=0.5, cmap='viridis')
axes[1].scatter(x_center, y_center, color='red', s=40)
axes[1].set_title("best sam mask")
axes[1].axis("off")

# cropped cutout
axes[2].imshow(crop)
axes[2].set_title("cropped object")
axes[2].axis("off")

# inpainted image
axes[3].imshow(inpainted_image)
axes[3].set_title("object removed (inpainted)")
axes[3].axis("off")

# final image
axes[4].imshow(relocated_image)
axes[4].set_title(f"object moved {direction}")
axes[4].axis("off")

plt.tight_layout()
plt.show()