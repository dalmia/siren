import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from siren import SIREN

BATCH_SIZE = 8192

img_filepath = 'data/celtic_spiral_knot.jpg'
img_raw = np.array(Image.open(img_filepath))
img_ground_truth = torch.from_numpy(img_raw).float()

rows, cols, channels = img_ground_truth.shape
pixel_count = rows * cols


def build_eval_tensors():
    img_mask_x = np.arange(0, rows)
    img_mask_y = np.arange(0, cols)

    img_mask_x, img_mask_y = np.meshgrid(img_mask_x, img_mask_y, indexing='ij')
    img_mask_x = torch.from_numpy(img_mask_x)
    img_mask_y = torch.from_numpy(img_mask_y)

    img_mask_x = img_mask_x.float() / rows
    img_mask_y = img_mask_y.float() / cols

    img_mask = torch.stack([img_mask_x, img_mask_y], dim=-1)
    img_mask = img_mask.reshape(-1, 2)
    img_eval = img_ground_truth.reshape(-1, 3)

    return img_mask, img_eval


img_mask, img_eval = build_eval_tensors()

test_dataset = TensorDataset(img_mask, img_eval)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Build model
layers = [256, 256, 256, 256, 256]
in_features = 2
out_features = 3
initializer = 'siren'
w0 = 1.0
w0_initial = 30.0
c = 6
model = SIREN(
    layers, in_features, out_features, w0, w0_initial,
    initializer=initializer, c=c)

# Restore model
checkpoint_path = 'checkpoints/siren/inpainting/model'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError('Checkpoint not found at {}'.format(
        checkpoint_path))

# load checkpoint
print('Loading model checkpoint from {}'.format(checkpoint_path))
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt['network'])
model.eval()

iterator = tqdm(test_dataloader)

predictions = []

for batch in iterator:
    inputs, _ = batch

    with torch.no_grad():
        prediction = model(inputs)

    predictions.append(prediction)

predicted_image = torch.cat(predictions).cpu().numpy()
predicted_image = predicted_image.reshape((rows, cols, channels)) / 255
predicted_image = predicted_image.clip(0.0, 1.0)

img_save_path = 'images/celtic_spiral_knot.jpg'
os.makedirs(os.path.dirname(img_save_path), exist_ok=True)

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0])
plt.imshow(img_ground_truth.numpy() / 255)
plt.title("Ground Truth Image")

plt.sca(axes[1])
plt.imshow(predicted_image)
plt.title("Predicted Image")

fig.tight_layout()
plt.savefig(img_save_path, bbox_inches='tight')
plt.show()
