# HandsOn-from-basic-encoder_decoder-to-stable-diffusion

In this document, I'm going to expand step by step from the basic Encoder-Decoder architecture to today's very popular image generation networks: Stable Diffusion.

The trajectory of learning should be from easy to hard, from shallow to deep. That's why I've divided this notebook into multiple sections. I think the math is the toughest part when it comes to learning the various models. Math is too abstract and not like code that you can read and execute to clearly show what you have done at each step. So I'll be interspersing architectural diagrams and mathematical implementations of the models with coffee breaks where I execute the code and wait for the results, which should make it a little more motivating for people like me who don't like math to learn it.

## 1. Introduction: What is Diffusion Model?

Generative Model As the name suggests, this model generates a photo that doesn't exist in reality, a photo that looks like it came from the void. But after you generate this picture, this picture comes into the world. It's kind of like quantum mechanics, where the quantum is in a quantum superposition state when it's not being observed, and when it's being observed it collapses into the state it was in when it was observed.

A bit too far off topic, the Diffusion model isn't actually that amazing. In fact, the model does some very simple things when reasoning. 1. receives the last completed picture and the progress of the reasoning. 2. adds noise to the picture. 3. predicts the noise for the current stage. 4. throws the de-noised picture and the current progress back at itself. The following picture is from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

<img src=images/encoder-decoder/Diffusion_Model_Architechure.jpg  width="100%">

Does this sound familiar? It seems like this is the architecture of the encoder and decoder! Still, they are slightly different. Let's start with the encoder-decoder architecture and expand to the Diffusion Model, and finally to the now very popular model: Stable Diffusion.

## 2. Encoder-Decoder architechure for learning noise

First, to vividly demonstrate the process of adding noise and predicting noise in Diffusion Network.

### Let's start by taking a picture of ourselves.

```
import time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()
time.sleep(2)
ret, frame = cap.read()
if ret:
    # Save the photo to a file without showing a window
    photo_path = "captured_photo.jpg"
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved as {photo_path}")
else:
    print("Error: Could not capture a frame.")
cap.release()
```

### Using YOLO model to detect and crop the person in the picture

It's a bit like hitting a mosquito with a cannon, but YOLO model is the most popular model to do objects detection. So, let's play with it.

```
from ultralytics import YOLO
# Load a pretrained YOLO model (recommended for training)
yolo_model = YOLO("yolov8n.pt")
# Path to your image
image_path = 'captured_photo.jpg'
# Perform person detection
results = yolo_model(image_path)

image_orig = cv2.imread(image_path)
# Convert the image from BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
# Display and save results
for result in results:
    # Boxes contain coordinates for detected objects
    boxes = result.boxes
    for box in boxes:
        if box.cls == 0:  # Class ID for 'person' in COCO dataset
            print(f"Detected a person at {box.xyxy.cpu().numpy()}")  # Bounding box coordinates
            bbox = box.xyxy[0].cpu().numpy()
            bbox = [int(coord) for coord in bbox]
            cropped_image = image_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Display the cropped image using matplotlib
            plt.imshow(cropped_image)
            plt.axis('off')  # Hide axis
            plt.title("Cropped Person")
            plt.show()
            
            # Save the cropped image if needed
            cropped_image_path = "cropped_person.jpg"
            cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(cropped_image_path, cropped_image_bgr)
            print(f"Cropped image saved at {cropped_image_path}")

```
### Let's see how does the picture looks like
<img src=images/encoder-decoder/cropped_person.jpg  width="55%">

As we already know, the processes of Diffusion model inference are adding noise and pridict how does the noise looks like for the next step. Then, how does the noize looks like? And, how can we add the noise to the image?

### Add Noise
If we read the [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) paper. Then, you will see the following equation:
$$ q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$
It looks CRAZY! But don't be panic. If we see this equation combined with this image:
<img src=images/encoder-decoder/Diffusion_Model_Architechure.jpg  width="100%">
Does it looks better? This equation basically saying we need to add different amout noise based on different time.

For better understanding, let's start with a simpler version:
```
def add_noise(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Adjust shape for batch processing
    noised_img = x * (1 - amount) + noise * amount
    return noised_img, noise
```

This is the math equation representation:
$$x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
$$
In this function, we firstly sample a Gaussian Distribution noise, then add the noise to the original image based on different amount.

### Add some noise to the image
```
image_batch = torch.tensor([image_rgb for i in range(0,8)])
image_batch = torch.tensor(image_batch, dtype=torch.float32) / 255.0

amount = torch.linspace(0, 1, image_batch.shape[0])  # Left to right -> more corruption

noised_img, noise = add_noise(image_batch, amount)

plt.figure(figsize=(30, 7))
for i in range(0, 16):
    plt.subplot(2, 8, i+1)
    if i < 8:
        plt.imshow(image_batch[i])
        plt.title("Original image")
        plt.axis('off')
    else:
        plt.imshow(noised_img[i-8])
        plt.title("Noized Image with amout" + str(round(amount[i-8].item(), 2)), rotation=5)
        plt.axis('off')

plt.show()

```
Original image v.s. Noised image
<img src=images/encoder-decoder/OrigvsNoised.png>
As we can see from the picture, the noise gets progressively louder from left to right until you can't see what's in the picture at all.

## Denoise
Actually, the denoising example here isn't very robust, it's just for presentation purposes. Because the Diffusion model predicts noise, the output is a completely noise-removed image. But for ease of understanding at this stage, I'm going to go straight to the noise we generated here and remove only a portion of the noise at each step. The rest of the noise is actually equivalent to us taking the denoised photo and adding noise based on the state of this STEP.
```
def denoise(corrupted_x, amount, noise):
    """Reconstruct the original input `x` from the corrupted version"""
    amount = amount.view(-1, 1, 1, 1)
    original_x = (corrupted_x - noise * amount) / (1 - amount)
    return original_x
```

This function directly removes the noise we added directly.

### Remove the noise partially
```
denoised_img = denoise(noised_img, amount, noise)

plt.figure(figsize=(30, 10))
for i in range(0, 24):
    plt.subplot(3, 8, i+1)
    if i < 8:
        plt.imshow(image_batch[i])
        plt.title("Original image")
        plt.axis('off')
    elif i >=8 and i < 16:
        plt.imshow(noised_img[i-8])
        plt.title("Noized Image with amout" + str(round(amount[i-8].item(), 2)), rotation=5)
        plt.axis('off')
    else:
        plt.imshow(denoised_img[i-16])
        plt.title("Denoised Image with amout" + str(round(amount[i-16].item(), 2)), rotation=5)
        plt.axis('off')
```
Original v.s. Noised v.s. Partially noise removed
<img src=images/encoder-decoder/OriginalvsNoisedvsPartially.png >
As we can see from the figure, if the drop noise is added completely when adding noise, which means ```amount = 1```. Then the photo will completely lose its original character. It would be impossible for us to restore the photo by simply subtracting the noise.

### Now Scale up the basic add_noise-denoise network to basic encoder-decoder network(UNet)

As we already touched MNIST and CIFAR10 dataset. I'll use these two dataset as examples.

The MNIST (Modified National Institute of Standards and Technology) dataset is a large database of handwritten digits that is commonly used for training various image processing systems and machine learning models.

#### Download the dataset and plot the images
```
dataset = torchvision.datasets.MNIST(
    root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

x, y = next(iter(train_dataloader))
print("Input shape:", x.shape)
print("Labels:", y)

images = x.squeeze(1)  # Shape becomes [8, 28, 28]
# Plot each image
batch_size = images.shape[0]
for i in range(batch_size):
    plt.subplot(1, batch_size, i + 1)
    plt.imshow(images[i].cpu().numpy(), cmap="Greys")  # Convert to numpy and specify grayscale
    plt.axis('off')  # Optional: Turn off axis for cleaner display
# plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
```
These are the images in MNIST dataset looks like:
<img src=images/encoder-decoder/MNIST_imgs.png>

Since from now we do not need to menorize the noises added to the image, then update the add_noise a little bit.

Updated add_noise function:

```
def add_noise(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount
```

Now Lets see the original images v.s. noised images

```
# Plotting the input data
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title("Input data")
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")

# Adding noise
amount = torch.linspace(0, 1, x.shape[0])  # Left to right -> more corruption
noised_x = add_noise(x, amount)

# Plotting the noised version
axs[1].set_title("Noised data (-- amount increases -->)")
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap="Greys")
```
<img src=images/encoder-decoder/MNIST_OrigvsNoised.png >

### UNet
UNet(U-Shaped Network) is a type of convolutional neural network (CNN) designed for image-to-image tasks.
1. Encoder Path: Extracts and compresses information, similar to a traditional convolutional network.
2. Decoder Path: Reconstructs the image back to its original resolution, often using upsampling techniques.
3. The "skip connections" directly link encoder layers to decoder layers. They ensure that fine details from earlier layers are available at higher resolution during upsampling.

This is how does the UNet looks like: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
<img src=images\encoder-decoder\Unet_Structure.png >

Well, Since the images in our dataset is 32 * 32. So we need to modify the network parameters to make it capeable*************** for our dataset.

```
class CustomizedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(128, 128, kernel_size=5, padding=2),
                nn.Conv2d(128 + 128, 64, kernel_size=5, padding=2),  # Note: Add channel from skip connection
                nn.Conv2d(64 + 64, 32, kernel_size=5, padding=2),    # Add channel from skip connection
                nn.Conv2d(32 + 32, out_channels, kernel_size=5, padding=2),  # Add channel from skip connection
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        
        # Use Upsample with align_corners=True
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        h = []  # Store skip connections
        
        # Down layers
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Apply convolution + activation
            if i < len(self.down_layers) - 1:  # For all but the last down layer
                h.append(x)  # Store output for skip connection
                x = self.downscale(x)  # Downscale for the next layer

        # Up layers
        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upsample
                # Ensure sizes are aligned for concatenation
                if x.shape[2:] != h[-1].shape[2:]:
                    x = torch.nn.functional.interpolate(x, size=h[-1].shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, h.pop()], dim=1)  # Concatenate skip connection
            x = self.act(l(x))  # Apply convolution + activation

        return x
```

Now Let's train this network!
```
net = CustomizedUNet()
x = torch.rand(8, 1, 28, 28)

# Dataloader (you can mess with batch size)
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# How many runs through the data should we do?
n_epochs = 4

# Create the network
net = CustomizedUNet()
net.to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):

    for x, y in train_dataloader:

        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x = add_noise(x, noise_amount)  # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x)

        # Calculate the loss
        loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

# View the loss curve
plt.plot(losses)
plt.ylim(0, 0.1)
```

