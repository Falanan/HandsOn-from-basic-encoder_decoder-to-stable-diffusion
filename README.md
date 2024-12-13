# HandsOn-from-basic-encoder_decoder-to-stable-diffusion

In this document, I'm going to expand step by step from the basic Encoder-Decoder architecture to today's very popular image generation networks: Stable Diffusion.

The trajectory of learning should be from easy to hard, from shallow to deep. That's why I've divided this notebook into multiple sections. I think the math is the toughest part when it comes to learning the various models. Math is too abstract and not like code that you can read and execute to clearly show what you have done at each step. So I'll be interspersing architectural diagrams and mathematical implementations of the models with coffee breaks where I execute the code and wait for the results, which should make it a little more motivating for people like me who don't like math to learn it.

## 1. Introduction: What is Diffusion Model?

Generative Model As the name suggests, this model generates a photo that doesn't exist in reality, a photo that looks like it came from the void. But after you generate this picture, this picture comes into the world. It's kind of like quantum mechanics, where the quantum is in a quantum superposition state when it's not being observed, and when it's being observed it collapses into the state it was in when it was observed.

A bit too far off topic, the Diffusion model isn't actually that amazing. In fact, the model does some very simple things when reasoning. 1. receives the last completed picture and the progress of the reasoning. 2. adds noise to the picture. 3. predicts the noise for the current stage. 4. throws the de-noised picture and the current progress back at itself. The following picture is from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

<img src=images/encoder-decoder/Diffusion_Model_Architechure.jpg  width="100%">

As we can see from this picture, the Diffusion model starts with a picture that is completely full of noise, and through a series of operations, ends up with less and less noise until it outputs a complete picture.

Math is always boring, but let's look at the basics of the Diffusion Model first. The following picture is from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) paper. It looks very CREAZY! Don't panic, Let's see what are they doing.

<img src= images/encoder-decoder/T-S.png>

### Training

1. We are using a loop to add different amount of noize to your image
2. x0 means the original image
3. We are going to add the noise to the original image based on different timestep(T). For example, (1-100), We will start by adding only 1% noise all the way up to adding 100% noise. 100% noise means you will lose all your original data in your image.
4. Sample a Gaussian distribution noise(e).
5. What does (x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \,\epsilon) do? This function will do a weigited sum between original image(x0) and noise(e). By doing this, you can get a image with noise added.

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
$$

6. Keep doing the loop thll the end of the loop.

<img src=images/encoder-decoder/x0t.png >

Image from [Hung-Yi Lee: Machine Learning 2023 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)

### Sampling (Inferencing)

1. Sample a Gaussian Distribution noise. This image is complete noised, at least from our human eyes, we can not tell if there is any objct.
2. Run T iterations in total.
3. The e in this equation means the predicted noise, xt means the image input from the previous step. In the equation, we are going to minus the predicted noise from the previous image. After that, as you can see ,we need to add z(New noise) to the image. Why do we need to add a new noise? We already predict the noise, isn't it great that we're predicting step by step like this? Actually NOT!!! In generative AI, we not always select the best result. If we keep choosing the best result, then the model will do some repetitive and useless work. Just like in LLM, if we keep choosing the token with the highest chance, the output of said LLM looks good though. But in reality it will just keep repeating the same point with nonsense. It's like getting stuck in a local optimal solution. So at each execution, we have to add some more bias. The new noise we added into the image is the bias.
4. Keep looping till the end.
5. Return the result

<img src =images/encoder-decoder/S_I.png >

Image from [Hung-Yi Lee: Machine Learning 2023 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)

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

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

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

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
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

The process of denoising is actually to completely predict the noise in the image and add a bias. here for the sake of demonstration, in the process of denoising, I will only remove a portion of the noise, so that a small portion of the noise remains inside the image. In this case, there is still a small amount of noise left in the image, which is equivalent to adding biased noise.

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
updated_amount = torch.where(amount >= 0.2, amount - 0.2, amount)
denoised_img = denoise(noised_img, updated_amount, noise)

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
        plt.title("Denoised Image with amout" + str(round(updated_amount[i-16].item(), 2)), rotation=5)
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

Well, Since the images in our dataset is 32 * 32. So we need to modify the network parameters to make it capable for our dataset.

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

Get some images from the dataset and add noise to it, then through the noised data into the network that we just trained. Let's see the result!

```
x, y = next(iter(train_dataloader))
x = x[:8]  # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
amount = torch.linspace(0, 1, x.shape[0])  # Left to right -> more corruption
# amount = torch.full((x.shape[0],), 0.9)
noised_x = add_noise(x, amount)

# Get the model predictions
with torch.no_grad():
    preds = net(noised_x.to(device)).detach().cpu()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title("Input data")
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap="Greys")
axs[1].set_title("Corrupted data")
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap="Greys")
axs[2].set_title("Network Predictions")
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys")
```

<img src=images/encoder-decoder/MNIST_BasicUnet.png >

Yeeeee! We successfully recover the images to human readable level except the last image. Since we add 100% noise to the last image, that is the reason we are seeing the recovered image seems nothing like the original image.

But, we are remove the noise entirely just in one step. Don't forget that the Diffusion model actually adds another bias after removing the noise at each step. here we change the inference steps a little bit. At each step of the inference process, only a small portion of the noise is subtracted, leaving a portion of the noise in there for the model to make the next inference. Let's see what the difference is.

```
n_steps = 5
amount = amount = torch.full((x.shape[0],), 0.9)
noised_x = add_noise(x, amount)
step_history = [noised_x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():  # No need to track gradients during inference
        pred = net(noised_x.to(device))  # Predict the denoised x0
    pred_output_history.append(pred.detach().cpu())  # Store model output for plotting
    mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
    noised_x = noised_x * (1 - mix_factor) + pred.cpu() * mix_factor  # Move part of the way there
    step_history.append(noised_x.detach().cpu())  # Store step for plotting

fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
axs[0, 0].set_title("x (model input)")
axs[0, 1].set_title("model prediction")
for i in range(n_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap="Greys")
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap="Greys")
```

<img src=images/encoder-decoder/MNIST_Basic_UNet_5steps.png >

As we can see in the picture. When there is a lot of noise, the result of multi-step denoising will be clearer than the single denoised image. At the same time, we lose some of the information of the image, which makes our results not look as accurate as single-step denoising. But that's what generative AI wants!

### Scale up to Diffusion model scheduler(DDPM UNet2DModel)

```
model = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
# print(model)
```

As we increase the size of the network, training would takes more time, it is almost doubled.

```
# Dataloader (you can mess with batch size)
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# How many runs through the data should we do?
n_epochs = 3

# Create the network
net = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)  # <<<
net.to(device)

# Our loss finction
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
        pred = net(noisy_x, 0).sample  # <<< Using timestep 0 always, adding .sample

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

# Plot the Loss
plt.plot(losses)
plt.ylim(0, 0.1)
plt.title("Loss over time")
```

Now Let's see the result

<img src=images/encoder-decoder/MNIST_Unet2D.png >

It seems much better than our BasicUNet, except the one with 100% noise added. We can have a way better result.

Now Let's do the same thing. Remove noise partially in each step:

```
n_steps = 50  # Increased to have more steps for 10-step history saving
amount = torch.full((x.shape[0],), 0.8)
noised_x = add_noise(x, amount)
step_history = [noised_x.detach().cpu()]
pred_output_history = []

for i in range(n_steps):
    with torch.no_grad():  # No need to track gradients during inference
        pred = net(noised_x.to(device), 0)  # Predict the denoised x0

    mix_factor = 1 / (n_steps - i)  # How much we move towards the prediction
    noised_x = noised_x * (1 - mix_factor) + pred.sample.cpu() * mix_factor  # Move part of the way there

    # Store history every 10 steps (including the last step)
    if i % 10 == 0 or i == n_steps - 1:
        step_history.append(noised_x.detach().cpu())  # Store step for plotting
        pred_output_history.append(pred.sample.cpu())  # Store model output for plotting
pred_output_history.append(pred.sample.cpu())
# Update the visualization to match the new history recording logic
n_rows = len(step_history)  # Number of rows corresponds to how many times we saved history
fig, axs = plt.subplots(n_rows, 2, figsize=(9, 4), sharex=True)
axs[0, 0].set_title("x (model input)")
axs[0, 1].set_title("model prediction")

for i in range(n_rows):
    axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap="Greys")
    axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap="Greys")
```

<img src=images/encoder-decoder/MNIST_Unet2D_steps.png >

It looks like the same as our defined BasicUnet. We lose some of the details, but the results are still pretty closed and we also can see there are While we can still discern the number, a new style has been created. This is what generative AI is all about.

Finally, If we go back to the math equation:

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
$$

If we plot the coefficients of the original image and noise.

```
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large")
```

<img src=images/encoder-decoder/Unet2D_amount.png >

We can clearly see that as the number of training steps increases, there is less and less information in the original photo, as well as more and more noise added into the photo.

## From Diffusion to Guidance Diffusion

#### Learning is very difficult, but cats save the day.

### Use pretrained diffusion model

Let's use pretrained cat generation diffusion model to demostrate this part.

```
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
image_pipe.to(device)
```

Lets use pretrained cat generation Diffusion Model to generate 4 cat images.

```
cat_image_list = []
for i in range(0, 4):
    images = image_pipe().images
    cat_image_list.append(images[0])

n_images = len(cat_image_list)

# Create a figure with 1 row and `n_images` columns
fig, axs = plt.subplots(1, n_images, figsize=(20, 5))  # Width 20, Height 5

# Loop through the images and plot them in one row
for idx, image in enumerate(cat_image_list):
    axs[idx].imshow(image)  # Display the image
    axs[idx].axis('off')    # Remove axis for cleaner display

# Display the plots
plt.tight_layout()
plt.show()
```

Result:
<img src=images/Diff2GuidDiff/1000steps.png >

It seems the first 2 images are fine. Expecially the first image, we can see this model generated a very good cat image. But the last 2 images seems not soo good, anyway, we can still see there is a cat in the image.

The default inference steps of this model is 1000. It takes around 40s to generate one image on 2080Ti. Since we can adjust the inference steps, I really want to save my precious time. Let's try 400 inference steps.

<img src=images\Diff2GuidDiff\400steps.png >

These images look still fine, but the quality compired with 1000 inference steps definitely worser. Because that is the feature of DDPM pipeline. DDPMPipeline use UNet to denoise images, and requires 1000 steps to generate high quality images.

Now Lets see what does Scheduler do during the generation process. Here we will replace the original scheduler (DDPMScheduler) to DDIMScheduler.

DDIMScheduler is a more efficient scheduler used in diffusion networks. It can use a significantly reduced number of inference steps to generate images of comparable quality. Furthermore, the intermediate steps, inputs, and outputs can be obtained, which provides valuable insights into the network's behavior.

Here is the paper: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

```
# Replace the default scheduler with DDIMScheduler (if you want DDIM behavior)
scheduler = DDIMScheduler.from_pretrained("google/ddpm-cat-256")
num_inference_steps = 400
scheduler.set_timesteps(num_inference_steps=num_inference_steps)
image_pipe.scheduler = scheduler

# Random starting point (batch of 4 images, 3 channels, 256x256 resolution)
x = torch.randn(4, 3, 256, 256).to(device)  # Batch of 4, 3-channel 256x256 px images

# Loop through the sampling timesteps
for i, t in tqdm(enumerate(image_pipe.scheduler.timesteps)):
    # Prepare model input
    model_input = image_pipe.scheduler.scale_model_input(x, t)
    # Get the prediction from the UNet (predict the noise)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    # Calculate what the updated sample should look like using the scheduler
    scheduler_output = image_pipe.scheduler.step(noise_pred, t, x)
    # Update x for the next iteration
    x = scheduler_output.prev_sample
    # Occasionally display the intermediate results
    if i % 10 == 0 or i == len(image_pipe.scheduler.timesteps) - 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Display the current `x` (partially denoised image)
        grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
        axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)  # Clip and normalize image for display
        axs[0].set_title(f"Current x (step {i})")
        # Display the predicted "clean" image (denoised image)
        pred_x0 = scheduler_output.pred_original_sample  # Original prediction from the scheduler
        if pred_x0 is not None:  # Not all schedulers support pred_original_sample
            grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
            axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
            axs[1].set_title(f"Predicted denoised images (step {i})")
        else:
            axs[1].set_title(f"Predicted denoised images (unavailable)")
        plt.show()
```

Well, let's set the inference steps to 400 and observe the results during the inference steps.

0 step(Initial step):
<img src=images/Diff2GuidDiff/intermediate_0.png >

100 steps:
<img src=images/Diff2GuidDiff/intermediate_100.png >

200 steps:
<img src=images/Diff2GuidDiff/intermediate_200.png >

300 steps:
<img src=images/Diff2GuidDiff/intermediate_300.png >

400 steps:
<img src=images/Diff2GuidDiff/intermediate_400.png >

The last image shows how adding random noise to an image affects the result of the inference. The 4th image in 100 steps shows a big cat face, but as the inference steps continue, the big cat face turns into the cat body.

Now Lets switch back to default scheduler(DDPMScheduler) and try to fine tuning the network.

```
scheduler = DDIMScheduler.from_pretrained("google/ddpm-cat-256")
scheduler.set_timesteps(num_inference_steps=1000)
image_pipe.scheduler = scheduler
```

I'm really excited to see if this network can be tuned to generate some dog images! Why not just give it a try? Here I'll fine-tune the cat generation diffusion model on [Stanford Dogs dataset](https://huggingface.co/datasets/Voxel51/StanfordDogs).

Prepare the dataset:

```
# Load the entire Stanford Dogs dataset (both train and test splits)
dataset_name = "voxel51/stanford_dogs"
dataset = load_dataset(dataset_name)

# Combine train and test splits into one dataset if needed
train_dataset = dataset['train']

print(f"Total number of images in the Stanford Dogs dataset: {len(train_dataset)}")

# Define image processing parameters
image_size = 256  # Size to resize images
batch_size = 4  # Number of images in each batch

# Define transformations to apply to each image
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Normalize for 3 channels (RGB)
    ]
)

# Transformation function to be applied to each example in the dataset
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

# Set transformation on the combined dataset
train_dataset.set_transform(transform)

# Create a DataLoader to load batches of images
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Display a batch of images
print("Previewing batch:")
batch = next(iter(train_dataloader))
grid = torchvision.utils.make_grid(batch["images"], nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)  # Convert back from normalized
plt.axis('off')
plt.show()
```

Some dog images of this dataset:

<img src= images/Diff2GuidDiff/StanfordDogDataset.png>

Now Lets start fine tuning!!!

```
num_epochs = 5  # @param
lr = 5e-7  # 2param
weight_decay = 1e-5
grad_accumulation_steps = 4  # @param

optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr, weight_decay=weight_decay)

losses = []

for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            image_pipe.scheduler.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction for the noise
        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        # Compare the prediction with the actual noise:
        loss = F.mse_loss(
            noise_pred, noise
        )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

        # Store for later plotting
        losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward(loss)

        # Gradient accumulation:
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):])/len(train_dataloader)}")

# Plot the loss curve:
plt.plot(losses)
```

This is really a large dataset! It took me 4 hours 40 minutes to finish fine-tuning on a single 2080Ti.

Let's generate 8 images and see the results:

```
x = torch.randn(8, 3, 256, 256).to(device)  # Batch of 8
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
```

<img src=images/Diff2GuidDiff/DogFineTonCatMpng.png >

We succeed. The second picture shows a cat face on a dog body. The cat in picture 4 sits like a pug. This is really funny. Since I'm using a very small learning rate, maybe next time I should try a larger leaning rate.

Now, save our own model!

```
image_pipe.save_pretrained("DogFT_on_Cat_Network.pth")
```

### Give the network some Guidance

We had fun making a model to generate dog images from cat images generation model. Now, reload the model and give it some guidance.

```
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")
image_pipe.to(device)
```

Firstly, try a very simple guidance, try to change the color style of the images.

```
def color_loss(images, target_color=(0.0, 0.0, 0.0)):
    """Given a target color (R, G, B) return a loss for how far away on average
    the images' pixels are from that color. Defaults to a light teal: (0.1, 0.9, 0.5)"""
    target = torch.tensor(target_color).to(images.device) * 2 - 1  # Map target color to (-1, 1)
    target = target[None, :, None, None]  # Get shape right to work with the images (b, c, h, w)
    error = torch.abs(images - target).mean()  # Mean absolute difference between the image pixels and the target color
    return error
```

Inference method:

```
scheduler.set_timesteps(num_inference_steps=40)
guidance_loss_scale = 40
x = torch.randn(4, 3, 256, 256).to(device)

for i, t in tqdm(enumerate(scheduler.timesteps)):

    # Set requires_grad before the model forward pass
    x = x.detach().requires_grad_()
    model_input = scheduler.scale_model_input(x, t)

    # predict (with grad this time)
    noise_pred = image_pipe.unet(model_input, t)["sample"]

    # Get the predicted x0:
    x0 = scheduler.step(noise_pred, t, x).pred_original_sample

    # Calculate loss
    loss = color_loss(x0, target_color=(0.0, 0.0, 0.0)) * guidance_loss_scale
    if i % 10 == 0:
        print(i, "loss:", loss.item())

    # Get gradient
    cond_grad = -torch.autograd.grad(loss, x)[0]

    # Modify x based on this gradient
    x = x.detach() + cond_grad

    # Now step with scheduler
    x = scheduler.step(noise_pred, t, x).prev_sample


grid = torchvision.utils.make_grid(x, nrow=4)
im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
# Image.fromarray(np.array(im * 255).astype(np.uint8))
plt.imshow(im)
plt.axis('off')  # Remove axis for a cleaner look
plt.show()
```

#### Let's look at the code and see what happens in the guidance stage.

In function ```def color_loss(images, target_color=(0.0, 0.0, 0.0))```

1. This function receives 2 parameters. ```images``` is the images tensor in ```[Batch_size, C, H, W]``` order, then ```target_color``` is the desiered color in ```[R, G, B]``` order.
2. Mapping the target color to the range (-1, 1). We map the color range to (-1, 1) because the diffusion model uses this range. This is common in UNet models. We use this range in the diffusion model because the center point is 0. This makes gradient mode effective and stable. The calculation method as following:

$$\text{target} = \text{target\_color} \times 2 - 1$$

3. Reshaping the target to match the shape of input image tensor for futher loss calculation.
4. Loss calculation. This step calculates the absolute difference between each pixel and the desired color. Then use the absolute mean as the global loss.

#### Inference steps

1. To demostrate the result, here I set the inference steps to 400 to make us can see the result faster since this is the color guidance.
2. Set the guidance scale. This number represent the scale that how by much does ```color_loss``` effect the image generation.
3. Set requires_grad ```x = x.detach().requires_grad_()```. This step means we need to calculate the gradient later respect to color loss.
4. Use UNet in the diffusion model and predict the noise.
5. Use the noise predicted by UNet to generate the denoised image.
6. Calculate the loss based on the guidance scale.
7. Compute and reverse the gradient. Since we want to move towards the target, so we want to minimize the loss, it is very samilar to gradian descent.
8. Update ```x``` towards to the desiered direction.
9. Step forward.

By doing these steps, we can have a image in generally black. But we still can somehow see there is a cat im some of the results.

<img src=images/Diff2GuidDiff/Black_Imgs.png >

This image is toooooooo dark, now lets downgrade the guidance scale. ```guidance_loss_scale = 30```

<img src=images/Diff2GuidDiff/Black_Imgs_30_Scale.png >

These images generated look better. There's a white cat on a black background, and a black cat too.

### Guidance using CLIP

import CLIP model from OpenAI pretrained.

```
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.to(device)

# Transforms to resize and augment an image + normalize to match CLIP's training data
tfms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),  # Random CROP each time
        torchvision.transforms.RandomAffine(5),  # One possible random augmentation: skews the image
        torchvision.transforms.RandomHorizontalFlip(),  # You can add additional augmentations if you like
        torchvision.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


# And define a loss function that takes an image, embeds it and compares with
# the text features of the prompt
def clip_loss(image, text_features):
    image_features = clip_model.encode_image(tfms(image))  # Note: applies the above transforms
    input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
    embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
    dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)  # Squared Great Circle Distance
    return dists.mean()
```

How does the loss being calculated?

1. Encode the image to image embedding representation for CLIP model.
2. Encode the given texts.
3. Normalization. Make both image embedding and text embedding to unit vector.
4. Calculate the geodesic distance (Great-Circle Distance).

What is geodesic distance (Great-Circle Distance)?

DCD is the shortest distance between two points on the suface of a sephere. After embedding and normalization step, the data points are considered distributing on the surface of a high-dimentional sephere. So when calculating the loss, the curved geometry of the space need to be considered in. The following foemula can gives the GCD distance.

$$
d(\mathbf{x}, \mathbf{y}) = \arccos\left( \mathbf{x} \cdot \mathbf{y} \right)
$$

If you see the implementation, we are using `arcsin` instead of `arccos`, this is because when two data points are very close, then the result of `argcos` becomes very unstable and leading to a very large gradient, so we are using `arcsin` on the chordal distance avoids this. And the following equations are mathematically equivalent.

$$
\arccos(x) = 2 \cdot \arcsin\left(\sqrt{\frac{1 - x}{2}}\right)
$$


Inference steps:

```
# x0 = scheduler.step(noise_pred, t, x).pred_original_sample
# This method not implemented on MPS devices on PyTorch 2.4.1, So only cpu available on MacBook
if device == "mps":
    device = "cpu"

prompt = "A cat on the grass"

guidance_scale = 10
n_cuts = 4

# More steps -> more time for the guidance to have an effect
scheduler.set_timesteps(1000)

# We embed a prompt with CLIP as our target
text = open_clip.xtokenize([prompt]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text)


x = torch.randn(4, 3, 256, 256).to(device)  # RAM usage is high, you may want only 1 image at a time

for i, t in tqdm(enumerate(scheduler.timesteps)):

    model_input = scheduler.scale_model_input(x, t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]

    cond_grad = 0

    for cut in range(n_cuts):

        # Set requires grad on x
        x = x.detach().requires_grad_()

        # Get the predicted x0:
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample

        # Calculate loss
        loss = clip_loss(x0, text_features) * guidance_scale

        # Get gradient (scale by n_cuts since we want the average)
        cond_grad -= torch.autograd.grad(loss, x)[0] / n_cuts

    if i % 100 == 0:
        print("Step:", i, ", Guidance loss:", loss.item())

    # Modify x based on this gradient
    alpha_bar = scheduler.alphas_cumprod[i]
    x = x.detach() + cond_grad * alpha_bar.sqrt()  # Note the additional scaling factor here!

    # Now step with scheduler
    x = scheduler.step(noise_pred, t, x).prev_sample


grid = torchvision.utils.make_grid(x.detach(), nrow=4)
im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
# Image.fromarray(np.array(im * 255).astype(np.uint8))
plt.imshow(im)
plt.axis('off')  # Remove axis for a cleaner look
plt.show()
```

Here are human understandable texts for inference steps:

1. Define parameters: ```prompt, guidance_scale, n_cuts```. ```Prompt``` is the text that how do we want to guide the diffusion model to generate. ```Guidance_scale``` is by how much does the images looks like the expected output. ```n_cuts``` can guide how does CLIP model cut the image. I'll go into this parameter later.
2. Get the noise predicted by UNet model.
3. CLIP Guidance:
   * Get `x0`, which are the clean images predicted by using the current predicted noise.
   * Calculate CLIP loss
   * Compute and update the gradient to make the image generate moves towards the direction that we want.
4. Make the scheduler takes another step in the reverse diffusion process, further removing noise from the image.

<img src=images/Diff2GuidDiff/CatOnGrass.png >
Since here I'm using a baby model, I believe the results are quite good.

And the following image is about the predection to the content in the image from OpenAI, further reading: [CLIP: Connecting text and images](https://openai.com/index/clip/)

<img src=images/Diff2GuidDiff/CLIP_Prediction.png >

## Classifier Based Guidance v.s. Classifier Free Guidance. Pros, Cons & Challenges

I believe that by this point we understand the Diffusion model, so it's time to talk about the Classifier Based Guidance and Classifier Free Guidance.

### What are the differences between these two?
Classifier Free Diffusion Models don't need an external model, which is classifier to steer the diffusion model's output toward a certain class. All above examples are Classifier free diffusion model. As you can see, the pretrained cat generation model can generate cat images without any guidance. This is because Classifier free model already involves incorporating conditioning information (like class labels) directly into the diffusion model itself so that it can generate images of a specific class without needing a separate classifier.

During training, the model is exposed to pairs of (input, condition). For example, in a text-to-image scenario, the condition is a text prompt. In a class-conditional scenario, the condition would be a class label.

Some fraction of the time, the condition is intentionally replaced with a “null” or empty condition (like an empty string for text or a special null token for a class label). This trains the model to also produce outputs without any conditioning.

Text-to-Image Stable diffusion is a very typical Classifier Free model.