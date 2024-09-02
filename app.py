import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import gc

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(
    master=app,
    height=40,
    width=512,
    font=("Arial", 20),
    text_color="black",
    fg_color="white",
)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = (
    "stabilityai/stable-diffusion-2-1"  # Use a more advanced model for better realism
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model pipeline with half-precision floats to reduce memory usage
pipe = StableDiffusionPipeline.from_pretrained(
    modelid, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def generate():
    try:
        with autocast(device):
            # Adjust the prompt to include words that guide the model towards realism
            input_prompt = (
                prompt.get()
                + ", highly detailed, photorealistic, ultra-realistic, 8k resolution"
            )

            # Generate the image with a lower guidance scale for realism
            output = pipe(input_prompt, guidance_scale=7.0)

        # Ensure the output contains images
        if "images" in output:
            image = output["images"][0]
        else:
            raise ValueError("Unexpected output format from the pipeline")

        image.save("generatedimage.png", "PNG", quality=95)  # Save with high quality
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)
        lmain.image = img  # Keep a reference to avoid garbage collection
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        if "CUDA out of memory" in str(e):
            print("Attempting to free up memory...")
            cleanup()
    except Exception as e:
        print(f"Error during generation: {e}")
    finally:
        cleanup()


trigger = ctk.CTkButton(
    master=app,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    command=generate,
)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
