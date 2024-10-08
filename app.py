import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from streamlit_drawable_canvas import st_canvas
import numpy as np

# Load the Qwen2-VL model and processor 
device = torch.device('mps' if torch.has_mps else 'cpu')
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
).to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Function to perform OCR and solve the math problem using Qwen2-VL
def solve_math_problem(image):
    try:
        # Prepare the input format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Solve the math problem in the image and provide only the answer."},
                ],
            }
        ]
        
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=[text_input], padding=True, return_tensors="pt").to(device)

        # Generate text using the model
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Handle the response using the assistant marker
        assistant_marker = "assistant\n"
        if assistant_marker in extracted_text:
            # Split the output at the assistant marker and extract the part after it
            assistant_response = extracted_text.split(assistant_marker)[-1].strip()
        else:
            # If no marker is found, return the full response as fallback
            assistant_response = extracted_text.strip()

        return assistant_response
    except Exception as e:
        return f"Error occurred during OCR: {str(e)}"

# Streamlit App
def main():
    st.title("Math Problem Solver Using Qwen2-VL")
    st.write("Draw a math equation, and the model will solve it for you.")

    # Create a canvas component for drawing math equations
    canvas_result = st_canvas(
        stroke_width=2,
        stroke_color="#000000",
        background_color="#ffffff",
        update_streamlit=True,
        height=800,
        width=800,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Solve Math Problem"):
        if canvas_result.image_data is not None:
            # Convert the canvas image to a PIL image
            image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("RGB")
            
            # Solve the math problem using the model
            solution = solve_math_problem(image)
            
            # Display the solution
            st.subheader("Solution:")
            st.text_area("Solution", solution, height=100)
        else:
            st.warning("Please draw a math equation first.")

if __name__ == "__main__":
    main()
