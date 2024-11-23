import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def simulate_attack(
    prompt,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Simulates a Blackhat AI scenario by generating attack strategies, potential impacts, and ethical countermeasures.
    """
    # Build the system message to define the simulator's behavior
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]

    # Include user and assistant message history
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Add the current user prompt
    messages.append({"role": "user", "content": prompt})

    # Initialize the response variable
    response = ""

    # Stream the AI's response from the inference API
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

# Define the Gradio ChatInterface with security-focused configuration
demo = gr.ChatInterface(
    simulate_attack,
    additional_inputs=[
        gr.Textbox(
            value=(
                "You are an AI simulator for cybersecurity training, designed to generate attack scenarios, analyze their impacts, and suggest countermeasures."
            ),
            label="System message",
        ),
        gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    title="Blackhat AI Simulator",
    description=(
        "This simulator generates adversarial scenarios, analyzes attack vectors, and provides ethical countermeasures. Use responsibly for cybersecurity training and awareness."
    ),
)

if __name__ == "__main__":
    demo.launch()
