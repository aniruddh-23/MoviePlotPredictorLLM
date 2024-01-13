from transformers import AutoTokenizer,AutoModelForCausalLM
import streamlit as st

@st.cache_resource
class AppModel:

    def __init__(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained("./output/model/checkpoint-6500")

    def generate_plot(self, prompt:str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95
            )

        outputs = self.tokenizer.batch_decode(outputs)[0]
        return outputs


model = AppModel()

prompt = st.text_area("Enter the begenning of the Plot . . . ")
clicked = st.button("Generate the plot!")

if clicked:
    output = model.generate_plot(prompt=prompt)

    chat_message = st.chat_message("ai")
    chat_message.markdown(output)