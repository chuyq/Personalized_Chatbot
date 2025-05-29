"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import re
# from video_llama.common.config import Config
# from video_llama.common.dist_utils import get_rank
# from video_llama.common.registry import registry
# from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')
from tqdm import tqdm
import json
#%%
# imports modules for registration
# from video_llama.datasets.builders import *
# from video_llama.models import *
# from video_llama.processors import *
# from video_llama.runners import *
# from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")

    parser.add_argument("--split_index", type=int, required=False, default=None)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    # seed = config.run_cfg.seed + get_rank()
    seed = config.run_cfg.seed 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
    
import sys
sys.argv += cmd.split()

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state):
    if chat_state is not None:
        chat_state['dialog'] = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please choose your scnarios first', interactive=True),gr.update(value="Start Chat", interactive=True), chat_state

def upload_video(video_path, text_input, chat_state,chatbot):
    if video_path is None and text_input is None:
        return None, chat_state, None
    elif video_path is not None and text_input is not None:
        print()
        # video_html = "<video src='/file={}' style='width: 140px; max-width:none; max-height:none' preload='auto' controls></video>".format(video_path)
        chatbot = chatbot + [((video_path,), None)]
        return chat_state,chatbot

def generate_audio(text):
    # Process text input to generate speech
    inputs_audio = processor(text_target=text, return_tensors="pt")
    speech = model_speech.generate_speech(inputs_audio["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save to a temporary file and return path (Gradio can handle files)
    file_path = "./face_generate/demo/video_processed/obama/obama.wav"
    # file_path = "./face_generate/demo/video_processed/W015_neu_1_002/W015_neu_1_002.wav"
    sf.write(file_path, speech.numpy(), samplerate=16000)
    print("wav saved ...")

def generate_face(root_wav):
    name = "deepprompt_eam3d_all_final_313"
    eat = EAT(root_wav=root_wav)
    emo = "neu"
    face_output = './face_generate/demo/output/'
    eat.test(f'./face_generate/ckpt/{name}.pth.tar', emo, save_dir= face_output)
    mp4_files = []
    try:
        # Check if the directory exists and is accessible
        if os.path.exists(face_output) and os.path.isdir(face_output):
            # List all files in the directory
            for file in os.listdir(face_output):
                if file.endswith(".mp4"):
                    mp4_files.append(os.path.join(face_output, file))
    except Exception as e:
        print(f"An error occurred while listing MP4 files: {str(e)}")

    print(mp4_files)

    # Optionally return the list of MP4 file paths
    return mp4_files


def save_history(history):
    save_name = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    if len(history['dialog']) > 0:
        with open(os.path.join('output_dir', save_name + '.json'), 'w') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


    
def gradio_ask(user_input, chatbot, chat_state):
    if not user_input.strip():
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    
    if chat_state is None:
        chat_state = {'dialog': []}

    # chatbot = chatbot + [[user_input, None]]

    normalized_input = _norm(user_input)
    chat_state['dialog'].append({
        'text': normalized_input,  # Assuming _norm is normalization function you might include
        'speaker': 'usr',
        'emotion': 'depression'
    })

    chat_state['dialog'].append({ # dummy tgt
        'text': 'n/a',
        'speaker': 'sys',
        'strategy':'Open question',
        'emotion':'neutral'
    })

    # Prepare the data for the model
    inputs = inputter.convert_data_to_inputs(chat_state, toker, **dataloader_kwargs)
    inputs = inputs[-1:]
    features = inputter.convert_inputs_to_features(inputs, toker, **dataloader_kwargs)
    batch = inputter.prepare_infer_batch(features, toker, interact=True)
    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    batch.update(generation_kwargs)

    # Remove batch_size from batch if exists
    batch.pop("batch_size", None)

    # Generate response from the model
    _, generations = model.generate(**batch)
    out = generations[0].tolist()
    out = cut_seq_to_eos(out, eos)
    response = toker.decode(out).encode('ascii', 'ignore').decode('ascii').strip()
    print(response)

        chat_state['dialog'].pop()
    # Update history with the response
    chat_state['dialog'].append({
        'text': response,
        'speaker': 'sys',
        'strategy': 'Restatement',
        'emotion': 'neutral'
    })



title = """
<style>
    .title {
        font-size: 32px; 
        font-weight: bold; 
        color: #333; 
        text-align: right; 
        margin: 10px 0;
    }
</style>
<div class='title'>Multimodel MBTI-Aware Emotional Support Chatbot</div>
"""

descriptions = {
    "INTJ": "**Architect (INTJ)**: Analytical, strategic, and independent thinkers who enjoy planning and implementing innovative solutions.",
    "INTP": "**Logician (INTP)**: Curious, abstract thinkers who enjoy exploring ideas and theoretical possibilities.",
    "ENTJ": "**Commander (ENTJ)**: Assertive leaders, strategic, and love taking charge to achieve goals.",
    "ENTP": "**Debater (ENTP)**: Enthusiastic, creative, and enjoy mental sparring and challenging ideas.",
    "INFJ": "**Advocate (INFJ)**: Insightful, empathetic, and driven by strong values and a desire to help others.",
    "INFP": "**Mediator (INFP)**: Idealistic, empathetic, and guided by their core values and beliefs.",
    "ENFJ": "**Protagonist (ENFJ)**: Charismatic, altruistic, and natural leaders who inspire others.",
    "ENFP": "**Campaigner (ENFP)**: Enthusiastic, imaginative, and enjoy exploring possibilities and new ideas.",
    "ISTJ": "**Logistician (ISTJ)**: Practical, detail-oriented, and value tradition and loyalty.",
    "ISFJ": "**Defender (ISFJ)**: Warm, responsible, and dedicated to supporting others.",
    "ESTJ": "**Executive (ESTJ)**: Organized, pragmatic, and natural administrators who enforce order.",
    "ESFJ": "**Consul (ESFJ)**: Friendly, conscientious, and enjoy nurturing and supporting others.",
    "ISTP": "**Virtuoso (ISTP)**: Bold, practical, and enjoy hands-on activities and troubleshooting.",
    "ISFP": "**Adventurer (ISFP)**: Flexible, charming, and enjoy exploring aesthetics and experiences.",
    "ESTP": "**Entrepreneur (ESTP)**: Energetic, action-oriented, and love living on the edge.",
    "ESFP": "**Entertainer (ESFP)**: Spontaneous, outgoing, and enjoy being the center of attention."
}

# List of MBTI types
mbti_list = list(descriptions.keys())
demo_images = {
    "Diplomats": "/mnt/storage/chat_bot/images/mbti1.PNG",
    "Sentinels": "/mnt/storage/chat_bot/images/mbti2.PNG",
    "Explorers": "/mnt/storage/chat_bot/images/mbti3.PNG",
    "Analysts": "/mnt/storage/chat_bot/images/mbti4.PNG"
}

def get_description(mbti_type):
    """
    Return the description for the selected MBTI type.
    """
    return descriptions.get(mbti_type, "Please select your personality type.")

def get_image_html(selection):
    img_path = demo_images.get(selection, "")
    # Return an <img> tag with fixed dimensions
    return f"<img src='{img_path}' width='400' height='300' />"

with gr.Blocks(theme='snehilsanyal/scikit-learn') as demo:
    gr.Markdown(title)
 
    with gr.Row():
        # Left column: MBTI intro and selection
        with gr.Column(scale=0.5):
            video_player = gr.Video(label="Upload your video", interactive=True)
            image_selector = gr.Radio(choices=list(demo_images.keys()), label="Personality Type", value=list(demo_images.keys())[0])
            intro_image = gr.Image(value=demo_images[image_selector.value], label="MBTI", interactive=False)
            image_selector.change(fn=lambda x: demo_images[x], inputs=image_selector, outputs=intro_image)

            
            mbti_dropdown = gr.Dropdown(choices=mbti_list, label="Select Your personality Type")
            output_md = gr.Markdown("Personality Type Traits")
            mbti_dropdown.change(fn=get_description, inputs=[mbti_dropdown], outputs=[output_md])
            # video_player = gr.Video(label="Upload your video", interactive=True)

        # Right column: video, scenario, chat UI
        with gr.Column():
            chat_state = gr.State()
            video_list = gr.State()
            chatbot = gr.Chatbot(label='chatbot',scale=5)
            text_input = gr.Textbox(label='User', placeholder='Start your chat at here.', interactive=True)
            with gr.Row():
                upload_button = gr.Button(value="Start Chat", interactive=True, variant="primary")
                clear = gr.Button("Restart")

    upload_button.click(gradio_ask, inputs=[text_input, chatbot, chat_state],
        outputs=[text_input,chatbot, chat_state])
    
    text_input.submit(gradio_ask, inputs=[text_input, chatbot, chat_state],
        outputs=[text_input,chatbot, chat_state])

    clear.click(
        gradio_reset,
        inputs=[chat_state],
        outputs= [chatbot, video_player, mbti_dropdown, text_input, upload_button, chat_state],
        queue=False
    )


demo.launch(share=False, enable_queue=True)

