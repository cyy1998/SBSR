import sys
sys.path.append('.')

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from model.sketch_model import SketchModel
from model.classifier import Classifier
from model.view_model import MVCNN
from pathlib import Path
from PIL import Image

from torchvision import datasets
from dataset.view_dataset_reader import MultiViewDataSet
from torchvision import transforms
import torch.nn.functional as F

use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
print(f"Currently using device: {device}")
#np.set_printoptions(threshold=np.inf)

sketch_model = SketchModel(backbone="../hf_model/models--openai--clip-vit-base-patch32")
view_model = MVCNN(backbone="../hf_model/models--openai--clip-vit-base-patch32")
classifier = Classifier(12, 512, 50)
if use_gpu:
    sketch_model = sketch_model.to(device)
    view_model = view_model.to(device)
    classifier = classifier.to(device)

# Load model
# sketch_model.load(args.ckpt_dir+'sketch_lora')
# view_model.load(args.ckpt_dir + 'view_lora')

classifier.load_state_dict(torch.load('../hf_model/Epoch5/mlp_layer.pth'))
sketch_model.load('../hf_model/Epoch5/sketch_lora')
view_model.load('../hf_model/Epoch5/view_lora')
sketch_model.eval()
view_model.eval()
classifier.eval()

view_data = MultiViewDataSet(root="E:/3d_retrieval/Dataset/D2O/render_img/train", transform=transforms.Resize(224))
feature = torch.load("../feature.mat")
print(feature['view_labels'])
def convert_transparent_to_white(input_image):
    # 检查图片是否为RGBA模式，即带有透明度通道
    image=input_image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 分割图片的RGBA通道
    r, g, b, a = image.split()

    # 创建一个新的白色背景图片，大小与原始图片相同
    white_background = Image.new('RGB', image.size, (255, 255, 255))

    # 将原始图片的RGB通道复制到新的白色背景图片上
    white_background.paste(image, mask=a)

    # 保存转换后的图片
    return white_background

def topk_result(sketch_feature, k):
    for key in feature.keys():
        feature[key] = torch.tensor(feature[key]).to("cuda")
    cosine_similarities = F.cosine_similarity(sketch_feature, feature['view_feature'], dim=1)
    top_scores, top_indices = torch.topk(cosine_similarities, k)
    # sketch_data = datasets.ImageFolder(root="/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_sketch_test_picture", transform=transforms.Resize(224))

    res_images = []
    res_labels = []
    for i in top_indices:
        res_images.append(view_data[i][0][5])
        res_labels.append(view_data[i][1])

    return res_images, res_labels


def sbsr(input_img):
    input_img=input_img['composite']
    #input_img=np.where(input_img[:, :, 3] == 0, 255, 0)
    # use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    # if torch.cuda.is_available(): device = "cuda"
    # elif torch.backends.mps.is_available(): device = "mps"
    # else: device = "cpu"
    # print(f"Currently using device: {device}")
    #
    # # sketch_model = SketchModel(backbone="./hf_model/models--openai--clip-vit-base-patch32")
    # # view_model = MVCNN(backbone="./hf_model/models--openai--clip-vit-base-patch32")
    # sketch_model = SketchModel(backbone="openai/clip-vit-base-patch32")
    # view_model = MVCNN(backbone="openai/clip-vit-base-patch32")
    # classifier = Classifier(12, 512, 133)
    #
    # if use_gpu:
    #     sketch_model =sketch_model.to(device)
    #     view_model = view_model.to(device)
    #     classifier = classifier.to(device)
    #
    # # Load model
    # # sketch_model.load(args.ckpt_dir+'sketch_lora')
    # # view_model.load(args.ckpt_dir + 'view_lora')
    #
    # classifier.load_state_dict(torch.load(Path('../hf_model') / 'mlp_layer.pth'))
    # sketch_model.eval()
    # view_model.eval()
    # classifier.eval()

    # 对输入数据进行处理：
    pil_img = Image.fromarray(np.uint8(input_img))
    pil_img=convert_transparent_to_white(pil_img)
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为指定大小，如果不为正方形，会压缩成正方形，而不是裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])
    input = image_transforms(pil_img).unsqueeze(0).to(device)

    # print(input.shape)
    with torch.no_grad():
        output = sketch_model.forward(input)
        mu_embeddings= classifier.forward(output)
        #mu_embeddings,logits = classifier.forward(output)

    # 提前计算好，数据库中的view embedding，进行比对排名，得到最相近的model
    sketch_feature = nn.functional.normalize(mu_embeddings, dim=1)
    res_images, res_labels = topk_result(sketch_feature, 10)
    return res_images


if __name__=='__main__':
    demo = gr.Interface(fn=sbsr,
                        inputs=gr.ImageEditor(image_mode='RGBA',label='sketch',
                                              # value={
                                              #     "background": np.full((600, 800), 255, dtype=np.uint8),
                                              #     "layers": None,
                                              #     "composite": np.full((600, 800), 255, dtype=np.uint8),
                                              # },
                                              brush=gr.Brush(colors=["#000000"])),
                        outputs=gr.Gallery(label="3D model"),
                        title="3D Shape Model Retrieval using Sketch",
                        description="Upload a Sketch to find the most similar 3D Shape Models!")

    demo.launch(share=True,)



   

