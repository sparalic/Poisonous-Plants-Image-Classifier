from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
import os
import json
import requests
import base64 
from PIL import Image as PILImage


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

app = Starlette()

plant_images_path = Path("/tmp")
classes = ['Common Corn-Cockle', 
              'Daffodil', 
              'Deadly Nightshade', 
              'Oleander', 'Poison Ivy', 
              'Poison Oak', 'Poke berry', 
              'Spotted spurge', 
              'Water hemlock', 
              'White hellebore']

plant_data = ImageDataBunch.single_from_classes(
             plant_images_path,
             classes,
             tfms=get_transforms(),
             size=224).normalize(imagenet_stats)

poisonous_plants = create_cnn(plant_data, models.resnet34)
poisonous_plants.model.load_state_dict(
    torch.load("Final_poisonous_plants_model.pth", map_location="cpu")
)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = poisonous_plants.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(poisonous_plants.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    img_data = encode(img)
    return HTMLResponse(
        """
        <html>
           <body>
             <p>Prediction: <b>%s</b></p>
             <p>Confidence: %s</p>
           </body>
        <figure class="figure">
          <img src="data:image/png;base64, %s" class="figure-img img-thumbnail input-image">
        </figure>
        </html>
    """ %(pred_class.upper(), pred_probs, img_data))


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h1>Classification of America's top 10 poisonous plants</h1>
        <p> This classifier can classify amongst these 10 poisonous plants
            <ul>
                <li>Common Corn-Cockle</li>
                <li>Daffodil</li>
                <li>Deadly Nightshade</li>
                <li>Oleander</li>
                <li>Poison Ivy</li>
                <li>Poke berry</li>
                <li>Spotted spurge</li>
                <li>Water hemlock</li>
                <li>White hellebore</li>
            </ul>
        </p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host="0.0.0.0", port=port)