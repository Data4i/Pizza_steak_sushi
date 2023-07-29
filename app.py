import gradio as gr
import gradio.components as grc

from fastai.vision.all import *


learn = load_learner('steak_pizza_sushi.pkl')

categories = ('Pizza', 'Steak', 'Sushi')

def classify_images(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float, probs)))

image = grc.Image(shape = (128, 128))
label = grc.Label()
examples = ['pizza2.jpeg', 'steak.jpeg', 'sushi.jpeg']

intf = gr.Interface(fn = classify_images, inputs = image, outputs = label, examples = examples)
intf.launch(inline = False)