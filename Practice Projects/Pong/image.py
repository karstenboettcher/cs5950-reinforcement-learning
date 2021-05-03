import imageio
from pathlib import Path

image_path = Path('D:\\Homework\\CS 5950\\Pong_Rainbow\\agent_viz\\Pong\\rainbow\\images')
images = list(image_path.glob('*.png'))
image_list = []
for file_name in images:
    image_list.append(imageio.imread(file_name))

imageio.mimwrite('animated_from_images.gif', image_list, fps=24)