import os

folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
image_path = os.path.abspath(os.path.join(folder_path, "images/output.gif"))
print(image_path)