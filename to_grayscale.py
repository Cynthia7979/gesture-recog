from PIL import Image, ImageOps
import sys, os

path = sys.argv[1]
if path[-3:].lower() in ('jpg', 'png', 'peg'):
    print('Processing', path)
    filename = os.path.basename(path)
    save_path = os.path.dirname(path).replace('data', 'grayscale_dataset')

    im = Image.open(path)
    gray_im = ImageOps.grayscale(im)
    gray_im.save(os.path.join(save_path, filename))
else:
    print('Skipping', path)
