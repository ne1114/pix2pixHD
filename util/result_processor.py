import os 
from PIL import Image, ImageChops
from numpy import concatenate

result_path = "/home/jeonghoon/Projects/pix2pixHD/results/0906Slice/test_latest"
input_path = "/home/jeonghoon/Projects/pix2pixHD/datasets/preprocessing-d20210831_vertical_slice/"

result_files = os.listdir(result_path + "/images/")

input_suffix = "_input_label.jpg"
infered_suffix = "_synthesized_image.jpg"
real_suffix = ""
boarder_img = Image.new("RGB", (30, 256), (255,255,255))

def concatenate_horizontally(img_array):
    widths, heights = zip(*(i.size for i in img_array))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    
    return new_im

for file in result_files:
    if file.endswith(input_suffix):
        doc_id = file.split('_')[0]
        inferred_img = Image.open( result_path + f"/images/{doc_id}{infered_suffix}" )
        input_img_with_gt = Image.open(input_path + f"{doc_id}/{doc_id}.png")
        w, h = input_img_with_gt.size
        w2 = int(w / 2)
        gt_img = input_img_with_gt.crop((0, 0, w2, h))
        input_img = input_img_with_gt.crop((w2, 0, w, h))
        inferred_img = inferred_img.resize((512,256))

        '''
        Concatenate Images [ input | inferred | ground_truth ]
        '''
        images = [input_img, boarder_img, inferred_img, boarder_img, gt_img]
        
        new_img = concatenate_horizontally(images)

        '''
        Create diff image 
        '''
        gt_input = ImageChops.difference(gt_img, input_img)
        infer_input = ImageChops.difference(inferred_img, input_img)
        images = [gt_input, infer_input]
        diffed_img = concatenate_horizontally(images)

        merged_img_path = result_path + f"/merged/{doc_id}.png"
        diffed_img_path = result_path + f"/diff/{doc_id}.png"

        new_img.save(merged_img_path)
        diffed_img.save(diffed_img_path)
