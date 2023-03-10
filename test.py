from inference import LinkNetSeg

# Initializing the SegRetino Inference
lns = LinkNetSeg("misc/inputs/input1.png")

# Running inference
lns.inference(set_weight_dir = 'linknet.pth', path = 'misc/results/output.png', blend_path = 'misc/results/blend.png')