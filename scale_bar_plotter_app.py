import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_cropper import st_cropper
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from  PIL import Image, ImageDraw


def get_ellipse_coords(point: (int, int)):
    center = point
    radius = 1
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized



"""
# 🤏 :straight_ruler: Scale bar plotter

### This app allows you to plot a scale bar on your image!

"""

st.set_option('deprecation.showfileUploaderEncoding', False)


# Upload an image and set some options for demo purposes
uploaded_file = st.sidebar.file_uploader(label='Step1: upload a file', type=['tif', 'png', 'jpg'])
use_example_file = st.sidebar.checkbox("Use example file", value=False, help="Use in-built example file to the app")

if use_example_file:
    uploaded_file = "sample1.tif"

st.sidebar.write('Step2: measure scale bar')

if uploaded_file:
    img = Image.open(uploaded_file)
    image_array = np.array(img)
    image_resized = image_resize(image_array, height=500)
    image_resized = Image.fromarray(image_resized)

    if "points" not in st.session_state:
        st.session_state["points"] = []

    with image_resized as img:
        draw = ImageDraw.Draw(img)

        # Draw an ellipse at each coordinate in points
        for point in st.session_state["points"]:
            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="white")

        value = streamlit_image_coordinates(img, key="pil")

        if value is not None:
            point = value["x"], value["y"]

            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.experimental_rerun()

            if len(st.session_state["points"]) == 2:
                pixel_num = st.session_state['points'][1][0] - st.session_state['points'][0][0]
                st.write(st.session_state['points'][1][0] - st.session_state['points'][0][0])

img_nm = st.sidebar.number_input("original scale bar length in the image", min_value=100, max_value=1000, value=400, step=10)
scale_bar_nm = st.sidebar.number_input("scale bar length", min_value=10, max_value=300, value=100, step=10)
# pixel_num = st.sidebar.number_input("the number of pixel", min_value=10, max_value=300, value=100, step=10)
pixel_per_num = pixel_num / img_nm
scale_bar_pixel = scale_bar_nm*pixel_per_num

st.write(f"scale bar length in the image: {scale_bar_pixel}")

st.sidebar.write('Step3: crop the image')

aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if uploaded_file:
    img = Image.open(uploaded_file)
    cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF', aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((150, 150))
    st.image(cropped_img)

# st.image(cropped_img, width=700)
img_out = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2RGB)
# scale bar plot
fontprops = fm.FontProperties(size=10)
fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
ax.imshow(img_out)
scalebar = AnchoredSizeBar(ax.transData,
                           size=scale_bar_pixel,
                           label=f'{int(scale_bar_nm)} nm',
                           loc='lower right',
                           pad=0.3,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           label_top=True,
                           fontproperties=fontprops)
ax.add_artist(scalebar)
ax.set_yticks([])
ax.set_xticks([])
ax.axis('off')
plt.tight_layout()
st.pyplot(fig)

fn = 'your_image_with_scale_bar.tif'
fig.savefig(fn, dpi=100, bbox_inches='tight', pad_inches=0)
with open(fn, "rb") as img:
    btn = st.download_button(
        label="Download image",
        data=img,
        file_name=fn,
        mime="image/tiff"
    )