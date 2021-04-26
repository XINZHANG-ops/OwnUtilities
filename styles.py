from PIL import Image, ImageDraw, ImageFont
from math import ceil
# specify how large the canvas needs to be
# size in pixels.
# horizontal size is fixed at 430,
# vertical size can vary based on how long the whole thing ends up being
multiplier = 5
canvas_size = (430 * multiplier, 7545)
canvas_background = (255, 255, 255)
dpi = 50 * multiplier
file_dpi = 200 * multiplier

# vertical spaces
space1rem = 16 * multiplier
space0p12rem = 0.25 * space1rem / 2
space0p25rem = 0.25 * space1rem
space0p5rem = 0.5 * space1rem
space1p5rem = 1.5 * space1rem
space2p5rem = 2.5 * space1rem

# ColoursH1
charcoal = '#333333'
grey = '#666666'
stone = '#BBBBBB'
green = '#4CB564'
forest = '#276234'
honeydew = '#F7F9F8'
purple = '#673494'
lilac = '#FBF5FF'

# H1
H1_size = (430 * multiplier, 30 * multiplier)
H1_text_offset = (5 * multiplier, 8 * multiplier)
H1_text_gap = 10 * multiplier
H1_colour = forest
H1_font_bold = ImageFont.truetype('Arial Bold.ttf', 14 * multiplier)
H1_font_regular = ImageFont.truetype('Arial.ttf', 14 * multiplier)
H1_font_space = ImageFont.truetype('Arial.ttf', 6 * multiplier)
H1_background = honeydew

# H2
H2_colour = charcoal
H2_colour_green = green
H2_colour_purple = purple
H2_font_regular = ImageFont.truetype('Arial.ttf', 18 * multiplier)
H2_font_bold = ImageFont.truetype('Arial Bold.ttf', 18 * multiplier)

# Subtitle
Sub_colour = grey
Sub_font = ImageFont.truetype('Arial.ttf', 14 * multiplier)

# Project Timeline Chart
BM1_canvas_size = (430 * multiplier, 1590)
BM1_duration_bar_colour = purple
BM1_duration_plot_size = (42, 21)
BM1_legend_diamond_size = (12 * multiplier, 12 * multiplier)

# Pricing Criteria
BM2_canvas_size = (430 * multiplier, 1974)
plot_size1 = (400 * multiplier, 130 * multiplier)
plot_size2 = (268 * multiplier, 130 * multiplier)
pricing_weight_calculation_colour = green
word_cloud_calculation_colour = purple
shape_colour = stone
line_length = (430 / 3) * multiplier
space_adjust_caption = (plot_size2[1] - 50 * multiplier)

# Vendor Pipeline
BM3_rect_size = (72 * multiplier, 36 * multiplier)
BM3_dash_size = (42 * multiplier, 28 * multiplier)
BM3_rect_bg = lilac
BM3_chart_height = 4 * BM3_rect_size[
    1] + 3 * (BM3_dash_size[1] - ceil(BM3_rect_size[1] / 2) + 1) + 1
BM3_chart_size = (430 * multiplier, BM3_chart_height * multiplier)
BM3_canvas_size = (430 * multiplier, 1595)
BM3_corner_radius = 8 * multiplier
BM3_dash_width = 6 * multiplier
BM3_dash_gap = 4 * multiplier

# Requested Information
BM4_square_size = (35 * multiplier, 35 * multiplier)
BM4_square_bg = lilac
BM4_canvas_size = (430 * multiplier, 1594)
BM4_corner_radius = 8 * multiplier
calculation_colour_purple = purple
calculation_font_bold = ImageFont.truetype('Arial Bold.ttf', 14 * multiplier)
calculation_font_bold_percent = ImageFont.truetype('Arial Bold.ttf', 10 * multiplier)
calculation_font_regular = ImageFont.truetype('Arial.ttf', 14 * multiplier)
space_adj_box = -0.7 * multiplier
space_adj_box_text_width = 15 * multiplier
space_adj_box_text_width_percent = 7 * multiplier
space_adj_box_text_height = 10 * multiplier
space_adj_text = 30 * multiplier
space_adj_text_percent = 45 * multiplier
space_adjust_dotted_line = space0p5rem
space_adjust_percent_vertical = 3 * multiplier
height_adj_after_bold = 1 * multiplier

# General text around charts
caption_colour = grey
body_colour = charcoal
body_font = ImageFont.truetype('Arial.ttf', 12 * multiplier)
body_font_bold = ImageFont.truetype('Arial Bold.ttf', 12 * multiplier)
caption_font_italic = ImageFont.truetype('Arial Italic.ttf', 12 * multiplier)
