"""Okabe-Ito color palette (https://siegal.bio.nyu.edu/color-palette/)

USAGE: in desired script, import like this:

from colorpalette import clist, cdict

Then you can directly call clist or cdict. Example:

import matplotlib.pyplot as plt
plt.plot(x, y, color = clist[1])
plt.plot(x, y, color = cdict["gold"])

If you use a color palette like in seaborn, you can also set the entire palette:

import seaborn as sns
palette = sns.color_palette(clist, 8) #8=number of colors
sns.set_palette(palette)"""

cdict = dict(black = "#000000",
         gold = "#E69F00",
         blue_light = "#56B4E9",
         green = "#009E73",
         yellow = "#F0E442",
         blue_dark = "#0072B2",
         red = "#D55E00",
         magenta = "#CC79A7")

clist = ["#000000",
          "#E69F00",
          "#56B4E9",
          "#009E73",
          "#F0E442",
          "#0072B2",
          "#D55E00",
          "#CC79A7"]

