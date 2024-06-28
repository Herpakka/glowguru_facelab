import glob
import matplotlib.pyplot as plt

src0 = "train/dry/*.*"
folder = "coutput\\*"

# Dictionaries to store contour values for each class
contour_data = {
    "dry": [],
    "Female Faces": [],
    "Male Faces": [],
    "normal": [],
    "oily": []
}

def summary(src):
    name = src.replace("\\", "/")
    name_parts = name.split("/")
    class_name = name_parts[1]
    nameN = name_parts[2].split("_")
    contours = int(nameN[0])
    if contours < 2000 :
        contour_data[class_name].append(contours)

# Process the files
for i in glob.glob(folder):
    for j in glob.glob(i + "\\*.*"):
        summary(j)

# Plotting the graphs using matplotlib
fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

class_names = ["dry", "Female Faces", "Male Faces", "normal", "oily"]

for idx, class_name in enumerate(class_names):
    axes[idx].plot(contour_data[class_name])
    axes[idx].set_title(f'{class_name} Contour Summary')
    axes[idx].set_ylabel('Contours')
    axes[idx].grid(True)

axes[-1].set_xlabel('Image Index')

plt.tight_layout()
plt.show()
