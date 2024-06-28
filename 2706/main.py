from DuplicateRemover import DuplicateRemover
import glob

dirname = "train/Female Faces"

# Remove Duplicates
dr = DuplicateRemover(dirname)
dr.find_duplicates()
