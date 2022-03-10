import os
import shutil

original = r"/home/labs/testing/class67-colab/data_new/GOformer_data/GOformer_data/"
target = r"/home/labs/testing/class67-colab/data_new/final_data/"

print(len(os.listdir(original)))
i = 0
for folder in os.listdir(original):
    i += 1
    print(i)
    folder_path = os.path.join(original, folder)
    if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 1:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if "pre_structure" in file_path:
                new_name = folder + '_' + file
                shutil.copyfile(file_path, os.path.join(target, new_name))



    # shutil.copyfile(original, target)
# os.rename()
