import os
import shutil

def copy_and_delete_sorted(
    original_folder,
    copied_folder,
    delete_percentage=30,
    sort_by="name"
):
    if os.path.exists(copied_folder):
        print(f"❗ Folder '{copied_folder}' already exists. Skipping to avoid overwrite.")
        return

    shutil.copytree(original_folder, copied_folder)
    print(f"✅ Copied '{original_folder}' → '{copied_folder}'")

    all_files = [f for f in os.listdir(copied_folder) if os.path.isfile(os.path.join(copied_folder, f))]

    if sort_by == "name":
        all_files.sort()
    elif sort_by == "date":
        all_files.sort(key=lambda f: os.path.getmtime(os.path.join(copied_folder, f)))
    else:
        print(f"Invalid sort method: {sort_by}")
        return

    total_files = len(all_files)
    num_to_delete = int((delete_percentage / 100) * total_files)
    if num_to_delete == 0:
        print("Nothing to delete — percent too low or folder empty.")
        return

    files_to_delete = all_files[:num_to_delete]

    for f in files_to_delete:
        os.remove(os.path.join(copied_folder, f))

    print(f"🗑️ Deleted {num_to_delete} of {total_files} files ({delete_percentage}%) in '{copied_folder}'\n")

for i in range(10,100,10):
    # Calculates the percentage that needs to be removed
    Percentage_To_Remove = 100 - i

    # Training Data
    Original_Folder = r"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data\train\depth"
    Copied_Folder = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data_{i}%\train\depth"
    copy_and_delete_sorted(original_folder= Original_Folder, copied_folder= Copied_Folder, delete_percentage= Percentage_To_Remove)

    Original_Folder = r"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data\train\rgb"
    Copied_Folder = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data_{i}%\train\rgb"
    copy_and_delete_sorted(original_folder= Original_Folder, copied_folder= Copied_Folder, delete_percentage= Percentage_To_Remove)

    # Validation Data
    Original_Folder = r"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data\val\depth"
    Copied_Folder = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data_{i}%\val\depth"
    copy_and_delete_sorted(original_folder= Original_Folder, copied_folder= Copied_Folder, delete_percentage= Percentage_To_Remove)

    Original_Folder = r"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data\val\rgb"
    Copied_Folder = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\Data_{i}%\val\rgb"
    copy_and_delete_sorted(original_folder= Original_Folder, copied_folder= Copied_Folder, delete_percentage= Percentage_To_Remove)

    print("One cycle is done.")
