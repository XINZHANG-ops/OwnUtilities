import os
import zipfile
from tqdm import tqdm


def zip_file(path_to_file, directory_to_zip_to):
    zipf = zipfile.ZipFile(directory_to_zip_to, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path_to_file):
        for file in tqdm(files):
            zipf.write(os.path.join(root, file))
    zipf.close()


def unzip_file(path_to_zip_file, directory_to_extract_to, delete_zip_after_unzipping=False):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        # zip_ref.extractall(directory_to_extract_to)
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extract(member, directory_to_extract_to)
            except zipfile.error as e:
                pass
    if delete_zip_after_unzipping:
        os.remove(path_to_zip_file)
        print("File Removed!")


#unzip('FasttextModels.zip',os.getcwd())
#zip('FasttextModels','FasttextModels.zip')
