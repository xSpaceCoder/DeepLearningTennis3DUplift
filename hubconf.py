dependencies = ['torch', 'numpy', 'einops', 'torch.utils.tensorboard.summary', 'hparams', 'random', 'os', 'matplotlib.pyplot', 'matplotlib.backends.backend_agg']

from torchhub.interface_rallies_dynamicAnkle import UpliftingModel as UpliftingModel_r_dynamicAnkle

IMAGES_ZIP_URL = "https://mediastore.rz.uni-augsburg.de/get/51XbRH38ZY/"
IMAGES_ZIP_FILENAME = "example_images.zip"

def tennis_uplifting(input_type="rallies", mode='dynamicAnkle', **kwargs):
    """
    Loads the Tennis Uplifting Model.
    Available iput_types: 'trajectories', 'rallies'
    Available modes: 'dynamic', 'dynamicAnkle'
    """
    if input_type == "rallies":
        if mode == 'dynamicAnkle':
            return UpliftingModel_r_dynamicAnkle()
        else:
            raise ValueError(f"Invalid mode '{mode}' for input_type 'rallies'. Available modes: 'dynamic', 'dynamicAnkle'")
    """ elif input_type == "trajectories":
        if mode == 'dynamic':
            return UpliftingModel_t_dynamic()
        elif mode == 'dynamicAnkle':
            return UpliftingModel_t_dynamicAnkle()
        else:
            raise ValueError(f"Invalid mode '{mode}' for input_type 'trajectories'. Available modes: 'dynamic', 'dynamicAnkle'") 
    else: 
        raise ValueError(f"Invalid input_type '{input_type}'. Available input_types: 'rallies', 'trajectories'")"""




""" 
def download_example_images(local_folder='example_images'):
    
    Downloads the example images zip archive, extracts it to the local folder,
    and returns the folder path.
    
    # 1. Check if the folder exists and is not empty
    # We assume if the folder has files, the download was already successful.
    if os.path.exists(local_folder) and len(os.listdir(local_folder)) > 0:
        print(f"Images already present in '{local_folder}'. Skipping download.")
        return local_folder

    print(f"Images not found in '{local_folder}'. Preparing to download...")

    # Ensure the directory exists (for the zip file)
    os.makedirs(local_folder, exist_ok=True)

    # Define the local path for the zip file
    # We place the zip inside the folder or right next to it.
    # Here we put it inside to keep the root directory clean.
    zip_path = os.path.join(local_folder, IMAGES_ZIP_FILENAME)

    # 2. Download zip if it doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading images from {IMAGES_ZIP_URL}...")
        try:
            torch.hub.download_url_to_file(IMAGES_ZIP_URL, zip_path, progress=True)
        except Exception as e:
            # Clean up the partial file if download failed
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"Failed to download images: {e}")

    # 3. Extract the zip
    print("Extracting images...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We extract into local_folder.
            # Note: If your zip already contains a folder named "example_images",
            # you might end up with "example_images/example_images/".
            # If so, change this to zip_ref.extractall('.')
            zip_ref.extractall(local_folder)

        print("Extraction complete.")

        # Optional: Remove the zip file after extraction to save space
        # os.remove(zip_path)

    except Exception as e:
        raise RuntimeError(f"Failed to extract images: {e}")

    # remove zip file after extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)

    return local_folder """