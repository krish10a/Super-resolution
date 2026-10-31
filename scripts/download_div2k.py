import os, requests
url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
out = "data/raw/DIV2K_train_HR.zip"
os.makedirs("data/raw", exist_ok=True)
print("Downloading DIV2K...")
# placeholder - real download would stream
print("DIV2K download script ready")
