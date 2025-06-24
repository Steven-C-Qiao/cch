from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Set up Chrome options to auto-download
download_dir = "/scratches/kyuban/cq244/datasets/4DDress"
options = webdriver.ChromeOptions()
prefs = {"download.default_directory": download_dir}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=options)
driver.get("https://4d-dress.ait.ethz.ch/download.php?dt=def5020023ea042271796ddbcc69e26d17737bac94e834269fa4ce1247b5acfe070036f87f3f38a792fc4df7a9fbdc1db4d81187a48b1e21ad63c2f3fff36fe5a86fb3eea33044591eadbf358132777b87da31dc6df95c6000a243d194a863002448544c0007058192218b5fcc9eba820d0f&dir=/4D-DRESS")

# Wait for page to load
time.sleep(5)

# Find all download links (adjust selector as needed)
links = driver.find_elements(By.XPATH, "//a[contains(@href, 'tar.gz')]")

def wait_for_downloads(download_dir, expected_count, timeout=10800):
    start_time = time.time()
    while True:
        files = [f for f in os.listdir(download_dir) if f.endswith('.tar.gz')]
        crdownloads = [f for f in os.listdir(download_dir) if f.endswith('.crdownload')]
        if len(files) >= expected_count and len(crdownloads) == 0:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Download timed out.")
        time.sleep(2)

downloaded = 0
total = len(links)
while downloaded < total:
    link = links[downloaded]
    link.click()
    time.sleep(5)  # Give browser time to start download
    wait_for_downloads(download_dir, downloaded + 1, timeout=10800)
    downloaded += 1
    print(f"Downloaded {downloaded}/{total}")

driver.quit()