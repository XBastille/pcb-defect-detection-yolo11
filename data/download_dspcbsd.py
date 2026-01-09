import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("ROBOFLOW_API_KEY")

def download():
    if not API_KEY:
        raise ValueError(
            "ROBOFLOW_API_KEY not found!\n"
            "Create a .env file with: ROBOFLOW_API_KEY=your_key\n"
            "Get your key from: https://app.roboflow.com/settings/api"
        )
    
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("pcb-egrla").project("dspcbsd")
    version = project.version(1)
    print("Downloading DsPCBSD+ dataset...")
    dataset = version.download("yolov11", location=os.path.join(DATA_DIR, "DsPCBSD"))
    print(f"Dataset downloaded to {DATA_DIR}/DsPCBSD")


if __name__ == "__main__":
    download()
