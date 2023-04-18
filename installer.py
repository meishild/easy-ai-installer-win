import shutil
import sys
import subprocess
import os
import pkg_resources
from zipfile import ZipFile

try:
    import requests
except ModuleNotFoundError as error:
    required = {'requests'}
    installed = {p.key for p in pkg_resources.working_set}
    missing = required - installed
    if missing:
        print("installing requests...")
        python = sys.executable
        subprocess.check_call([python, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL)

def main():
    # 1.check python git
    if sys.platform != "win32":
        print("ERROR: This installer only works on Windows")
        quit()
    else:
        print("Running on windows...")

    version = sys.version_info
    if version.major != 3 or version.minor != 10 or version.micro < 6:
        print("ERROR: You don't have python version than 3.10.6 installed, please install python 3.10.6, and add it to path")
        quit()
    else:
        print("Python version than 3.10.6 detected...")

    try:
        subprocess.check_call(['git', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("ERROR: Git is not installed, please install git")
        quit()
    print("Git is installed... installing")

    #2. checkout or only create venv
    python_real = sys.executable
    python = r"venv\Scripts\pip.exe"
    this_path = os.path.dirname(os.path.abspath(__file__))

    reply = None
    while reply not in ("y", "n"):
        reply = input(f"Do you git cloning repositories (y/n): ").casefold()

    repo_url = None
    floder_path = ""
    if reply == 'y':
        while not repo_url:
            repo_url = input(f"set git repositories url: ")
        while not os.path.exists(floder_path):
            floder_path = input(f"set check local path: ")
        print("cloning repositories")
        os.chdir(floder_path)
        try:
            subprocess.check_call(['git', 'clone', repo_url])
        except Exception:
            print("ERROR: checkout git repositories error:%s" % repo_url)
            quit()

        floder_path = os.path.join(floder_path, os.path.basename(repo_url).split('.git')[0])
        os.chdir(floder_path)
        subprocess.check_call(['git', 'submodule', "init"])
        subprocess.check_call(['git', 'submodule', 'update'])
    else:
        floder_path = ""
        while not os.path.exists(floder_path):
            floder_path = input(f"set install virtual env path: ")

    os.chdir(floder_path)
    print("install venv in %s" % floder_path)

    print("setting execution policy to unrestricted")
    subprocess.check_call(f"{os.path.join(this_path, 'installables', 'change_execution_policy.bat')}")

    print("creating venv and installing requirements")
    subprocess.check_call([python_real, "-m", "venv", "venv"])
    
    # 3. install tourch torchvision
    t_version = None
    while t_version not in ("0", "1", "2"):
        t_version = input("which version of torch do you want to install?\n"
                      "0 = 1.12.1\n"
                      "1 = 2.0.0\n"
                      "2 = 2.1.0: ").casefold()
    if t_version == "2":
        torch_version = "torch==2.1.0.dev20230322+cu118 torchvision==0.16.0.dev20230322+cu118 --extra-index-url https://download.pytorch.org/whl/nightly/cu118"
    elif t_version == '1':
        torch_version = "torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_version = "torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
    print("installing torch")
    subprocess.check_call(f"{python} install {torch_version}".split(" "))
    
    # 3. install requirements
    print("installing requirements")
    req_type = None
    while req_type not in ("txt", "toml"):
        req_type = input("which type of requirements choice requirements.txt or pyproject.toml (txt/toml): ").casefold()
    if req_type == "toml":
        subprocess.check_call(f"{python} install -e .".split(" "))  
    else:
         subprocess.check_call(f"{python} install -r requirements.txt".split(" "))  

    # 4. install other
    reply = None
    if reply in {'1', '2'}:
        reply = None
        while reply not in ("y", "n"):
            reply = input(f"Do you want to install the triton built for torch 2 (y/n)").casefold()
        if reply == 'y':
            subprocess.check_call(f"{python} install -U -I --no-deps {os.path.join(this_path,'installables', 'triton-2.0.0-cp310-cp310-win_amd64.whl')}".split(" "))

    print("Setting up default config of accelerate")
    with open("default_config.yaml", 'w') as f:
        f.write("command_file: null\n")
        f.write("commands: null\n")
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("deepspeed_config: {}\n")
        f.write("distributed_type: 'NO'\n")
        f.write("downcase_fp16: 'NO'\n")
        f.write("dynamo_backend: 'NO'\n")
        f.write("fsdp_config: {}\n")
        f.write("gpu_ids: '0'\n")
        f.write("machine_rank: 0\n")
        f.write("main_process_ip: null\n")
        f.write("main_process_port: null\n")
        f.write("main_training_function: main\n")
        f.write("megatron_lm_config: {}\n")
        f.write("mixed_precision: fp16\n")
        f.write("num_machines: 1\n")
        f.write("num_processes: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("tpu_name: null\n")
        f.write("tpu_zone: null\n")
        f.write("use_cpu: false")
    if os.path.exists(os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'accelerate', 'default_config.yaml')):
        os.remove(os.path.join(os.environ['USERPROFILE'], '.cache', 'huggingface', 'accelerate', 'default_config.yaml'))
    shutil.move("default_config.yaml", os.path.join(os.environ['USERPROFILE'], ".cache", "huggingface", "accelerate"))
    
    # install bitsandbytes_windows
    if os.path.exists(os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes')):
        for file in os.listdir(os.path.join(this_path, "bitsandbytes_windows")):
            shutil.copy(os.path.join(this_path, "bitsandbytes_windows", file), 
                        os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes'))
        shutil.copy(os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'main.py'),
                    os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup'))

    # install xformers
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"Do you want to install xformers (y/n): ").casefold()
    
    if reply == "y":
        print("installing xformers ")
        if t_version == '0':
            xformers = "https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl"
        elif t_version == '1':
            xformers = "https://github.com/DDStorage/LoRA_Easy_Training_Scripts/releases/download/torch2.1.0/xformers-0.0.17+c36468d.d20230318-cp310-cp310-win_amd64.whl"
        else:
            xformers = "https://github.com/DDStorage/LoRA_Easy_Training_Scripts/releases/download/torch2.0.0/xformers-0.0.17+b3d75b3.d20230320-cp310-cp310-win_amd64.whl"
        subprocess.check_call(f"{python} install -U -I --no-deps {xformers}".split(' '))

    # install 30X0 and 40X0 patch
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"Do you want to install the optional cudnn patch for faster "
                      f"training on high end 30X0 and 40X0 cards? (y/n): ").casefold()

    if reply == 'y':
        cndnn_zip_path = os.path.join(this_path, "cudnn.zip")
        if not os.path.exists(cndnn_zip_path):
            print("start download cudnn.zip file.")
            r = requests.get("https://b1.thefileditch.ch/mwxKTEtelILoIbMbruuM.zip")
            with open(cndnn_zip_path, 'wb') as f:
                f.write(r.content)
            print("finish download cudnn.zip file.")

        shutil.copy(cndnn_zip_path, floder_path)
        shutil.copy(os.path.join(this_path, "cudnn.py"), floder_path)
        with ZipFile("cudnn.zip", 'r') as f:
            f.extractall(path="cudnn_windows")
        subprocess.check_call(f"{os.path.join('venv', 'Scripts', 'python.exe')} cudnn.py".split(" "))
        shutil.rmtree("cudnn_windows")
        os.remove("cudnn.zip")
        os.remove("cudnn.py")
    else:
        reply = None
        while reply not in ('y', 'n'):
            reply = input("Are you using a 10X0 series card? (y/n): ")
        if reply:
            shutil.copy(os.path.join(this_path, "installables", "libbitsandbytes_cudaall.dll"),
                        os.path.join("venv", 'Lib', 'site-packages', 'bitsandbytes'))
            os.remove(os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup', 'main.py'))
            shutil.copy(os.path.join(this_path, 'installables', 'main.py'),
                        os.path.join('venv', 'Lib', 'site-packages', 'bitsandbytes', 'cuda_setup'))

if __name__ == "__main__":
    main()
