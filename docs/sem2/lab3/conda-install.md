1. Установить CUDA 12.1 (именно эту версию!)
2. Установить Miniconda
3. Открыть консоль Anaconda Prompt и принять условия обслуживания (Terms of Service)
    ```
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 
    ```

4. Создать окружение с именем ml-labs и версией Python 3.10 
   - Средствами PyCharm
   - Командой `conda create -n ml-labs python=3.10` в Anaconda Prompt

5. Активировать окружение и установить пакеты
   ```
   conda activate ml-labs
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install pandas scikit-learn matplotlib tqdm pyarrow
   ```