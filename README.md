# agrxai
# Description of the project

# Installation instructions
```
git clone https://github.com/gongzir1/agrxai-exp.git
cd agrxai-exp
pip3 install -r requirements.txt
```


# Usage instructions
To run the code for a particular experiment, you can open the corresponding '_script.py' file in PyCharm and run it. The 'Defender.py' file is where the defence work is done, while the 'FL_models.py' file contains the structure of Federated Learning. The 'constant.py' file contains the settings for the constants used in the experiment. The 'output' folder contains the result files in the CSV format after running the code. 
Note: the script is optimized for CUDA, so recommended to run with GPU device.
# Understanding the output
The output is shown as follows:

|epoch|test_acc|test_loss|training_acc|trainig_loss|
|-----|--------|-----------|----------|------------|
|  0  |  0.08  |   0.22    |   0.32   |     0.12   |
|  1  |  0.01  |   0.32    |   0.39   |     0.12   |
| ... |  ...   |    ...    |    ...   |    ...     |



#References
[1] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. 2020. FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. arXiv preprint arXiv:2012.13995 (2020).

[2] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. 2020. Local model poisoning attacks to byzantine-robust federated learning. In 29th {USENIX} Security Symposium ( {USENIX } Security 20). 1605–1622.

[3] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. 2018. Byzantine-robust distributed learning: Towards optimal statistical rates. In Inter- 1246 national Conference on Machine Learning. PMLR, 5650–5659.

[4] Virat Shejwalkar and Amir Houmansadr. 2021. Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning. Internet 1223 Society (2021), 18.

[5]Liyue Shen, Yanjun Zhang, Jingwei Wang, and Guangdong Bai. Better Together: Attaining the Triad of Byzantine-robust Federated Learning via Local Update Amplification. In Proceedings of the 38th Annual Computer Security Applications Conference, ACSAC ’22, pages 201–213, New York, NY, USA, 2022. Association for Computing Machinery.

# License 
# Contact us
