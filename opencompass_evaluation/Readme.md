# Opencompass Tool

- Website: https://opencompass.org.cn/home

- Datasets: https://hub.opencompass.org.cn/home

- Tutorial: https://github.com/open-compass/opencompass/tree/main/docs/en/get_started

# Install Opencompass

Run `install_opencompass.sh` to create environment and install packages. Besides, datasets are downloaded through it.

If you restart the VM and it loses the environment but retain the packages. You can just run `env_opencompass.sh`.

# Run Evaluation
Drag `configs.sh` into opencompass file after you install it into your folder. Run `configs.sh` inside opencompass folder and the results will be automatically saved into `./output` folder. For more details, see: https://github.com/open-compass/opencompass/tree/main/docs/en/get_started.
 

# Debugging
You may notice that there are some missing values in the results. That is caused by erros during running. To check more info, you should dig into the `./output` folder and the`./log` folder under it could help you find out what is wrong. 