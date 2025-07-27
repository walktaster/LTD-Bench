# LTD-Bench

## Setup
Before running LTD-Bench, please ensure that your Linux environment has already installed Xvfb, as it may be required for the Hard-level generation tasks. 

You can install it using the following command.
```bash
apt-get install xvfb
apt-get install ghostscript
```
or
```bash
yum install xorg-x11-server-Xvfb
yum install ghostscript
```

Then you need to run Xvfb
```bash
Xvfb :1 -screen 0 800x600x24&
```

Setup your Python environment
```bash
pip install -r requirements.txt
```

## Run
Set up the model configuration in "run.sh" file, including your model_id, API_BASE_URL and API_KEY. (We have set model_id to GPT-4o by default)
Then you can start running model inference!
```bash
sh run.sh
```

## Evaluation
Run GPT-4.1 automatic evaluation
```bash
sh run_eval.sh
```
