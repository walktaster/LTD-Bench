# LTD-Bench

## RUN
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

Setup & installation

```bash
pip install -r requirements.txt
```

Run model inference

```bash
sh run.sh
```

Run GPT-4.1 evaluation

```bash
sh run_eval.sh
```
