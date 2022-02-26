# scala-arm-ibc
This project is for reproducing the result of ibc (https://github.com/google-research/ibc) in my hoby scala-arm robot.

# Setup
```bash
git clone git@github.com:syundo0730/scala-arm-ibc.git --recursive
```

# Workflow
### collect oracle
```bash
./configs/collect_oracle.sh
```

### train
```bash
./configs/scala_arm_pixels/train_pixel_ebm.sh
```

### execution
```bash
./configs/rum_mlp_ebm.sh
```
