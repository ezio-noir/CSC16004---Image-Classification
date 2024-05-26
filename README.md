- Install depedencies (running python with a virtural environment is recommended):
  ```bash
  pip install -r requirements.txt
  ```
- Show available options:
  ```bash
  python3 train.py --help
  ```
  or
  ```bash
  python3 transfer.py --help
  ```
- Clean up and train/continue training on a pretrained model:
  ```bash
  ./clean_up.sh && python3 train.py [options]
  ```
  or
  ```bash
  ./clean_up.sh && python3 transfer.py [options]
  ```
