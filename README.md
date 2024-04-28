# LOL-Robot-Detector

The LOL-Robot-Detector is a tool designed to identify and analyze cheating behavior patterns within the online video game "League of Legends". Utilizing machine learning models and anomaly detection techniques, this project aims to enhance the integrity of gameplay by distinguishing between normal and cheating players.

## Installation

1. Install PyTorch using the link below <https://pytorch.org/get-started/locally/>. Remember to download the version that suits your operating system and environment.

2. Install the required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Update Log

### Apr 13, 2024

The pre-trained model: `mouse_movement_anomaly_detection_model.pth`

## TODO

- [ ] Demo
- [ ] Trasnformer
- [ ] ...

## Usage

- **Tip**: If you are using Windows System, please use `python` instead of `python3`.

### Test

- Run the following command in the Command Line with a `python3` command (for Ubuntu).

    ```bash
    python3 ./new/eval.py
    ```

### Train

- Optional 1: Use our dataset
  - The processed mouse positions is ready in the `ready_for_training` folder.
  - Run `train.py` in the Command Line with a `python3` command (for Ubuntu).

    ```bash
    python3 ./new/train.py
    ```

  - The trained model is saved as `mouse_movement_anomaly_detection_model.pth`.
  - Since we have already uploaded the pre-trained model named `mouse_movement_anomaly_detection_model.pth`, you can change the name you want based on your needs from the file `./new/train.py` line $51$.

- Optional 2: Use your own dataset

  - **Data Preparation**: Use `cursurDetector.py` to read the mouse positions of your raw videos. The file will be saved in the `./mouse_positions` folder.
  - **Scaler Preparation**: Use `dataModifier.py` to extract the features of your raw mouse positions and use 'universal_scaler' to standrize them.
  - **Model Training**: Use `./new/train.py` to train your own model.
  - **Tip**: Make sure you are consistently using 1080p, 30fps videos.

### Validate

- Run the following command in the Command Line with a `python3` command (for Ubuntu).

    ```bash
    python3 ./new/validate.py
    ```

    | Validate metric | DataLoader 0           |
    | --------------- | ---------------------- |
    | val_loss        | 0.00014062559057492763 |

## Contacts

E-mail: <solistoriashenny@gmail.com>
QQ: 3480547309

## License

This project is licensed under the MIT License - see the LICENSE file for details.
