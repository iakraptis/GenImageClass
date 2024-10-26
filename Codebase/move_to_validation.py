import os
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()
train_actual_dir = current_path / "Dataset" / "Images" / "Train" / "Actual"
validation_actual_dir = current_path / "Dataset" / "Images" / "Validation" / "Actual"
validation_generated_dir = (
    current_path / "Dataset" / "Images" / "Validation" / "Generated"
)

if __name__ == "__main__":
    for filename in os.listdir(validation_generated_dir):
        try:
            os.rename(
                train_actual_dir / filename,
                validation_actual_dir / filename,
            )
        except FileNotFoundError:
            print(f"File {filename} not found in Train/Actual directory")
            continue
