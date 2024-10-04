from pathlib import Path

import numpy as np

img_size = 28 * 28


def create_digit_sequence(
    data_dir: Path,
    sequence_type: int,
    T_train: int,
    T_test: int,
    T_val: int,
    train_randomness: bool,
    test_randomness: bool,
    train_signflip: bool,
    test_signflip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img_train = read_mnist_image(data_dir / "train-images-idx3-ubyte")  # (N1, img_size)
    num_train = read_mnist_label(data_dir / "train-labels-idx1-ubyte")  # (1, N1)
    img_test = read_mnist_image(data_dir / "t10k-images-idx3-ubyte")  # (N2, img_size)
    num_test = read_mnist_label(data_dir / "t10k-labels-idx1-ubyte")  # (1, N2)

    lab_train: list[np.ndarray] = [np.where(num_train == i)[0] for i in range(10)]
    lab_test: list[np.ndarray] = [np.where(num_test == i)[0] for i in range(10)]

    if sequence_type not in [1, 2]:
        raise ValueError("Invalid sequence type")
    create_seq = create_ascending if sequence_type == 1 else create_fibonacci
    input_train, label_train = create_seq(img_train, num_train, lab_train, T_train, train_randomness, train_signflip)
    input_test, label_test = create_seq(img_test, num_test, lab_test, T_test, test_randomness, test_signflip)
    input_val, label_val = create_seq(img_train, num_train, lab_train, T_val, train_randomness, train_signflip)

    return (
        input_train,
        input_test,
        input_val,
        label_train,
        label_test,
        label_val,
    )


def read_mnist_image(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(img_size, -1, order="F")
    return data / 255


def read_mnist_label(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data


def create_ascending(
    imgs,
    nums,
    lab,
    T,
    randomness,
    signflip,
):
    count, pm = 0, 1
    input_seq = np.zeros((img_size, T))
    label_seq = np.zeros((1, T), dtype=int)

    for t in range(T):
        rnd = np.random.randint(len(lab[count]))
        input_seq[:, t] = pm * imgs[:, lab[count][rnd]]
        label_seq[:, t] = nums[lab[count][rnd]]
        count = (count + 1) % 10

        if randomness:
            if np.random.randint(50) == 0:
                count = np.random.randint(10)
                if signflip == 1:
                    pm = -pm
        elif signflip and np.random.randint(50) == 0:
            pm = -pm

    return input_seq, label_seq


def create_fibonacci(
    imgs,
    nums,
    lab,
    T,
    randomness,
    signflip,
):
    count, count2, pm = 0, 1, 1
    input_seq = np.zeros((img_size, T))
    label_seq = np.zeros((1, T), dtype=int)

    for t in range(T):
        rnd = np.random.randint(len(lab[count]))
        input_seq[:, t] = pm * imgs[:, lab[count][rnd]]
        label_seq[:, t] = nums[lab[count][rnd]]
        count3 = (count + count2) % 10
        count, count2 = count2, count3

        if randomness:
            if np.random.randint(200) == 0:
                count2 = np.random.randint(10)
                if signflip == 1:
                    pm = -pm
        elif signflip and np.random.randint(200) == 0:
            pm = -pm

    return input_seq, label_seq
