# Path of Exile Machine Learning
Set of Python scripts to train models on Path of Exile stuff using Tensorflow.

## Models
List of the current models : 
- Currency Type (e.g. It's a chaos orb) (Most used ones in trade for now)
- Stack Size (e.g. It's a stack of 12 currency) (Up to 40)
- Item Links (e.g. It's a 5L item) (2L, 3L, 4L, 5L, 6L, no link)
- Item Sockets (e.g. It's 4 sockets item) (1S, 2S, 3S, 4S, 5S, 5S, no socket)
- Socket Color (e.g. That socket is blue) (Blue, Red, Greeb, White, Abyssal)

## Examples
Show the image:

![2](https://user-images.githubusercontent.com/25111613/126014531-2a5c7bf2-e2f1-460c-8a5e-58775ffc5c51.png)

Models anwser: 
> It's a `ancient_orb`. The stack size is `2`.

--- 

Show the image:

![2021-07-13_16-33_3](https://user-images.githubusercontent.com/25111613/126014667-f58aeaec-d36d-4b81-a8e4-99d68115ec72.png)

Models anwser: 
> It's a `5L` and `6S` item. Socket colors are `green`, `red`, `red`, `green`, `red`, `blue`.

## Requirements
- Python 3.x

If you want to train on the GPU (it's a lot faster, then on the CPU) : 

- CUDA Toolkit (NVIDIA Computing Toolkit)
- cuDNN (NVIDIA CUDA Deep Neural Network library)
