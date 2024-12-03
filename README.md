# Deep Learning for Neuroscience using OOP Principles and Best Practices
### Contributors: Sierra Reschke and Nolan Brady

## Overview

This project is an effort to combine object-oriented programming (OOP) principles with deep learning to explore how neural network architectures can recreate statistical responses from complex BOLD signal inputs. In neuroscience, statistical models have long been used to estimate variable weights on neural activation, despite the inherently non-linear nature of brain activity. By leveraging deep learning, we aim to provide a more flexible and accurate approach, while demonstrating how OOP principles can make such workflows scalable and maintainable.

The primary goal of this project is to show how design patterns like Factory and Builder can streamline the development process. These patterns allow researchers to efficiently switch between models and data configurations, reducing repetitive code and making experimentation far more manageable. The result is not only a powerful deep learning framework but also an example of how to organize a project in a way that’s easy to maintain and expand.

---

## Key Features

The project revolves around two main design patterns: Factory and Builder. The Factory pattern is used to create deep learning models dynamically. This means you can easily benchmark different architectures on the same dataset without the overhead of writing boilerplate code for each model. For example, swapping between an autoencoder and a convolutional network is as simple as changing a single argument.

For generating datasets, we’ve implemented the Builder pattern. This gives you precise control over the synthetic data being passed to the models, while allowing for flexible addition or removal of preprocessing steps. This approach is particularly useful in larger projects where preprocessing pipelines can become unwieldy. By centralizing this logic in a Builder, we avoid the need for multiple scripts or tangled configurations.

To ensure everything works as intended, we’ve also included a suite of tests. These tests validate that the models are being constructed correctly and that the datasets meet the expected specifications. While this is a simple example, it demonstrates how testing can be incorporated into deep learning projects to improve robustness and reproducibility.

---

## How to Use

### Setting Up
Before starting it's advised you use `anaconda` to create a new project environment. This can be done by running:
```bash
conda create --neuro_dl_example python=3.12.2
conda activate neuro_dl_example
```
Once you have done this please run:
```bash
pip install -r requirements.txt
```
### To run the tests
We have been using PyCharm as our editor in which you can click on the `./tests` folder and select run tests.

### How to run code
Finally, to run the code navigate the `./pytorch_glm.ipynb` file and run all cells. This will run the Builder and ModelFactory then run the models on the synthesized data.
