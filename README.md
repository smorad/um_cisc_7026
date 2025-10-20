# Steven's Introduction to Deep Learning Course
This repository contains the source code for my UM CISC 7026 course. I will update the code each time I teach the course. If you want to refer to a specific course year, please look for the specified tag (e.g., `fall-2024`).

## PDFs
There is a very nice tool that will automatically compile and render `typst` source code (slides) in your browser. Please see the below links to view the PDFs. Note that the first time you run can be a little slow. Be patient and it should load in a minute or two. If you get an error like `Failed to load git repository: CheckoutConflictError:`, please clear your browser cache or use incognito mode.


- [Lecture 0 - Welcome](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_00_welcome.typ?g-mode=slide)
- [Lecture 1 - Intro](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_01_intro.typ?g-mode=slide)
- [Lecture 2 - Linear Regression](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_02_linear_regression.typ?g-mode=slide)
- [Lecture 3 - Neural Networks](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_03_neural_networks.typ?g-mode=slide)
- [Lecture 4 - Backpropagation](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_04_backpropagation.typ?g-mode=slide)
- [Lecture 5 - Classification](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_05_classification.typ?g-mode=slide)
- [Lecture 6 - Modern Techniques](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_06_techniques.typ?g-mode=slide)
- [Lecture 7 - Convolution](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_07_convolution.typ?g-mode=slide)
- [Lecture 8 - Recurrence](https://gistd.myriad-dreamin.com/@any/github.com/smorad/um_cisc_7026/blob/main/lecture_08_recurrent.typ?g-mode=slide)



## Compiling the Material
I utilize `typst` for slide generation. You can compile the slides on your own.

If on MacOS, you can install `typst` using `brew`.
```bash
brew install typst@0.13.1
```

Alternatively, you can install `typst` from [this GitHub link](https://github.com/typst/typst/releases/tag/v0.13.1). 

To compile the material is quite simple. Navigate to this repository using the command line, and run
```bash
typst compile path_to_lecture.typ
```

This will produce a PDF labeled `path_to_lecture.pdf`

**Note:** The generated slides have transitions (often one line per slide). If you would like to generate slides without transitions for study purposes, consider setting `#let handout = False` at the beginning of the source file.