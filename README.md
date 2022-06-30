# Lateral Lisp Detection

## Requirements

```
FFTW, WAV, Statistics, ArgParse, LinearAlgebra, Plots
```

## Usage

### Calibration

To get the calibration values, create two WAV recordings of a lisp and proper pronunciations, e.g. `lisp.wav` and `normal.wav`, then:

```sh
$ julia calibrate.jl normal.wav lisp.wav
```

This will output your frequency bands:

```
Detected peak range: [2099, 3091]
Detected peak range: [2779, 3777]
```

### Analysis

To analyze texts using the calibration results on texts in e.g. `texts`:

```sh
$ julia analyzetext.jl texts -l 2779 -L 3777 -n 2250 -N 3249
```

For analyzing larger datasets, it's recommended to enable multithreading:

```sh
$ julia -t auto analyzetext.jl texts -l 2779 -L 3777 -n 2250 -N 3249
```

## Documentation/paper

A rendered PDF of the documentation file in `paper` can be found [HERE](https://patrick.munni.ch/lateral-lisp.pdf).
