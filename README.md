[![Build](https://img.shields.io/github/actions/workflow/status/adamvvu/tf_compactprogbar/tf_compactprogbar_tests.yml?style=for-the-badge)](https://github.com/adamvvu/tf_compactprogbar/actions/workflows/tf_compactprogbar_tests.yml)
[![PyPi](https://img.shields.io/pypi/v/tf_compactprogbar?style=for-the-badge)](https://pypi.org/project/tf-compactprogbar/)
[![Downloads](https://img.shields.io/pypi/dm/tf_compactprogbar?style=for-the-badge)](https://pypi.org/project/tf-compactprogbar/)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](https://github.com/adamvvu/tf_compactprogbar/blob/master/LICENSE)
[![Binder](https://img.shields.io/badge/DEMO-binder-blue?style=for-the-badge)](https://mybinder.org/v2/gh/adamvvu/tf_compactprogbar/HEAD?labpath=examples%2Fexample.ipynb)

A simple and compact one-line progress bar for TensorFlow 2 Keras.

---

## TensorFlow Compact Progress Bar

Existing ways of monitoring training of TensorFlow Keras models either display nothing (`verbose=0`), one progress bar per epoch (`verbose=1`), or prints one line of metrics per epoch (`verbose=2`). When training for thousands of epochs, this often leads to bloated log files or crashed/sluggish Jupyter environments when working interactively.

This library provides a compact progress bar that simply displays the overall training progress by epoch. There are also a few small additional features for convenience, such as excluding certain metrics to avoid excessive clutter.

**Notebook mode**
![notebook_demo](assets/compact.png)

**Console mode**
![console_demo](assets/compact_noninteractive.png)

### Getting Started

Install from PyPi:

`$ pip install tf-compactprogbar`

#### Dependencies

- `tensorflow >= 2` *(TF 1 likely works, but untested)*
- `tqdm`
- `python >= 3.7`
- `ipywidgets` *(Optional)*
- `jupyterlab_widgets` *(Optional)*

For nice looking Jupyter/IPython progress bars, make sure you have `ipywidgets` and `jupyterlab_widgets` if you are on Jupyter Lab.

#### Usage

To use it, disable the built-in logging (`verbose=0`) and pass it in as a Callback:
```
from tf_compactprogbar import CompactProgressBar

progBar = CompactProgressBar()
history = model.fit(X_train, Y_train,
                    epochs=200,
                    batch_size=100,
                    verbose=0,
                    validation_data = (X_test, Y_test),
                    callbacks=[progBar])
```


### Documentation

```
# Call signature
CompactProgressBar(show_best=True, best_as_max=[], exclude=[], notebook='auto', epochs=None)

Args:
        - show_best    (bool)      Display best metrics. Default: True
        - best_as_max  (list)      Metrics which should be maximized (see note)
        - exclude      (list)      Metrics which should be excluded from display
        - notebook     (str/bool)  Whether to use IPython/Jupyter widget or console. Default: 'auto'
        - epochs       (int)       Optional total number of epochs. Default is inferred from `.fit`.
        
Note: When using `show_best`, by default the "best" metric is the minimum. Pass
in the metric name to `best_as_max` to change this behavior.
```

If there are any issues in Jupyter, please see the [tqdm Issues](https://github.com/tqdm/tqdm/issues) page for help or disable notebook mode with `notebook=False`.