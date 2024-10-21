from tensorflow.keras.callbacks import Callback
import re

class CompactProgressBar(Callback):
    """
    Compact one-line progress bar for training TensorFlow 2 Keras models.

    Args:
        - show_best    (bool)      Display best metrics. Default: True
        - best_as_max  (list)      Metrics which should be maximized (see note)
        - exclude      (list)      Metrics which should be excluded from display
        - notebook     (str/bool)  Whether to use IPython/Jupyter widget or console. Default: 'auto'
        - epochs       (int)       Optional total number of epochs. Default is inferred from `.fit`.
        
    Note: When using `show_best`, by default the "best" metric is the minimum. Pass
    in the metric name to `best_as_max` to change this behavior.

    This should be used as a callback during training with `.fit(..., verbose=0)`.
    """

    def __init__(self, show_best=True, best_as_max=[], exclude=[], notebook='auto', epochs=None):
        super(CompactProgressBar, self).__init__()
        self.numEpochs = epochs
        self.show_best = show_best
        self.best_as_max = best_as_max
        self.exclude = exclude
        self.notebook = notebook

        self.progBar = None
        self.bestMetrics = {}
    
    def _initialize_progbar(self):
        """
        Initializes the progress bar
        """
        if self.numEpochs is None:
            self.numEpochs = self.params.get('epochs', float('inf'))

        # Progress bar type
        if self.notebook == 'auto':
            from tqdm.auto import tqdm
        elif self.notebook:
            from tqdm.notebook import tqdm
        else:
             from tqdm import tqdm

        # Initialize progress bar
        self.progBar = tqdm(total=self.numEpochs, 
                            dynamic_ncols=True,
                            mininterval=0.1)
        self.progBar.set_description(desc='[Training]')
    
    def on_train_begin(self, logs=None):
        self._initialize_progbar()

    def on_train_end(self, logs=None):
        self.progBar.close()
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Updates the progress bar
        """
        self.progBar.update(1)

        # Current metrics
        metrics = logs

        # Excluded metrics
        for metric in self.exclude:
            metrics.pop(metric)
            metrics.pop(f'val_{metric}')

        # Compute best metrics
        if self.show_best:
            for metric, val in metrics.items():
                
                # Check mode
                toMax = False
                if metric in self.best_as_max:
                    toMax = True
                else:
                    match = re.search(r'val_(.*)', metric)
                    if match:
                        toMax = True
                        metric = match.group(1)

                # Update
                x = f'best_{metric}'
                if toMax:
                    if val > self.bestMetrics.get(x, -999):
                        self.bestMetrics[x] = val
                else:
                    if val < self.bestMetrics.get(x, 999):
                        self.bestMetrics[x] = val
            metrics.update(self.bestMetrics)

        # Display metrics
        self.progBar.set_postfix(metrics)