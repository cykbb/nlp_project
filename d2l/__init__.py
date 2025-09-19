from .plot import (
    plot, 
    show_images
)

from .regression import (
    SyntheticRegressionDataScratch, 
    SyntheticRegressionDataTorch, 
    LinearRegressionScratch, 
    LinearRegressionTorch, 
    LinearRegressionTorchL2,
    LinearRegressionTorchL2Optim
)

from .classification import (
    FashionMNIST,
    SoftmaxClassifierScratch,
    SoftmaxClassifierTorch,
    MLPClassifierTorch
)

from .utils import (
    cpu,
    gpu,
    num_gpus,
    try_gpu
)

__all__ = [
    # Plot utilities
    'plot', 
    'show_images',

    # Regression
    'SyntheticRegressionDataScratch',
    'SyntheticRegressionDataTorch',
    'LinearRegressionScratch',
    'LinearRegressionTorch',
    'LinearRegressionTorchL2',
    'LinearRegressionTorchL2Optim',
    
    # Classification
    'FashionMNIST',
    'SoftmaxClassifierScratch',
    'SoftmaxClassifierTorch',
    'MLPClassifierTorch',
    
    # Utils
    'cpu',
    'gpu',
    'num_gpus',
    'try_gpu'
]