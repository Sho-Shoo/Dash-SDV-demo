import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.lite import SingleTablePreset
from matplotlib import pyplot as plt


def get_linear_dataframe(n: int = 1000, grad: float = 2.0, random_state: int = None):
    np.random.seed(seed=random_state)
    X = np.random.rand(n) * 100
    noise = np.random.rand(n) * 20
    Y = X * grad + noise

    df = pd.DataFrame({"X": X, "Y": Y})
    return df


def get_sin_dataframe(n: int = 1000, random_state: int = None):
    np.random.seed(seed=random_state)
    X = np.random.rand(n) * 100
    noise = np.random.rand(n) * 20
    Y = np.sin(X / 10) * 20 + noise

    df = pd.DataFrame({"X": X, "Y": Y})
    return df


def get_synthetic_datasets(real_data, sample_n: int = 500, epochs: int = 100) -> dict[str, pd.DataFrame]:
    data = real_data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)

    ct_gan_synthesizer = CTGANSynthesizer(metadata, epochs=epochs, enforce_rounding=False)
    ct_gan_synthesizer.fit(data)

    tvae_synthesizer = TVAESynthesizer(metadata, epochs=epochs, enforce_rounding=False)
    tvae_synthesizer.fit(data)

    fast_ml_synthesizer = SingleTablePreset(metadata, name='FAST_ML')
    fast_ml_synthesizer.fit(data)

    gaussian_synthesizer = GaussianCopulaSynthesizer(metadata, enforce_rounding=False)
    gaussian_synthesizer.fit(data)

    ct_gan_synthetic_data = ct_gan_synthesizer.sample(sample_n)
    tvae_synthetic_data = tvae_synthesizer.sample(sample_n)
    fast_ml_synthetic_data = fast_ml_synthesizer.sample(sample_n)
    gaussian_synthetic_data = gaussian_synthesizer.sample(sample_n)

    return {
            'ct_gan_synthetic_data': ct_gan_synthetic_data,
            'tvae_synthetic_data': tvae_synthetic_data,
            'fast_ml_synthetic_data': fast_ml_synthetic_data,
            'gaussian_synthetic_data': gaussian_synthetic_data
           }


def visually_compare_models(real_data_getter: 'function', sample_n: int = 500, epochs: int = 100):
    size = 2

    data = real_data_getter()
    synthetic_datasets = get_synthetic_datasets(real_data_getter, sample_n=sample_n, epochs=epochs)

    ct_gan_synthetic_data = synthetic_datasets['ct_gan_synthetic_data']
    tvae_synthetic_data = synthetic_datasets['tvae_synthetic_data']
    fast_ml_synthetic_data = synthetic_datasets['fast_ml_synthetic_data']
    gaussian_synthetic_data = synthetic_datasets['gaussian_synthetic_data']

    plt.scatter(data.X, data.Y, s=size, c='blue', label="Original data")
    plt.scatter(ct_gan_synthetic_data.X, ct_gan_synthetic_data.Y, s=size, c='red', label="CTGANSynthesizer")
    plt.scatter(tvae_synthetic_data.X, tvae_synthetic_data.Y, s=size, c='purple', label="TVAESynthesizer")
    # plt.scatter(fast_ml_synthetic_data.X, fast_ml_synthetic_data.Y, s=size, c='black', label="Fast ML")
    # plt.scatter(gaussian_synthetic_data.X, gaussian_synthetic_data.Y, s=size, c='pink', label="Gaussian Copula")
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    visually_compare_models(get_sin_dataframe)
