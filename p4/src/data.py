# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# other libraries
import os


class ElectricDataset(Dataset):
    """
    This class is the dataset loading the data.

    Attr:
        dataset: tensor with all the prices data. Dimensions:
            [number of days, 24].
        past_days: length used for predicting the next value.
    """

    dataset: torch.Tensor
    past_days: int

    def __init__(self, dataset: pd.DataFrame, past_days: int) -> None:
        """
        Constructor of ElectricDataset.

        Args:
            dataset: dataset in dataframe format. It has three columns
                (price, feature 1, feature 2) and the index is
                Timedelta format.
            past_days: number of past days to use for the
                prediction.
        """
        # Extract price data and organize by day
        price_series = dataset["Price"].values

        # Calculate how many complete days we have
        hours_total = len(price_series)
        complete_days = hours_total // 24

        # Reshape to daily format (each row is one day with 24 hours)
        daily_prices = price_series[: complete_days * 24].reshape(-1, 24)

        # Convert to tensor and store - using float64 to match test expectations
        self.dataset = torch.tensor(daily_prices, dtype=torch.float64)

        # Store context length for predictions
        self.past_days = past_days

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            number of days in the dataset.
        """
        # Return available samples (total days minus context window)
        return len(self.dataset) - self.past_days

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method returns an element from the dataset based on the
        index. It only has to return the prices.

        Args:
            index: index of the element.

        Returns:
            past values, starting to collecting those in the zero
                index. Dimensions: [sequence length, 24].
            current values. Start to collect those in the index
                self.sequence. Dimensions: [24].
        """
        # Get context window (input sequence)
        input_sequence = self.dataset[index:index + self.past_days]

        # Get target day to predict
        target_day = self.dataset[index + self.past_days]

        return input_sequence, target_day


def load_data(
    save_path: str,
    past_days: int = 7,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    This method returns Dataloaders of the chosen dataset. Use  the
    last 42 weeks of the training dataframe for the validation set.

    Args:
        save_path: path to save the data.
        past_days: number of past days to use for the prediction.
        batch_size: size of batches that wil be created.
        shuffle: indicator of shuffle the data. Defaults to true.
        drop_last: indicator to drop the last batch since it is not
            full. Defaults to False.
        num_workers: num workers for loading the data. Defaults to 0.

    Returns:
        train dataloader.
        val dataloader.
        test dataloader.
        means of price.
        stds of price.
    """
    # Obtenemos los datos de entrenamiento y prueba
    training_data, testing_data = download_data(save_path)

    # Calculamos el número total de días en los datos de entrenamiento
    total_hours = len(training_data)
    days_in_training = total_hours // 24

    # Extraemos todos los precios y los convertimos a tensor para calcular estadísticas
    all_prices = (
        training_data["Price"]
        .values[: days_in_training * 24]
        .reshape(days_in_training, 24)
    )
    price_tensor = torch.tensor(all_prices, dtype=torch.float64).flatten().float()

    # Calculamos la media y desviación estándar para la normalización
    price_mean = float(price_tensor.mean().item())
    price_std = float(price_tensor.std().item())

    # Separamos los datos de validación (últimas 42 semanas = 42*7 días)
    validation_size_days = 42 * 7
    validation_size_hours = validation_size_days * 24

    # Creamos los DataFrames para entrenamiento, validación y prueba
    # Para validación y prueba, necesitamos incluir días adicionales para el contexto
    train_portion = training_data.iloc[:-validation_size_hours]

    # Para validación, incluimos past_days adicionales para el primer elemento
    val_with_context = training_data.iloc[-(validation_size_hours + past_days * 24):]

    # Para prueba, necesitamos los últimos días de entrenamiento como contexto
    context_for_test = training_data.iloc[-(past_days * 24):]
    test_with_context = pd.concat([context_for_test, testing_data])

    # Creamos los datasets
    train_set = ElectricDataset(train_portion, past_days)
    val_set = ElectricDataset(val_with_context, past_days)
    test_set = ElectricDataset(test_with_context, past_days)

    # Normalizamos los datos usando la media y desviación estándar calculadas
    train_set.dataset = (train_set.dataset - price_mean) / price_std
    val_set.dataset = (val_set.dataset - price_mean) / price_std
    test_set.dataset = (test_set.dataset - price_mean) / price_std

    # Creamos los dataloaders con los parámetros especificados
    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader, price_mean, price_std


def download_data(
    path, years_test=2, begin_test_date=None, end_test_date=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download data from day-ahead electricity markets.

    Args:
        path: path to save the data
        years_test: year for the test data. Defaults to 2.
        begin_test_date: beginning test date. Defaults to None.
        end_test_date: end test date. Defaults to None.

    Raises:
        IOError: Error when reading dataset with pandas.
        Exception: Starting date for test dataset should be midnight.
        Exception: End date for test dataset should be at 0h or 23h.

    Returns:
        training dataset.
        testing dataset.
    """
    dataset: str = "NP"

    # Checking if provided directory exist and if not create it
    if not os.path.exists(path):
        os.makedirs(path)

    # If dataset is one of the existing open-access ones,
    # they are imported if they exist locally or download from
    # the repository if they do not
    if dataset in ["PJM", "NP", "FR", "BE", "DE"]:
        file_path = os.path.join(path, dataset + ".csv")

        # The first time this function is called, the datasets
        # are downloaded and saved in a local folder
        # After the first called they are imported from the local
        # folder
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0)
        else:
            url_dir = "https://zenodo.org/records/4624805/files/"
            data = pd.read_csv(url_dir + dataset + ".csv", index_col=0)
            data.to_csv(file_path)
    else:
        try:
            file_path = os.path.join(path, dataset + ".csv")
            data = pd.read_csv(file_path, index_col=0)
        except IOError as e:
            raise IOError("%s: %s" % (path, e.strerror))

    data.index = pd.to_datetime(data.index)

    columns = ["Price"]
    n_exogeneous_inputs = len(data.columns) - 1

    for n_ex in range(1, n_exogeneous_inputs + 1):
        columns.append("Exogenous " + str(n_ex))

    data.columns = columns

    # The training and test datasets can be defined by providing a number
    # of years for testing
    # or by providing the init and end date of the test period
    if begin_test_date is None and end_test_date is None:
        number_datapoints = len(data.index)
        number_training_datapoints = number_datapoints - 24 * 364 * years_test

        # We consider that a year is 52 weeks (364 days) instead of the traditional 365
        df_train = data.loc[
            : data.index[0] + pd.Timedelta(hours=number_training_datapoints - 1), :
        ]
        df_test = data.loc[
            data.index[0] + pd.Timedelta(hours=number_training_datapoints):, :
        ]

    else:
        try:
            begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
            end_test_date = pd.to_datetime(end_test_date, dayfirst=True)
        except ValueError:
            print("Provided values for dates are not valid")

        if begin_test_date.hour != 0:
            raise Exception("Starting date for test dataset should be midnight")
        if end_test_date.hour != 23:
            if end_test_date.hour == 0:
                end_test_date = end_test_date + pd.Timedelta(hours=23)
            else:
                raise Exception("End date for test dataset should be at 0h or 23h")

        print("Test datasets: {} - {}".format(begin_test_date, end_test_date))
        df_train = data.loc[:begin_test_date - pd.Timedelta(hours=1), :]
        df_test = data.loc[begin_test_date:end_test_date, :]

    return df_train, df_test
