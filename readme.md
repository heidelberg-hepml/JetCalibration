# JetCalibration

JetCalibration is a PyTorch Lightning-based framework for jet calibration regression tasks, supporting various neural network regression approaches including MSE, heteroscedastic, and Gaussian mixture model (GMM) regression.

## Code Structure
JetCalibration/
├── Source/
│ ├── dataset.py # Data loading and PyTorch Lightning DataModule(s)
│ ├── model.py # Model definitions (BaseModel, MLP_MSE_Regression, etc.)
│ ├── network.py # MLP and network architecture definitions
│ ├── plots.py # Plotting utilities and analysis scripts
│ └── ... # Other source files
├── main.py # Main entry point for training and plotting
├── param_cards/ # Example YAML parameter cards for runs
├── results/ # Output directory for trained models and logs (gitignored)
├── data/ # Data directory (gitignored)
├── data_v2/ # Alternative data directory (gitignored)
├── runfiles/ # Run-related files/scripts
├── misc/ # Miscellaneous scripts/utilities
├── .gitignore # Git ignore rules
└── README.md # (You are here)


### Main Components

- **main.py**  
  The main script to launch training or plotting. It parses command-line arguments to determine the run type and loads parameters from a YAML file.

- **Source/model.py**  
  Contains the core model classes:
  - `BaseModel`: Abstract base class with common training/validation logic.
  - `MLP_MSE_Regression`: Standard MLP with MSE loss.
  - `MLP_Heteroscedastic_Regression`: MLP predicting both mean and variance.
  - `MLP_GMM_Regression`: MLP predicting parameters for a Gaussian mixture model.

- **Source/dataset.py**  
  Implements data loading and preprocessing, typically via PyTorch Lightning's `DataModule` interface.

- **Source/network.py**  
  Defines the MLP and other network architectures used by the models.

- **Source/plots.py**  
  Provides plotting and analysis utilities for evaluating model performance.

- **param_cards/**  
  Contains YAML files specifying hyperparameters and run configurations.

- **results/**  
  Output directory for model checkpoints, logs, and TensorBoard files (ignored by git).

- **data/**, **data_v2/**  
  Directories for input data (ignored by git).

## Getting Started

### 1. Install Dependencies

Make sure you have Python 3.8+ and install the required packages (ideally in a virtual environment):

```bash
pip install -r requirements.txt
```
or manually install the main dependencies:
```bash
pip install pytorch-lightning torch numpy matplotlib pyyaml
```

### 2. Prepare Data

Place your data files (e.g., `.npy` files) in the `data/` or `data_v2/` directory.

### 3. Configure a Run

Edit or create a parameter card in `param_cards/`, e.g. `param_cards/example.yaml`. This YAML file should specify:
- `run_name`: Name for the run
- `data_path`: Path to the data file
- `model`: Model type (e.g., `MLP_MSE_Regression`)
- `model_params`: Model hyperparameters (input/output dimensions, hidden layers, etc.)
- `epochs`, `batch_size`, etc.

### 4. Start Training

Run the following command to start a training run:

```bash
python main.py train param_cards/example.yaml
```

This will:
- Create a new results directory with a timestamp and run name
- Save the parameter card and logs
- Train the model and save checkpoints and TensorBoard logs

### 5. Plotting/Evaluation

To run plotting or evaluation scripts, use:

```bash
python main.py plot <path_to_results_dir>
```

This will load the saved model and parameters from a previous run and generate plots or evaluation metrics.

## Notes

- All output (checkpoints, logs, TensorBoard) is saved in the `results/` directory.
- Data directories (`data/`, `data_v2/`) and results are gitignored by default.
- For custom models or data, extend the classes in `Source/model.py` and `Source/dataset.py`.

---

For more details, see the source code and parameter card examples.

---

*Repository: [heidelberg-hepml/JetCalibration](https://github.com/heidelberg-hepml/JetCalibration)*