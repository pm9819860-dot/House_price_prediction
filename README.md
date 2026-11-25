# House Price Prediction Using Machine Learning

An interactive machine-learning web app that estimates median house values from housing and demographic features using a Random Forest Regressor.

## Features
- California Housing dataset loaded through scikit-learn
- Data split into training and test sets
- Random Forest regression model
- Evaluation using MAE, RMSE, and R²
- Interactive Streamlit interface for predictions

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Streamlit

## Run locally

1. Install Python 3.10 or newer.
2. Extract this project folder and open a terminal inside it.
3. Create and activate a virtual environment (recommended):

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The first run downloads the California Housing dataset through scikit-learn and trains the model. Internet access may be needed the first time.

## Machine learning workflow
1. Load the dataset.
2. Separate input features (`X`) and target (`y`).
3. Split data into 80% training and 20% testing.
4. Train a `RandomForestRegressor`.
5. Evaluate using MAE, RMSE, and R².
6. Predict from user-entered feature values.

## Dataset and limitations
The project uses the California Housing dataset distributed by scikit-learn. The target is median house value in units of **$100,000**. The app converts the model output to dollars for display.

This is an educational portfolio project. It is not a current market valuation tool and should not be used for actual property, lending, or investment decisions. The dataset describes California districts, not Bengaluru or other Indian property markets.

## Suggested next improvements
- Compare Linear Regression, Decision Tree, and Random Forest.
- Add feature importance visualization.
- Add cross-validation and hyperparameter tuning.
- Deploy on Streamlit Community Cloud.
