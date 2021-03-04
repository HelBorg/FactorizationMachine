import os.path

from src.analyze import FactorisationMachine
from src.utils import TransformData

RAW_DATA_PATH = "data/combined_data_1.txt"
PROCESSED_DATA_PATH = "data/processed_data.csv"
CHUNK_SIZE = 10000

if __name__ == "__main__":
    data_transform = TransformData(RAW_DATA_PATH)
    if os.path.exists(PROCESSED_DATA_PATH):
        data_transform.load_data(PROCESSED_DATA_PATH)
    else:
        data_transform.transform()
        data_transform.save_data(PROCESSED_DATA_PATH)

    fm = FactorisationMachine()
    results = []
    for num, fold in enumerate(data_transform.get_fold()):
        X_train, X_test, y_train, y_test = fold

        fm.fit(X_train, y_train)
        y_train_pred, rmse_train, r2_train = fm.predict_and_analyze(X_train, y_train)
        y_test_pred, rmse_test, r2_test = fm.predict_and_analyze(X_test, y_test)

        result = [r2_test, rmse_test, r2_train, rmse_train]
        results.append(result)
        print(results)
    fm.analyze_results(results)
