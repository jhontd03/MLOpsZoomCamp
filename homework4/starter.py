import pickle
import pandas as pd

def main(parser):

    args = parser.parse_args()
    year = args.year  
    month = args.month

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(y_pred.mean())

    df['ride_id'] = f'{int(year):04d}/{int(month):02d}_'+ df.index.astype('str')

    df['prediction'] = y_pred

    df_result = df[['ride_id', 'prediction']]

    df_result.to_parquet(
        'output.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(
            description="Read year and month values",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        parser.add_argument("--year", type=str, required=True, help="Year data extraction")
        parser.add_argument("--month", type=str, required=True, help="Month data extraction")

        main(parser)
