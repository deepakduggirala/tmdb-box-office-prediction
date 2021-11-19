import numpy as np
import pandas as pd

def make_submission_file(name, model, X, y, test_data, submissions_filename, output_location):
    model.fit(X, y)
    y_pred = model.predict(test_data)
    
    subs = pd.read_csv(submissions_filename)
    subs['revenue'] = np.exp(y_pred)
    
    output_filename = output_location + '/' + name + '.csv'
    subs.to_csv(output_filename, index=False)
    
    print(f'kaggle competitions submit -c tmdb-box-office-prediction -f {output_filename} -m "{name}"')
    
