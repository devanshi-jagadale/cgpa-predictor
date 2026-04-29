import numpy as np

def compute_features(data):
    sgpas = np.array([
        data['S1_SGPA'],
        data['S2_SGPA'],
        data['S3_SGPA'],
        data['S4_SGPA']
    ])

    x = np.arange(1, 5)

    data['sgpa_trend'] = np.polyfit(x, sgpas, 1)[0]
    data['avg_4'] = np.mean(sgpas)
    data['max_4'] = np.max(sgpas)
    data['min_4'] = np.min(sgpas)

    return data