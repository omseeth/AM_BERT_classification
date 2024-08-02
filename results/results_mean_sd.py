"""
This script helps with calculating the mean and standard deviation for the
test results from the experiments of this project.

test_results = [
    {
        'fact': {'precision': 0.5436241610738255, 'recall': 0.6136363636363636, 'f1-score': 0.5765124555160143, 'support': 132},
        'policy': {'precision': 0.8198757763975155, 'recall': 0.8627450980392157, 'f1-score': 0.8407643312101911, 'support': 153},
        'reference': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
        'testimony': {'precision': 0.9136363636363637, 'recall': 0.8237704918032787, 'f1-score': 0.8663793103448277, 'support': 244},
        'value': {'precision': 0.8282828282828283, 'recall': 0.8266129032258065, 'f1-score': 0.8274470232088799, 'support': 496},
        'accuracy': 0.804093567251462,
        'macro avg': {'precision': 0.8210838258781067, 'recall': 0.8253529713409329, 'f1-score': 0.8222206240559826, 'support': 1026},
        'weighted avg': {'precision': 0.8108722598500199, 'recall': 0.804093567251462, 'f1-score': 0.8065758889269157, 'support': 1026}
    },
    {
        'fact': {'precision': 0.6491228070175439, 'recall': 0.5606060606060606, 'f1-score': 0.6016260162601625, 'support': 132},
        'policy': {'precision': 0.9078014184397163, 'recall': 0.8366013071895425, 'f1-score': 0.8707482993197279, 'support': 153},
        'reference': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
        'testimony': {'precision': 0.8547717842323651, 'recall': 0.8442622950819673, 'f1-score': 0.8494845360824742, 'support': 244},
        'value': {'precision': 0.8279773156899811, 'recall': 0.8830645161290323, 'f1-score': 0.8546341463414633, 'support': 496},
        'accuracy': 0.8255360623781677,
        'macro avg': {'precision': 0.8479346650759213, 'recall': 0.8249068358013206, 'f1-score': 0.8352985996007657, 'support': 1026},
        'weighted avg': {'precision': 0.8234102256164914, 'recall': 0.8255360623781677, 'f1-score': 0.8234033989588201, 'support': 1026}
    },
    {
        'fact': {'precision': 0.5355191256830601, 'recall': 0.7424242424242424, 'f1-score': 0.6222222222222222, 'support': 132},
        'policy': {'precision': 0.8791946308724832, 'recall': 0.8562091503267973, 'f1-score': 0.8675496688741721, 'support': 153},
        'reference': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
        'testimony': {'precision': 0.8787878787878788, 'recall': 0.8319672131147541, 'f1-score': 0.8547368421052632, 'support': 244},
        'value': {'precision': 0.8506493506493507, 'recall': 0.7923387096774194, 'f1-score': 0.8204592901878913, 'support': 496},
        'accuracy': 0.8050682261208577,
        'macro avg': {'precision': 0.8288301971985547, 'recall': 0.8445878631086426, 'f1-score': 0.8329936046779098, 'support': 1026},
        'weighted avg': {'precision': 0.8212004127290198, 'recall': 0.8050682261208577, 'f1-score': 0.810304122883002, 'support': 1026}
    }
]

last change
OMS 27.07.2024
"""

import statistics
from tabulate import tabulate

test_results = [
{

'0': {'precision': 0.9463667820069204, 'recall': 0.9298203011170472, 'f1-score': 0.9380205781479667, 'support': 4118.0},

'1': {'precision': 0.17897727272727273, 'recall': 0.225, 'f1-score': 0.19936708860759494, 'support': 280.0},

'accuracy': 0.8849477035015917,

'macro avg': {'precision': 0.5626720273670965, 'recall': 0.5774101505585236, 'f1-score': 0.5686938333777808, 'support': 4398.0},

'weighted avg': {'precision': 0.8975106968322271, 'recall': 0.8849477035015917, 'f1-score': 0.8909939803600394, 'support': 4398.0}
},
{

'0': {'precision': 0.945715676728335, 'recall': 0.9434191355026712, 'f1-score': 0.9445660102115244, 'support': 4118.0},

'1': {'precision': 0.19655172413793104, 'recall': 0.20357142857142857, 'f1-score': 0.2, 'support': 280.0},

'accuracy': 0.8963165075034106,

'macro avg': {'precision': 0.571133700433133, 'recall': 0.5734952820370499, 'f1-score': 0.5722830051057622, 'support': 4398.0},

'weighted avg': {'precision': 0.8980199271318563, 'recall': 0.8963165075034106, 'f1-score': 0.8971629900070617, 'support': 4398.0}
},
{

'0': {'precision': 0.944015444015444, 'recall': 0.9499757163671685, 'f1-score': 0.9469862018881626, 'support': 4118.0},

'1': {'precision': 0.1889763779527559, 'recall': 0.17142857142857143, 'f1-score': 0.1797752808988764, 'support': 280.0},

'accuracy': 0.9004092769440655,

'macro avg': {'precision': 0.5664959109840999, 'recall': 0.56070214389787, 'f1-score': 0.5633807413935195, 'support': 4398.0},

'weighted avg': {'precision': 0.8959456535430582, 'recall': 0.9004092769440655, 'f1-score': 0.898141486590982, 'support': 4398.0}
}

]


def calculate_stats(data, key):
    """
    Calculates means and standard deviations for given data set.

    Parameters:
        data, dict: with metrics as keys and reported results as values.
        key, str: of metric

    Returns:
        tuple: of mean and standard deviation.
    """
    values = [entry[key] for entry in data]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return mean, stdev


table_data = []

# Extract keys from the first entry of the provided data
keys = test_results[0].keys()

# The support for the data is always the same. We don't need to consider
# accuracy and macro avg. For the other metrics, mean and standard deviation
# are calculated
for key in keys:
    if key in ["accuracy", "macro avg", "support"]:
        continue

    row = [key]

    for metric in ["precision", "recall", "f1-score"]:
        mean, stdev = calculate_stats([entry[key] for entry in test_results], metric)
        row.extend([f"{round(mean, 2)}", f"$\pm${stdev:.3f}"])
    table_data.append(row)

print("\nMean results from all training and test rounds with standard deviations:\n")

headers = ["Category", "Precision Mean", "Precision SD", "Recall Mean",
           "Recall SD", "F1 Mean", "F1 SD"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
