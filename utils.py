import numpy as np

def write_output(observed, predicted):
    result = observed[(observed[:, 2] == 2) | (observed[:,2] == 3)]
    result[:,2] = np.where(result[:, 2] == 2, 0,1)
    result = result[:, [0,2]]
    result = np.column_stack((result, predicted))
    print("Resultado:\n", "InstanteTempo Observado Predito\n")
    for i in range(len(result)):
        print(result[i][0], " ", result[i][1], " ", result[i][2])

    return