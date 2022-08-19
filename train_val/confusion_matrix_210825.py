from beautifultable import BeautifulTable

import copy


def confusion_matrix(actual_class, num_classes, table_header):
    table = BeautifulTable()

    for i in range(len(actual_class)):
        table.rows.append(actual_class[i])

    table.columns.header = table_header
    table.rows.header = table_header

    # ↓↓↓↓↓< True Positive >↓↓↓↓↓

    TP = [actual_class[i][i] for i in range(num_classes)]

    # ↓↓↓↓↓< True Negative >↓↓↓↓↓

    transpose_after = []
    TN_sum = []
    TN = []

    for i in range(num_classes):
        actual_class_transpose = []
        for j in range(num_classes):
            actual_class_transpose.append(actual_class[j][i])
        transpose_after.append(actual_class_transpose)
    transpose_FP = copy.deepcopy(transpose_after)
    transpose_TN = copy.deepcopy(transpose_after)

    for y in range(num_classes):
        for z in range(num_classes):
            del transpose_after[z][y]

            TN_sum.append(sum(transpose_after[z]))

        del TN_sum[y]

        TN.append(sum(TN_sum))
        TN_sum = []
        transpose_after = []
        transpose_after = copy.deepcopy(transpose_TN)

    # ↓↓↓↓↓< False Positive >↓↓↓↓↓

    FP = []
    for k in range(num_classes):
        del transpose_FP[k][k]
        FP.append(sum(transpose_FP[k]))

    # ↓↓↓↓↓< False Negative >↓↓↓↓↓

    FN = []
    actual_class_copy = copy.deepcopy(actual_class)

    for n in range(num_classes):
        del actual_class_copy[n][n]

        FN.append(sum(actual_class_copy[n]))

    # ↓↓↓↓↓< Recall >↓↓↓↓↓ -> TP / TP + FN

    Recall = []
    for s in range(num_classes):
        try:
            Recall.append(round((TP[s] / (TP[s] + FN[s])), 3))

        except:
            Recall.append(float(0))

    # ↓↓↓↓↓< Precision >↓↓↓↓↓ -> TP / TP + FP

    Precision = []
    for p in range(num_classes):
        try:
            Precision.append(round((TP[p] / (TP[p] + FP[p])), 3))
        except:
            Precision.append(float(0))

    # ↓↓↓↓↓< F1 Score >↓↓↓↓↓ -> 2 * ((Precision * Recall)/(Precision + Recall))

    Average_precision = sum(Precision) / num_classes
    Average_recall = sum(Recall) / num_classes

    F1_Score = round(2 * ((Average_precision * Average_recall) / (Average_precision + Average_recall)), 3)

    # #↓↓↓↓↓< Accuracy >↓↓↓↓↓ -> TP + TN / TP + TN + FP + FN

    numerator_accuracy = 0
    denominator_accuracy = 0
    for i in range(num_classes):
        numerator_accuracy += actual_class[i][i]
        denominator_accuracy += sum(actual_class[i])

    Accuracy = round(numerator_accuracy / denominator_accuracy, 3)

    # ↓↓↓↓↓< confusion_matrix_outcome >↓↓↓↓↓

    confusion_outcome = BeautifulTable()
    confusion_outcome.rows.append(Recall)
    confusion_outcome.rows.append(Precision)
    confusion_outcome.rows.header = ["Recall", "Precision"]
    confusion_outcome.columns.header = table_header

    print(table)
    print(confusion_outcome)
    print("Average_precision :", Average_precision)
    print("Average_recall :", Average_recall)
    print("Accuracy :", Accuracy)
    print("F1 Score :", F1_Score)
    return Average_precision, Average_recall, Accuracy, F1_Score
