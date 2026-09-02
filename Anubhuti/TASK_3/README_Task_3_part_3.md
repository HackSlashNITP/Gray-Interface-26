# Task 3_Part 3: Support Vector Machines (SVM)

Dataset: Dry Bean Dataset

1. Objective
Classify dry beans into their correct type (multi-class: 7 bean classes) and see how different SVM kernels and hyperparameters (C, gamma, degree) affect performance.

2. Dataset Overview
-'shape' : 13,611 rows × 17 columns
-'Class': 7 bean types - DERMASON (3,546), SIRA (2,636), SEKER (2,027), HOROZ (1,928), CALI (1,630), BARBUNYA (1,322), BOMBAY (522),  moderately imbalanced.
- No missing values
- 68 duplicate rows were identified and removed (13,611 → 13,543 rows).

3. Preprocessing
- Checked and confirmed no missing values.
- Identified and removed duplicate rows before splitting.
- Ran ANOVA F-test ('f_classif') to rank feature importance - top features were 'Area', 'ConvexArea', 'EquivDiameter', 'Perimeter', 'MinorAxisLength', 'MajorAxisLength'.
- Plotted boxplots of these top features across classes - BOMBAY can diffrentiated (much bigger beans), but classes like SEKER and DERMASON are very similar , means non-liner kernel can do better
- Label-encoded 'Class' (0 = BARBUNYA 0 -> BARBUNYA , 1 -> BOMBAY,2 -> CALI,3 -> DERMASON ,4 -> HOROZ,5 -> SEKER,6 -> SIRA).
- Splited data 80/20 with stratification on 'Class' to keep the class ratio in both train and test datasets.
- Applied 'StandardScaler' (fit on train, applied to train and test) as SVM is distance based and features like Area/ConvexArea have a much bigger scale.

4.Kernel Comparison (Default Parameters)

All four kernels were trained with default hyperparameters and training time was recorded:

Linear : Accuracy 0.9210, F1(macro) 0.9319, Precision 0.9329, Recall 0.9310, Time 2.877s
Poly : Accuracy 0.9140, F1(macro) 0.9271, Precision 0.9359, Recall 0.9221, Time 0.730s
RBF : Accuracy 0.9210, F1(macro) 0.9328, Precision 0.9342, Recall 0.9317, Time 0.667s
Sigmoid : Accuracy 0.7305, F1(macro) 0.6510, Precision 0.6864, Recall 0.6282, Time 1.309s

Observation : Linear took the longest time to train even though it's the simplest kernel but f1 score is heighest for linear , while the accuracy was tie for Linear ans RBF

5. Hyperparameter Tuning 
- Used 'GridSearchCV' (cv=5, scoring='f1_macro') separately for each kernel.
- Linear: C in [0.1, 1, 10, 100]
- RBF / Sigmoid: C in [0.1, 1, 10, 100], gamma in [0.001, 0.01, 0.1, 1, 'scale']
- Poly: C in [0.1, 1, 10], gamma in ['scale', 0.01, 0.1], degree in [2, 3, 4]

Best parameters found:

Linear : C=1, CV  F1(macro) 0.9385
RBF : C=10, gamma='scale', CV F1(macro) 0.9441
Sigmoid : C=100, gamma=0.001, CV F1(macro) 0.9383
Poly : C=10, degree=2, gamma=0.1, CV F1(macro) 0.9427

Each tuned model was then refit and evaluated on the test set with accuracy, precision, recall, F1-score, and confusion matrix is in the notebook.

Observation: After tuning, RBF came out as the best kernel overall (highest F1-macro, 0.944), which makes sense since the bean features are continuous and the class boundaries aren't linear - RBF's local similarity measure captures that better than a flat linear boundary. The biggest jump was Sigmoid, going from the worst kernel (0.65 F1 default) to almost matching everyone else (0.938) once gamma was tuned properly - so its poor default performance was really just a hyperparameter issue, that means the kernel being a bad fit for the data. Poly (degree 2) also did well after tuning.

6. Why Macro-Averaged Metrics, not just Accuracy**
As 'Class'is imbalanced (BOMBAY(522) while DERMASON's (3,546) ), only accuracy_score look good beacuse of the the majority classes. Precision, Recall, and F1 were all macro-averaged (treats every class equally even though the number of dataset are less) so the model's performance on the smaller classes

7. Conclusion
RBF kernel with C=10, gamma='scale' - it gives the best F1-score, is fast to train, and stays robust to the class imbalance when checked with macro-averaged metrics.

Link to the notebook :(https://colab.research.google.com/drive/1ygDZrKCI0lil1tTEzQHUfMGxq5WkImYV?usp=sharing)
