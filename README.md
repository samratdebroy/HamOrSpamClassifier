# HamOrSpamClassifier
A Naive-Bayes Classifier that identifies whether email is ham or spam based on the words contained in the email.

## How to Run

* Download the repo in a ZIP file or clone it locally
* Extract the archive and/or go to the folder where the repo is
* Open a command prompt inside the repo or shift right-click, "Open PowerShell window here"
* Make sure you have Python installed and that it is in your PATH environment variable
* Write "python experiments.py"
* You should see the program executing on its own

## Outputs
After running the program, there should be 6 new files in the repo directory. Those are:

* model.txt
* baseline-result.txt
* stopword-model.txt
* stopword-result.txt
* wordlength-model.txt
* wordlength-result.txt

These files are in the .gitignore and will only appear once you run the program at least once

## Experiment Results
### Experiment 1
```
Doing Experiment 1
Error Analysis
Accuracy: 0.9125
Ham:
 Ham Correct: 394
 Ham incorrect: 64
 Precision: 0.8602620087336245
 Recall: 0.985
 F-Measure: 0.9184149184149185

Spam:
 Spam Correct: 336
 Spam incorrect: 6
 Precision: 0.9824561403508771
 Recall: 0.84
 F-Measure: 0.9056603773584906 
```

### Experiment 2
```
Doing Experiment 2
Error Analysis
Accuracy: 0.91375
Ham:
 Ham Correct: 394
 Ham incorrect: 63
 Precision: 0.862144420131291
 Recall: 0.985
 F-Measure: 0.9194865810968494

Spam:
 Spam Correct: 337
 Spam incorrect: 6
 Precision: 0.9825072886297376
 Recall: 0.8425
 F-Measure: 0.9071332436069985
```

### Experiment 3
```
Doing Experiment 3
Error Analysis
Accuracy: 0.92
Ham:
 Ham Correct: 393
 Ham incorrect: 57
 Precision: 0.8733333333333333
 Recall: 0.9825
 F-Measure: 0.924705882352941

Spam:
 Spam Correct: 343
 Spam incorrect: 7
 Precision: 0.98
 Recall: 0.8575
 F-Measure: 0.9146666666666667
```

## Contributors
| Name              | ID          |
| ----------------- | ------------|
| Samrat Debroy     | 40002159    |
| Gabriel Belanger  | 40002109    |
| Simon Bourque     | 40000680    |
| Nathan Shummoogum | 40004336    |
