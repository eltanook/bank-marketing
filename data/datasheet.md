# Bank Marketing Dataset Documentation

## Source
- **Origin**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Version**: Bank Additional Full (`bank-additional-full.csv`)
- **Size**: 41,188 instances, 20 features + target.

## Variables
| Variable | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the customer. |
| `job` | Categorical | Type of job. |
| `marital` | Categorical | Marital status. |
| `education` | Categorical | Education level. |
| `default` | Categorical | Has credit in default? |
| `housing` | Categorical | Has housing loan? |
| `loan` | Categorical | Has personal loan? |
| `contact` | Categorical | Contact communication type. |
| `month` | Categorical | Last contact month. |
| `day_of_week` | Categorical | Last contact day of the week. |
| `duration` | Numeric | Last contact duration in seconds. |
| `campaign` | Numeric | Number of contacts during this campaign. |
| `pdays` | Numeric | Days passed after last contact from a previous campaign. |
| `previous` | Numeric | Number of contacts before this campaign. |
| `poutcome` | Categorical | Outcome of the previous marketing campaign. |
| `emp.var.rate` | Numeric | Employment variation rate - quarterly indicator. |
| `cons.price.idx` | Numeric | Consumer price index - monthly indicator. |
| `cons.conf.idx` | Numeric | Consumer confidence index - monthly indicator. |
| `euribor3m` | Numeric | Euribor 3 month rate - daily indicator. |
| `nr.employed` | Numeric | Number of employees - quarterly indicator. |
| `y` | Binary | Has the client subscribed a term deposit? (target) |

## Protected Attributes (for Equity Analysis)
- **Age (`age`)**: Important for analyzing potential bias against older or younger populations.
- **Marital Status (`marital`)**: Identified as a sensitive attribute in the project guidelines.

## Usage in Project
This dataset is used to train a classification model (Random Forest) to predict the subscription rate and perform fairness audits on the results.
