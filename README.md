Algerian Forest Fires Regression Project
Overview
This project explores the Algerian Forest Fires dataset using correlation analysis, scatter plots, and multiple linear regression models.

What the code does
Checks multicollinearity with a correlation heat map.

Compares humidity and temperature across the two regions.

Fits a multiple linear regression model for humidity.

Fits an interaction model for FFMC and fire status.

Fits a polynomial regression model for FFMC and humidity.

Runs additional regression models for FFMC and FWI.

Files
script.py — cleaned and organized analysis code.

README.md — project overview and review.

Notes on the code
The code is logically organized by task.

It uses sm.add_constant() to include intercepts correctly.

It accesses regression coefficients by column name, which is clearer and safer than positional indexing.

It uses plt.scatter() for the plots to avoid seaborn version issues.

Each plot is shown and cleared before the next one.

Review
Purpose
The code matches the project requirements well. It completes the main analysis steps and follows the task structure from the assignment.

Readability
The code is easy to follow because the tasks are labeled and the variables are named clearly. The comments also make it easier to understand what each section is doing.

Performance
The code is efficient for this dataset because it uses standard OLS models and straightforward plotting. There are no unnecessary loops or heavy computations.

Suggestions
Add model.summary() calls if you want more model detail.

Save plots or outputs if you want a reusable analysis report.

Add short interpretation comments under each regression section for a more complete submission.

Summary
This is a solid educational analysis script and should work well for the Codecademy project once the dataset is available in the workspace.
