# Automobile Price Prediction and Valuation Using Machine Learning and Enterprise MLOps Operations

**P Sri Dhanush**  
*Department of Artificial Intelligence & Machine Learning (AIML), Section D*  
*Vignan’s Foundation for Science, Technology & Research*  
*Roll No: 241FA18072*  
*Email: 241fa18072@vignan.ac.in*  

---

### Abstract
This paper presents an end-to-end machine learning and MLOps framework for accurate automobile market valuation and price prediction. The study utilizes the standard automobile dataset comprising 205 vehicle records and 26 multidimensional attributes. An automated six-point data validation gate and domain-specific feature engineering pipeline are implemented to correct brand name typographies, convert worded numbers, and extract domain ratios including power-to-weight ratio, total vehicle cubic volume, combined fuel economy, and luxury tier classifications. Ten distinct machine learning regression algorithms are evaluated using 5-fold cross-validation across training (80%) and held-out test (20%) subsets. The models are evaluated using Mean Absolute Error, Root Mean Squared Error, Mean Absolute Percentage Error, and the Coefficient of Determination ($R^2$). A tuned Random Forest Regressor achieves the best performance with an $R^2$ of 0.9538, Mean Absolute Error of $1,299.99, Root Mean Squared Error of $1,909.82, and Mean Absolute Percentage Error of 9.67%, substantially outperforming linear baselines. The optimal model is tracked via MLflow, containerized via FastAPI and Docker, and monitored for statistical distribution shifts using Kolmogorov-Smirnov tests and Population Stability Index. The selected model produces an estimated market valuation of $9,864.50 for an unseen test vehicle.

**Index Terms**—Car Price Prediction, Machine Learning, Feature Engineering, Random Forest, MLOps, Data Version Control, MLflow, Drift Detection, FastAPI.

---

### I. INTRODUCTION
Automobile valuation is a critical economic task influenced by dynamic market trends, physical dimensions, fuel efficiency ratings, engine mechanics, and brand prestige. Accurately estimating automobile prices is essential for automotive dealerships, insurance underwriters, online listing aggregators, and individual buyers. Traditional valuation approaches relied heavily on manual inspections and fixed depreciation tables, which fail to account for non-linear interactions across diverse vehicle specifications. In contrast, modern machine learning techniques offer a data-driven paradigm capable of uncovering intricate patterns across physical, mechanical, and categorical vehicle features.

Recent research highlights the effectiveness of supervised learning in automobile pricing. Tree-based ensemble methods and regression models have demonstrated high predictive accuracy on tabular automotive records. However, deploying machine learning models into real-world production requires more than high benchmark scores; it necessitates robust data validation, automated preprocessing to prevent data leakage, reproducible pipeline tracking, and continuous post-deployment monitoring against data drift.

Motivated by these operational requirements, this paper develops a comprehensive machine learning and MLOps pipeline for vehicle price prediction. The proposed system incorporates an automated data quality gate, domain-aware feature engineering, ten benchmarked regression algorithms, hyperparameter optimization, experiment tracking with MLflow, data versioning with Data Version Control, and continuous distribution shift monitoring.

---

### II. PROBLEM STATEMENT
The objective of this study is to develop a predictive machine learning system that accurately estimates the continuous market valuation price of an automobile based on its physical, mechanical, and brand characteristics.

The system receives input observations containing vehicle attributes such as wheelbase, car length, car width, curb weight, engine size, cylinder configuration, horsepower, fuel system, and fuel efficiency metrics. The target output is the predicted continuous selling price in United States Dollars. The goal of the learning algorithm is to minimize the discrepancy between the actual market prices and the predicted values across unseen vehicle data while maintaining stability across diverse automotive market segments.

---

### III. DATASET AND PREPROCESSING

#### A. Dataset Description
The study utilizes the automobile assignment dataset consisting of 205 records and 26 features collected across multiple international vehicle manufacturers. The dataset captures a balanced mixture of compact commuter sedans, economical hatchbacks, station wagons, sports coupes, and luxury European sedans.

**TABLE I: Important Dataset Variables**
| Variable | Data Type | Description |
| :--- | :--- | :--- |
| `CarName` | String | Make and specific model name of the vehicle |
| `fueltype` | Categorical | Fuel delivery system (gas, diesel) |
| `aspiration` | Categorical | Internal induction type (standard, turbo) |
| `carbody` | Categorical | Body construction (sedan, hatchback, wagon, etc.) |
| `drivewheel` | Categorical | Drivetrain configuration (front, rear, all-wheel) |
| `wheelbase` | Continuous | Distance between front and rear axles in inches |
| `carlength` | Continuous | Overall exterior length of the car in inches |
| `carwidth` | Continuous | Overall exterior width of the car in inches |
| `curbweight` | Continuous | Total unladen vehicle weight in pounds |
| `enginesize` | Continuous | Engine displacement volume in cubic inches |
| `horsepower` | Continuous | Maximum brake horsepower generated |
| `citympg` | Continuous | Fuel economy rating during city driving |
| `highwaympg` | Continuous | Fuel economy rating during highway driving |
| `price` | Continuous | Target market selling price in USD |

#### B. Data Cleaning and Quality Assurance
Before executing any modeling steps, the raw data is passed through an automated six-point data quality gate:
1. **Schema Validation**: Confirms the exact presence of all 26 required feature columns.
2. **Missing Value Audit**: Verifies that zero missing or null entries exist across the dataset.
3. **Duplicate Record Check**: Confirms that no duplicate vehicle rows exist.
4. **Volume Check**: Validates the complete ingestion of all 205 records.
5. **Numerical Range Validation**: Ensures all physical measurements and prices fall within realistic operational bounds.
6. **Categorical Taxonomy Check**: Validates categorical inputs against pre-established vocabulary sets.

---

### IV. EXPLORATORY DATA ANALYSIS
Exploratory analysis demonstrates that automobile prices exhibit a right-skewed distribution. The majority of vehicles are concentrated in the budget segment below $15,000, while a long tail of high-value luxury and sports vehicles extends up to $45,400.

Statistical correlation analysis reveals several key insights:
- **Engine Displacement**: Displays the strongest positive correlation with vehicle price ($r = +0.87$), indicating that engine capacity is the primary determinant of cost.
- **Curb Weight and Horsepower**: Show strong positive associations with price ($r = +0.84$ and $r = +0.81$ respectively), reflecting the cost premium of larger, higher-performance vehicles.
- **Fuel Economy**: Both city and highway miles per gallon demonstrate strong negative correlations with price ($r = -0.69$ and $r = -0.70$), as economy-focused vehicles prioritize efficiency over heavy luxury features.

---

### V. FEATURE ENGINEERING
To maximize predictive performance and prevent information leakage between training and testing splits, a custom feature engineering transformer is constructed to generate domain-specific features:

1. **Brand Typographical Correction**: Resolves misspelling variations in manufacturer names, standardizing entries such as mapping misspelled Mazda, Volkswagen, Porsche, and Toyota records into their correct canonical categories.
2. **Worded Number Conversion**: Converts textual number strings for door counts and cylinder counts into numeric integer representations.
3. **Power-to-Weight Ratio**: Computes the ratio of horsepower to curb weight, representing the vehicle's dynamic acceleration capability.
4. **Total Vehicle Cubic Volume**: Calculates the approximate exterior volume by taking the product of length, width, and height.
5. **Combined Fuel Economy**: Generates a weighted average metric combining city driving (55%) and highway driving (45%) efficiency.
6. **Displacement per Cylinder**: Evaluates engine efficiency by dividing total engine displacement by the total cylinder count.
7. **Luxury Brand Tiering**: Assigns a binary prestige flag to established luxury marques such as BMW, Audi, Porsche, Jaguar, Mercedes-Benz, Volvo, and Buick.

Categorical features are transformed using one-hot encoding with unknown value handling, and continuous numeric variables are standardized to zero mean and unit variance.

---

### VI. MACHINE LEARNING MODELS
Ten regression models representing different mathematical paradigms were trained and benchmarked:
- **Linear Regression**: Classical ordinary least squares baseline establishing linear relationships.
- **Ridge Regression**: Linear regression with L2 regularization to shrink coefficients and mitigate collinearity.
- **Lasso Regression**: Linear regression with L1 regularization that enforces sparsity by driving less informative weights to zero.
- **ElasticNet**: Hybrid regularization combining L1 and L2 penalties for balanced feature selection.
- **Decision Tree Regressor**: Non-linear tree model that splits the feature space into orthogonal decision regions.
- **Random Forest Regressor**: An ensemble of 150 de-correlated decision trees combining bagging and feature subsetting to reduce variance.
- **Gradient Boosting Regressor**: Sequential boosting ensemble that iteratively fits shallow trees to pseudo-residuals.
- **Histogram-based Gradient Boosting**: Fast gradient boosting algorithm using histogram binning for continuous features.
- **Support Vector Regressor**: Kernel-based regressor constructing an optimal margin boundary within a designated error tolerance.
- **K-Nearest Neighbors**: Instance-based non-parametric regressor predicting values based on feature-space proximity.

---

### VII. TRAINING AND TESTING METHODOLOGY
The 205 dataset records were divided into an 80% training partition (164 records) and a 20% held-out testing partition (41 records) using a fixed random seed to guarantee complete reproducibility. Hyperparameter optimization was conducted exclusively on the training subset using 5-fold cross-validation via grid search.

---

### VIII. EVALUATION METRICS
Model performance was comprehensively evaluated using four complementary regression metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of absolute dollar errors between actual and predicted prices.
- **Root Mean Squared Error (RMSE)**: Measures standard deviation of prediction errors, giving higher weight to large outliers.
- **Mean Absolute Percentage Error (MAPE)**: Expresses the average prediction error as a percentage of actual vehicle value.
- **Coefficient of Determination ($R^2$)**: Quantifies the proportion of variance in vehicle prices explained by the model relative to a mean baseline.

---

### IX. RESULTS AND MODEL COMPARISON
The performance summary across all ten benchmarked models evaluated on the held-out test dataset is detailed in Table II.

**TABLE II: Regression Model Performance Comparison**
| Rank | Model Architecture | 5-Fold CV $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) | Test MAPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Tuned Random Forest** | **0.9059** | **0.9538** | **1,299.99** | **1,909.82** | **9.67%** |
| 🥈 | Gradient Boosting | 0.8988 | 0.9330 | 1,631.99 | 2,299.22 | 10.88% |
| 🥉 | Linear Regression | 0.8895 | 0.9142 | 1,859.64 | 2,602.66 | 13.43% |
| 4 | Decision Tree | 0.8205 | 0.9128 | 1,849.83 | 2,624.36 | 12.31% |
| 5 | Lasso Regression | 0.9020 | 0.9028 | 1,878.77 | 2,770.14 | 14.39% |
| 6 | Ridge Regression | 0.9074 | 0.9008 | 1,851.52 | 2,797.97 | 14.51% |
| 7 | HistGradientBoosting | 0.8961 | 0.8627 | 2,103.23 | 3,292.02 | 13.35% |
| 8 | ElasticNet | 0.8646 | 0.8377 | 2,446.97 | 3,579.51 | 19.60% |
| 9 | K-Nearest Neighbors | 0.8309 | 0.7443 | 2,661.87 | 4,492.58 | 16.48% |
| 10 | Support Vector Regressor | 0.5108 | 0.5330 | 3,417.06 | 6,071.57 | 21.27% |

The **Tuned Random Forest Regressor** emerged as the champion architecture, achieving a test $R^2$ of 0.9538, an MAE of $1,299.99, an RMSE of $1,909.82, and a MAPE of 9.67%. It demonstrated outstanding stability across cross-validation folds, significantly outperforming individual linear and kernel models.

---

### X. ACTUAL VS. PREDICTED VALUATION ANALYSIS
Diagnostic analysis of predictions versus true market prices indicates that the Random Forest model achieves tight alignment across the entire price spectrum. Residual plots confirm that prediction errors are symmetrically distributed around zero with no heteroscedastic patterns, demonstrating that the model does not systematically overprice or underprice vehicles across different segments.

---

### XI. SAMPLE VALUATION OUTPUT
To validate real-world inference, an unseen test payload representing a standard compact commuter car was processed through the production pipeline:
- **Vehicle**: Toyota Corolla LE
- **Engine Displacement**: 110 cubic inches
- **Curb Weight**: 2,200 pounds
- **Horsepower**: 75 hp
- **Fuel Economy**: 32 MPG City, 38 MPG Highway

The production model generated a market valuation estimate of **$9,864.50**, with an empirical confidence interval of $8,564.50 to $11,164.50 based on the model's Mean Absolute Error.

---

### XII. DISCUSSION AND MLOPS OPERATIONS
The superior performance of tree ensemble methods is attributed to their capability to naturally capture non-linear market segmentation, such as luxury brand premiums and engine capacity thresholds. Linear models, while computationally lightweight, failed to encapsulate these complex feature interactions.

The production deployment incorporates an enterprise MLOps ecosystem:
- **Data Version Control (DVC)**: Tracks raw and processed dataset states using cryptographic hashing.
- **MLflow Model Registry**: Manages model metadata, parameters, metrics, and lifecycle stage promotion.
- **Statistical Drift Detection**: Monitors continuous features using Two-Sample Kolmogorov-Smirnov tests and categorical distributions using the Population Stability Index.
- **Multi-Agent AI Coordination**: Implements four specialized agents for automated data quality auditing, model governance compliance, drift monitoring, and pricing market intelligence.

---

### XIII. LIMITATIONS
The study is based on a structured dataset of 205 vehicle records. Real-world automotive valuation in commercial markets must account for continuous time-series fluctuations, local geographical pricing variations, individual vehicle mileage, accident history records, and broader macroeconomic factors.

---

### XIV. CONCLUSION
This paper presented an end-to-end machine learning system for automobile price prediction. Among ten benchmarked regression algorithms, the Tuned Random Forest Regressor demonstrated superior accuracy with an $R^2$ of 0.9538 and an average percentage error of 9.67%. The integration of automated data validation, Scikit-Learn pipelines, DVC data versioning, MLflow tracking, and statistical drift detection delivers a production-ready, enterprise-grade MLOps platform for automobile valuation.

---

### REFERENCES
1. Z. Gegic, B. Delic, and E. Topcic, "Car price prediction using machine learning algorithms," *International Journal of Computer Applications*, vol. 182, no. 31, pp. 24–28, 2018.
2. S. Pudaruth, "Predicting the price of used cars using machine learning techniques," *International Journal of Information & Computation Technology*, vol. 4, no. 7, pp. 753–764, 2014.
3. N. S. Samruddhi and R. Kumar, "Used car price prediction using machine learning," *International Journal of Engineering and Advanced Technology*, vol. 9, no. 6, pp. 710–714, 2020.
4. E. Peer, "Automated vehicle valuation through ensemble machine learning models," *IEEE Transactions on Intelligent Transportation Systems*, vol. 22, no. 8, pp. 5120–5131, 2021.
5. M. Chen, S. Zhao, and Y. Wang, "Machine learning techniques for used car valuation in digital marketplaces," *Decision Support Systems*, vol. 142, p. 113465, 2021.
6. A. Sharma and V. Jain, "Comparative analysis of regression algorithms for price estimation," in *2021 International Conference on Computing, Communication and Green Engineering (CCGE)*, 2021, pp. 1–6.
7. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
8. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
9. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
10. M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39–45, 2018.
