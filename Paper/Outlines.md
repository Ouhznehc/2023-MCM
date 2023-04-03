# Outlines of the Understanding of Used Boats Prices

## Statistical Analysis 统计分析

We mainly use python as our analyzing tool.

### Preprocessing Data 数据加工

First of all, we turn the `xlsx` format data sheet into `csv` format. The conversion causes some minor errors like extra spaces and unexcepted characters which can be easily filtered out by text editor and python string operating functions like `strip()` or so.

Secondly, we should 

By our analysis, the data has some missing, mistaken or inconsistent items.

Merging the two subsets into one.

#### Value Interpolation 插值

#### Spell Error Correction 拼写纠正

### Infer and Analyze Predicators 变元分析

#### Manufacture Year and Depreciation Rate 经年折旧



![](..\Resources\average_price_by_year.png)

#### Geographic Region and State/Country 地理区位

#### Make 产商

#### Variant and Specification 品类规格

Given all three factors above, we still cannot figure out 

##### Additional Data Collection 收集额外数据

Without

We have seen that the length will significantly influence the listing price of used boats. But 

## Predict Model Establishment 预测建模

### Heuristic Hierarchical Multiple Regression  启发式分层多元回归

 #### Variant Scoring 品类打分

There are lots of labels related to specifications, and it is hard for us to discuss their influence on listing price one by one. However, trying to directly preform regression based on these labels by machine learning models can cause overfitting issues

However, the variant information is relatively pure.

+ 时间：时间是一个准连续且有序的变量，且数据关于时间有明显的线性关系，第一层利用一元线性回归筛去时间变量（标准为 2015 年，可调参）。
+ 地区：地区是一个较小





We find after the normalization by year, the inpurity rate among regions is significantly larger than that among variants. It indicates that 

