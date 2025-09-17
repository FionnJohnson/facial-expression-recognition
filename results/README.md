
## **Results**

### Model Performance Across Feature Sets
![Accuracy Table](/results/accuracy_table1.png)

### Raw vs Aligned Data
![Alignment Results](/results/accuracy_table2.png)

### Pairwise Expression Classification
![Pairwise Results](/results/accuracy_table3.png)

### Principal component plot
![PCA Variance](/results/pc_plot.png)

<br/>

### Results Summary

- Random Forests achieved the highest accuracy score when comparing all four expressions.
  - This was achieved using four time points from the landmark data
- The addition of procrustes analysis (the method of aligning the shape data) is crucial and proved to significantly improve model performance.
- Some expression pairs required complex boundaries to seperate them, whereas others were linearly separable.
  - This doesn't follow how humans can distinguish one expression from another.
