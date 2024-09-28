Visual Chart Recognition focuses on understanding chart images and extraction of data elements presented in the charts. The system is mainly focuses on chart images extracted from annual financial reports published by the Central Bank of Sri Lanka. Chart images can be either bar charts, pie charts, line charts or any other type found in the financial reports.

The Visual Chart recognition process involves four major steps:
1. Chart Type Classification - Classify each chart as either bar, pie or line.
2. Key-Point Detection - Detect top left and bottom right points of bar elements.
3. Data Range Extraction - Extract y-axis data range.
4. Raw Data Extraction - Integrate data from both step 2 and 3 to extract bar values.

The system begins by receiving an input image of a chart and categorizes it as either a bar chart, a pie chart, or another type. If the chart is identified as a bar, the process proceeds; otherwise if the chart is line or pie, the process is terminated. Subsequently, two parallel processes are initiated. One process involves extracting top left and bottom right points found in the bar elements, while the other involves extracting the y-axis data range. Ultimately, a raw data table is constructed using the extracted common elements and data values.

Since the dataset is quite large and the model has a large number of parameters to train, the P100 GPU provided by Kaggle was opted for.
