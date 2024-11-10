# Week 1: Introduction to Machine Learning

## Video 1: Aprendizaje automático supervisado frente a no supervisado

- **What is Machine Learning?**
  - Definition attributed to **Arthur Samuel**: "Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed."
  - Example of **Samuel's Checkers Program**:
    - Samuel created a program that learned to play checkers by playing thousands of games against itself.
    - Through this experience, it identified patterns in board positions associated with winning or losing, improving its performance.
    - Outcome: The program eventually became a better player than Samuel himself.

- **Importance of Training Data**:
  - The more data or experience a learning algorithm has, the better it can perform.
  - Practice quizzes are provided throughout the course to help reinforce understanding of these concepts.

- **Types of Machine Learning**:
  - **Supervised Learning**:
    - Most commonly used in real-world applications and the primary focus of this course’s first two segments.
  - **Unsupervised Learning**:
    - Introduced in the third segment, along with recommender systems and reinforcement learning.


- **Practical Advice on Machine Learning**:
  - Many experienced teams often apply algorithms incorrectly, leading to inefficiencies.
  - This course aims to equip learners with both the tools and the know-how to avoid common pitfalls.


## Video 2: Supervised Learning Part 1

- **Importance of Supervised Learning**:
  - 99% of the current economic value created by machine learning comes from supervised learning.
  - Supervised learning algorithms learn mappings from **input (x)** to **output (y)**.

- **Definition of Supervised Learning**:
  - In supervised learning, algorithms are given examples with correct labels (input-output pairs).
  - The algorithm learns from these examples to predict output for new inputs.

- **Examples of Supervised Learning Applications**:
  - **Spam filtering**: Input = email, Output = spam or not spam.
  - **Speech recognition**: Input = audio clip, Output = text transcript.
  - **Machine translation**: Input = English text, Output = translation in another language.
  - **Online advertising**: Predicts likelihood of user clicking an ad (revenue-driven).
  - **Self-driving cars**: Input = image/sensor data, Output = location of other cars.
  - **Visual inspection in manufacturing**: Detects defects in products like scratches or dents.

- **How Supervised Learning Works**:
  - The model is trained on labeled examples of input (x) and output (y) pairs.
  - After training, it can predict the output for a new, unseen input.

- **Regression Example: Predicting Housing Prices**:
  - **Scenario**: Predict house prices based on the size of the house.
  - **Data Visualization**: Plot of house size (square feet) vs. price (thousands of dollars).
  - **Example predict

## Video 3: Supervised Learning Part 2

- **Types of Supervised Learning Algorithms**:
  - Supervised learning algorithms learn mappings from input (x) to output (y).
  - Two main types: **Regression** (predicts continuous numbers) and **Classification** (predicts categories).

- **Classification Algorithms**:
  - **Definition**: Classification algorithms predict a limited, finite set of possible categories or classes.
  - **Example - Breast Cancer Detection**:
    - Goal: Use a diagnostic tool to classify a tumor as benign (0) or malignant (1).
    - Data: Tumor size is plotted on a graph where the output is either 0 (benign) or 1 (malignant).
    - Difference from regression: Predicts discrete categories (e.g., 0 or 1) rather than continuous values.
  - **Multiple Categories**:
    - Classification can include more than two categories, e.g., different cancer types.
    - Terms "class" and "category" are used interchangeably.

- **Distinguishing Classification from Regression**:
  - Classification predicts a small set of categories (e.g., 0, 1, 2).
  - Unlike regression, classification doesn’t predict intermediate values (e.g., 0.5 or 1.7).

- **Using Multiple Inputs**:
  - **Example**: Tumor classification using both age and tumor size.
  - New dataset with two inputs (age and tumor size) to predict tumor type.
  - **Boundary in Classification**:
    - The learning algorithm may find a boundary (line) that separates categories (benign vs. malignant).
    - Additional inputs (e.g., cell size, cell shape) can improve predictions.



## Video 4: Unsupervised Learning Part 1

- **Introduction to Unsupervised Learning**:
  - Unsupervised learning is widely used after supervised learning.
  - It involves working with data **without output labels** (no "right answer" or supervision).
  - The goal is to **find patterns, structures, or clusters** in the data without predefined labels.

- **Clustering**:
  - **Definition**: Clustering is a type of unsupervised learning algorithm that groups data into clusters based on similarities.
  - Example - **Breast Cancer Data**:
    - With data on tumor size and patient age but no labels (benign/malignant), clustering can group the data based on similarities.

- **Applications of Clustering**:
  - **News Aggregation**:
    - Clustering is used to group related news stories (e.g., Google News groups articles about a single topic, like “panda birth”).
    - The algorithm identifies clusters based on shared keywords (e.g., “panda,” “twin,” “zoo”) without human supervision.
  - **Genetic Data Analysis**:
    - DNA microarray data represents genetic expression (rows = genes, columns = individuals).
    - Clustering groups individuals into types based on genetic similarities (e.g., “Type 1,” “Type 2”) without predefined categories.
  - **Market Segmentation**:
    - Companies use clustering on customer data to identify market segments.
    - Example from deep learning.ai: clustering identified groups with distinct motivations, such as skill growth, career advancement, and staying updated.



## Video 5: Unsupervised Learning Part 2

- **Formal Definition of Unsupervised Learning**:
  - In unsupervised learning, data consists only of **inputs (x)**, with **no output labels (y)**.
  - The goal is to **identify structure or patterns** within the data without predefined labels.

- **Types of Unsupervised Learning**:
  - **Clustering**:
    - Groups similar data points together into clusters.
  - **Anomaly Detection**:
    - Detects unusual events or data points that deviate from the norm.
    - Commonly used in **fraud detection** to identify suspicious financial transactions.
  - **Dimensionality Reduction**:
    - Reduces the size of a dataset by compressing data with minimal loss of information.
    - Useful for handling large datasets and reducing computational complexity.

- **Examples of Supervised vs. Unsupervised Learning**:
  - **Spam Filtering**: A **supervised learning** problem, as it uses labeled data (spam or non-spam).
  - **News Article Clustering**: An **unsupervised learning** problem, where clustering algorithms group similar articles without labels.
  - **Market Segmentation**: An **unsupervised learning** problem that discovers customer segments from unlabeled data.
  - **Diagnosing Diabetes**: A **supervised learning** problem, similar to the breast cancer example, using labeled data (diabetes or no diabetes).

