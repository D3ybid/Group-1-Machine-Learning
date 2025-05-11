import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

MIN_SUPP = 0.01 # For minimum support
MIN_CONF = 0.3   # For minimum confidence

# STEP 1: Load the dataset
df = pd.read_csv("apriori/Groceries_dataset.csv")

# STEP 2: Create unique transaction IDs
df['Transaction'] = df['Member_number'].astype(str) + "_" + df['Date'].astype(str)

# STEP 3: Filter to first 300 unique transactions
first_300_transactions = df['Transaction'].drop_duplicates().head(300)
df = df[df['Transaction'].isin(first_300_transactions)]

# STEP 4: Convert to list of transactions
transactions = df.groupby('Transaction')['itemDescription'].apply(list).tolist()

# Log transactions
with open("log_transactions.txt", "w") as f:
    for tid, items in df.groupby('Transaction')['itemDescription'].apply(list).items():
        f.write(f"{tid}:\t" + ', '.join(items) + '\n')

# STEP 5: One-hot encode
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# STEP 6: Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPP, use_colnames=True)

# STEP 7: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONF)

# STEP 8 (optional): Filter itemsets and rules
frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
rules = rules[(rules['antecedents'].apply(len) > 0) & (rules['consequents'].apply(len) > 0)]

# Log frequent Itemsets and Support Values
with open("log_frequent_item_sets.txt", "w") as f:
    for _, row in frequent_itemsets.sort_values(by="support", ascending=False).iterrows():
        items = ', '.join(sorted(row['itemsets']))
        support = f"{row['support']:.4f}"
        f.write(f"Itemset ({len(row['itemsets'])}): {items} | Support: {support}\n")

# Log Rules in readable format
with open("log_association_rules.txt", "w") as f:
    for _, row in rules.iterrows():
        antecedent = ' + '.join(sorted(row['antecedents']))
        consequent = ' + '.join(sorted(row['consequents']))
        f.write(f"[{antecedent}] => [{consequent}]\n")

# Print top results
if frequent_itemsets.empty:
    print("No frequent itemsets found.")
else:
    print("Top Frequent Itemsets:")
    for idx, row in frequent_itemsets.sort_values(by="support", ascending=False).head(10).iterrows():
        items = ', '.join(sorted(row['itemsets']))
        print(f"{items}: support = {row['support']:.4f}")

if rules.empty:
    print("\n No association rules found.")
else:
    print("\n Top Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
          .sort_values(by='lift', ascending=False).head(10))

# Visualization functions
def plot_top_frequent_items(itemsets, top_n=10):
    from collections import Counter
    all_items = [item for itemset in itemsets['itemsets'] for item in itemset]
    support_dict = {}
    for itemset, support in zip(itemsets['itemsets'], itemsets['support']):
        for item in itemset:
            support_dict[item] = support_dict.get(item, 0) + support / len(itemset)
    item_support_df = pd.DataFrame(list(support_dict.items()), columns=['item', 'support'])
    top_items = item_support_df.sort_values(by='support', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_items, x='support', y='item', hue='item', palette='magma', legend=False)
    plt.title('Top Frequent Items (By Estimated Support)')
    plt.xlabel('Support')
    plt.ylabel('Item')
    plt.tight_layout()
    plt.show()

def plot_association_rules(rules, top_n=10):
    if rules.empty:
        print("⚠️ No association rules to display in the graph.")
        return
    G = nx.DiGraph()
    top_rules = rules.sort_values(by='lift', ascending=False).head(top_n)
    for _, row in top_rules.iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                G.add_edge(ant, cons, weight=row['confidence'])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2500)
    nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=20, edge_color=edge_weights,
                        edge_cmap=plt.cm.Blues, width=2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                            norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Confidence')
    plt.title('Top Association Rules Network')
    plt.axis('off')
    plt.show()

# Call visualizations
plot_top_frequent_items(frequent_itemsets)
plot_association_rules(rules)
