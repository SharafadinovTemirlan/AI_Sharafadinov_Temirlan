from itertools import combinations
import os

def load_transactions(filepath):
    """
    Преобразуем числовые данные из blood_data.txt в категориальные признаки.
    """
    transactions = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # Пропускаем заголовок
        for line in lines:
            recency, frequency, monetary, time, donated = map(int, line.strip().split(','))
            tags = []

            if recency <= 2:
                tags.append('recent_donor')
            if frequency >= 20:
                tags.append('frequent_donor')
            if monetary >= 5000:
                tags.append('high_volume')
            if time >= 50:
                tags.append('long_term_donor')
            if donated == 1:
                tags.append('donated_march')

            transactions.append(tags)
    return transactions

def apriori(transactions, min_support=0.3):
    itemsets = []
    support_data = {}

    def get_support(itemset):
        count = sum(1 for transaction in transactions if all(item in transaction for item in itemset))
        return count / len(transactions)

    single_items = sorted(set(item for t in transactions for item in t))
    current_lset = [[item] for item in single_items]

    while current_lset:
        new_lset = []
        for itemset in current_lset:
            support = get_support(itemset)
            if support >= min_support:
                itemsets.append(itemset)
                support_data[tuple(itemset)] = support

        next_candidates = set()
        for i in range(len(current_lset)):
            for j in range(i+1, len(current_lset)):
                candidate = sorted(set(current_lset[i]) | set(current_lset[j]))
                if len(candidate) == len(current_lset[0]) + 1:
                    next_candidates.add(tuple(candidate))
        current_lset = [list(item) for item in next_candidates]

    return itemsets, support_data

# === Main ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "blood_data.txt")
transactions = load_transactions(DATA_PATH)

itemsets, support = apriori(transactions, min_support=0.3)
print("Frequent Itemsets:")
for itemset in itemsets:
    print(f"{itemset} - support: {support[tuple(itemset)]:.2f}")


def run():
    return "Алгоритм Apriori выполнен успешно!"
