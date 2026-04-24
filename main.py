import csv
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_rules
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(filename):
    """อ่านข้อมูล transaction จากไฟล์ CSV แบบ binary matrix (1 = มีสินค้า, 0 = ไม่มี)"""
    dataset = []
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                transaction = set()
                for i, value in enumerate(row):
                    if value == '1':
                        transaction.add(headers[i])
                if transaction:
                    dataset.append(transaction)
    except FileNotFoundError:
        print("File not found.")
    return dataset


def create_C1(dataset):
    """สร้าง Candidate 1-itemsets จากทุก item ที่ปรากฏใน dataset"""
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scan_D(D, Ck, min_support):
    """
    นับ support ของแต่ละ candidate itemset
    คืนค่าเฉพาะ itemset ที่ support >= min_support
    """
    sscnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                sscnt[can] = sscnt.get(can, 0) + 1

    num_items = float(len(D))
    retList = []
    support_data = {}

    for key in sscnt:
        support = sscnt[key] / num_items
        if support >= min_support:
            retList.insert(0, key)
        support_data[key] = support
    return retList, support_data


def apriori_gen(Lk, k):
    """
    สร้าง Candidate k-itemsets จาก frequent (k-1)-itemsets
    โดยรวม 2 itemsets ที่มี k-2 items แรกเหมือนกัน (join step)
    """
    retList = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataset, min_support=0.15):
    """
    Apriori Algorithm (from scratch)
    วนหา frequent itemsets ทุกขนาด โดยใช้ Apriori Property:
    ถ้า itemset ไม่ frequent ทุก superset ก็ไม่ frequent
    """
    C1 = create_C1(dataset)
    D = list(map(set, dataset))
    L1, support_data = scan_D(D, C1, min_support)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = apriori_gen(L[k - 2], k)        # สร้าง candidates
        Lk, supK = scan_D(D, Ck, min_support) # กรอง candidates
        support_data.update(supK)
        L.append(Lk)
        k += 1
    return L, support_data


def generate_rules(L, support_data, min_confidence=0.6):
    """
    สร้าง Association Rules จาก frequent itemsets
    คำนวณ confidence = support(A∪B) / support(A)
    คำนวณ lift = confidence / support(B)
    """
    rules = []
    for i in range(1, len(L)):
        for itemset in L[i]:
            for size in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    confidence = support_data[itemset] / support_data[antecedent]
                    lift = confidence / support_data[consequent]
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support':    round(support_data[itemset], 4),
                            'confidence': round(confidence, 4),
                            'lift':       round(lift, 4),
                        })
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    return rules


def run_mlxtend(dataset, min_support=0.15, min_confidence=0.6):
    """รัน Apriori ด้วย mlxtend library เพื่อ validate ผลลัพธ์"""
    te = TransactionEncoder()
    te_array = te.fit_transform([list(t) for t in dataset])
    df = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = mlxtend_apriori(df, min_support=min_support, use_colnames=True)
    rules = mlxtend_rules(freq_items, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values('confidence', ascending=False).reset_index(drop=True)
    return freq_items, rules


def plot_frequent_itemsets(L, support_data):
    """แสดง bar chart ของ support แต่ละ frequent itemset เรียงจากมากไปน้อย"""
    labels, supports = [], []
    for level in L:
        for itemset in level:
            labels.append(', '.join(sorted(itemset)))
            supports.append(round(support_data[itemset] * 100, 2))

    sorted_pairs = sorted(zip(supports, labels), reverse=True)
    supports, labels = zip(*sorted_pairs)

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, supports, color='steelblue')
    ax.set_xlabel('Support (%)', fontsize=12)
    ax.set_title('Frequent Itemsets — Support', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, supports):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'frequent_itemsets.png'), dpi=150)
    plt.close()


def plot_rules(rules, top_n=10):
    """แสดง top rules เรียงตาม confidence พร้อม lift"""
    top_rules = rules[:top_n]
    rule_labels = [f"{', '.join(sorted(r['antecedent']))} → {', '.join(sorted(r['consequent']))}"
                   for r in top_rules]
    confidences = [r['confidence'] * 100 for r in top_rules]
    lifts       = [r['lift'] for r in top_rules]

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(rule_labels, confidences, color='steelblue')
    axes[0].set_xlabel('Confidence (%)', fontsize=12)
    axes[0].set_title('Top Association Rules — Confidence', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    for i, val in enumerate(confidences):
        axes[0].text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    axes[1].barh(rule_labels, lifts, color='coral')
    axes[1].set_xlabel('Lift', fontsize=12)
    axes[1].set_title('Top Association Rules — Lift', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    for i, val in enumerate(lifts):
        axes[1].text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'association_rules.png'), dpi=150)
    plt.close()


def plot_comparison(my_rules, ml_rules):
    """เปรียบเทียบ confidence และ lift ระหว่าง from scratch กับ mlxtend"""
    def rule_key(ant, con):
        return f"{', '.join(sorted(ant))} → {', '.join(sorted(con))}"

    my_dict = {rule_key(r['antecedent'], r['consequent']): r for r in my_rules}
    ml_dict = {rule_key(set(r['antecedents']), set(r['consequents'])): r
               for _, r in ml_rules.iterrows()}

    common_keys = [k for k in my_dict if k in ml_dict]

    if not common_keys:
        print("No common rules to compare.")
        return

    labels      = common_keys
    my_conf     = [my_dict[k]['confidence'] * 100 for k in labels]
    ml_conf     = [ml_dict[k]['confidence'] * 100 for k in labels]
    my_lift     = [my_dict[k]['lift'] for k in labels]
    ml_lift     = [ml_dict[k]['lift'] for k in labels]

    x = range(len(labels))
    width = 0.35

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar([i - width/2 for i in x], my_conf, width, label='From Scratch', color='steelblue')
    axes[0].bar([i + width/2 for i in x], ml_conf, width, label='mlxtend',      color='coral')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    axes[0].set_ylabel('Confidence (%)')
    axes[0].set_title('Confidence Comparison', fontweight='bold')
    axes[0].legend()

    axes[1].bar([i - width/2 for i in x], my_lift, width, label='From Scratch', color='steelblue')
    axes[1].bar([i + width/2 for i in x], ml_lift, width, label='mlxtend',      color='coral')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    axes[1].set_ylabel('Lift')
    axes[1].set_title('Lift Comparison', fontweight='bold')
    axes[1].legend()

    plt.suptitle('Apriori (from scratch) vs mlxtend', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'comparison.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    dataset = load_data(os.path.join(BASE_DIR, 'ShopCT.csv'))
    print(f"Transactions loaded: {len(dataset)}")

    # ── From scratch ──────────────────────────────────────────────────
    L, support_data = apriori(dataset, min_support=0.15)

    print("\n========== Frequent Itemsets (from scratch) ==========")
    for i, level in enumerate(L):
        if not level:
            continue
        print(f"\nL{i + 1} (size {i + 1}): {len(level)} itemsets")
        for itemset in level:
            print(f"  {str(set(itemset)):<40} support: {support_data[itemset] * 100:.2f}%")

    rules = generate_rules(L, support_data, min_confidence=0.6)

    print("\n========== Association Rules (from scratch) ==========")
    print(f"{'Antecedent':<25} {'Consequent':<20} {'Support':>9} {'Confidence':>11} {'Lift':>7}")
    print("-" * 76)
    for r in rules:
        ant = ', '.join(sorted(r['antecedent']))
        con = ', '.join(sorted(r['consequent']))
        print(f"{ant:<25} {con:<20} {r['support'] * 100:>8.2f}% {r['confidence'] * 100:>10.2f}% {r['lift']:>7.2f}")

    # ── mlxtend ───────────────────────────────────────────────────────
    freq_ml, rules_ml = run_mlxtend(dataset, min_support=0.15, min_confidence=0.6)

    print("\n========== Association Rules (mlxtend) ==========")
    print(f"{'Antecedent':<25} {'Consequent':<20} {'Support':>9} {'Confidence':>11} {'Lift':>7}")
    print("-" * 76)
    for _, r in rules_ml.iterrows():
        ant = ', '.join(sorted(r['antecedents']))
        con = ', '.join(sorted(r['consequents']))
        print(f"{ant:<25} {con:<20} {r['support'] * 100:>8.2f}% {r['confidence'] * 100:>10.2f}% {r['lift']:>7.2f}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_frequent_itemsets(L, support_data)
    plot_rules(rules)
    plot_comparison(rules, rules_ml)

    print("\nGraphs saved: frequent_itemsets.png, association_rules.png, comparison.png")
