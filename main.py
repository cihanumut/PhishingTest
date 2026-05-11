import argparse
from data_loader import DataLoader
from ml_models import MLModels
from graph_models import GraphModels
from graph_engine import GraphEngine
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Phishing Detection Hybrid Comparison')
    parser.add_argument('--mode', type=str, default='compare', choices=['train', 'compare'], help='Execution mode')
    args = parser.parse_args()

    # 1. Load Data
    loader = DataLoader('dataset_full.csv')
    df = loader.load_data()
    
    # 2. Pure Traditional ML (XGBoost)
    print("\n--- Phase 1: Pure Traditional ML ---")
    X_pure = df.drop('phishing', axis=1).values
    y = df['phishing'].values
    ml_pure = MLModels(model_type='xgboost')
    acc_pure = ml_pure.train(X_pure, y)

    # 3. Hybrid Approach (ML + Graph Metrics)
    print("\n--- Phase 2: Hybrid Approach (ML + Graph) ---")
    engine = GraphEngine()
    engine.build_feature_graph(df)
    df_hybrid = engine.augment_features(df)
    
    X_hybrid = df_hybrid.drop('phishing', axis=1).values
    ml_hybrid = MLModels(model_type='xgboost')
    acc_hybrid = ml_hybrid.train(X_hybrid, y)

    # 4. Pure GNN (for reference)
    print("\n--- Phase 3: Pure GNN ---")
    gnn_handler = GraphModels(num_features=X_pure.shape[1])
    gnn_data = gnn_handler.build_graph(X_pure, y, sample_size=5000)
    gnn_handler.train(gnn_data, epochs=50)
    acc_gnn = gnn_handler.evaluate(gnn_data)

    # 5. Final Results
    print("\n" + "="*40)
    print("HYBRID COMPARISON RESULTS")
    print(f"Pure XGBoost Accuracy:    {acc_pure:.4f}")
    print(f"Hybrid (XGBoost+Graph):   {acc_hybrid:.4f}")
    print(f"Pure GNN Accuracy:        {acc_gnn:.4f}")
    print("="*40)

    # 6. Visualization
    methods = ['Pure XGBoost', 'Hybrid (ML+Graph)', 'Pure GNN']
    accuracies = [acc_pure, acc_hybrid, acc_gnn]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylim(0, 1.05)
    plt.ylabel('Accuracy')
    plt.title('Hybrid Phishing Detection: Comparison')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', fontweight='bold')
    
    plt.savefig('hybrid_comparison_results.png')
    print("Hybrid comparison chart saved as 'hybrid_comparison_results.png'")

if __name__ == "__main__":
    main()
