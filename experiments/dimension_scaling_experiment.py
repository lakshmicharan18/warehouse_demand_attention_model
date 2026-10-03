"""Run the dimension-scaling attention mathematics experiment."""
from pathlib import Path
import csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from warehouse_demand_attention.dimension_scaling_experiment import run_experiment  # noqa: E402

def main():
    results=run_experiment(); out=Path(__file__).resolve().parents[1]/'outputs'
    fields=list(results[0].__dataclass_fields__)
    with (out/'dimension_scaling_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([r.__dict__ for r in results])
    print('d_k | raw std | scaled std | raw/sqrt(d_k) | unscaled entropy | scaled entropy | unscaled max | scaled max | grad unscaled | grad scaled')
    for r in results:
        print(f'{r.d_k:>3} | {r.raw_std:.4f} | {r.scaled_std:.4f} | {r.raw_std/(r.d_k**0.5):.4f} | {r.unscaled_entropy:.4f} | {r.scaled_entropy:.4f} | {r.unscaled_max_weight:.4f} | {r.scaled_max_weight:.4f} | {r.unscaled_gradient_norm:.3e} | {r.scaled_gradient_norm:.3e}')
    x=[r.d_k for r in results]
    plt.figure(figsize=(6,4)); plt.plot(x,[r.raw_std for r in results],marker='o',label='Unscaled'); plt.plot(x,[r.scaled_std for r in results],marker='o',label='Scaled'); plt.xlabel('d_k'); plt.ylabel('Logit standard deviation'); plt.title('Attention Logit Scale'); plt.legend(); plt.tight_layout(); plt.savefig(out/'dimension_scaling_logits.png',dpi=150); plt.close()
    plt.figure(figsize=(6,4)); plt.plot(x,[r.unscaled_entropy for r in results],marker='o',label='Unscaled'); plt.plot(x,[r.scaled_entropy for r in results],marker='o',label='Scaled'); plt.xlabel('d_k'); plt.ylabel('Attention entropy'); plt.title('Attention Concentration'); plt.legend(); plt.tight_layout(); plt.savefig(out/'dimension_scaling_entropy.png',dpi=150); plt.close()
if __name__ == '__main__': main()
