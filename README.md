# Interest Rate Models
**Jan Miksa**

Flexible playground for instrument rate model simulations and visualizations. 📈

<p align="center"><img src="ir_plot.png" alt="Example IR Plot" width="700"/></p>

## Features
- Hydra-based configurability
- Multiprocessed chain sampling
- Visualizations

| Solver | Status |
| ------ | -- |
| Euler-Maruyama | ✅ |
| Milstein | ✅ |

| Model | Status |
| ----- | -- |
| Vasicek | ✅ |
| CIR | ❌ |
| RB | ❌ |
| Ho-Lee | ❌ |
| Hull-White | ❌ |
| BDT | ❌ |
| BK | ❌ |

## Commands
**Setup**
```
conda create -n "irm" python=3.11
pip install -r requirements.txt
cp example.env .env
edit .env
```

**Running**
```
conda acitvate irm
HYDRA_FULL_ERROR={0/1} python src/main.py --config-name config 
```