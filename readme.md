## Parte 0

### Rodar a 0(a) para ter o "piso" da CNN do zero

```bash
python scripts/train_supervised.py --backbone unet --budget 5 --seed 0
```

### **Rodar a 0(b)** com 3 seeds (o PA pede 3 seeds no orçamento de 5 na Parte 2, e a 0(b) vira a "curva do piso"; vale já rodar as 3):

```bash
for s in 0 1 2; do
    python scripts/train_supervised.py --backbone resnet18 --budget 5 --seed $s
done
```