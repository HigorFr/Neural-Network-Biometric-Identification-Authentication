import os
import matplotlib.pyplot as plt
import numpy as np

base = "Resultados_Autenticacao"
descritores = ["HOG", "LBP"]
modelos = ["linear", "mlp"]

os.makedirs("Graficos", exist_ok=True)

for descritor in descritores:
    for modelo in modelos:

        caminho = os.path.join(base, descritor, modelo, "run_error.txt")

        if not os.path.exists(caminho):
            print(f"[ERRO] Arquivo não encontrado: {caminho}")
            continue

        # Dicionários para acumular os erros por época
        treino_dict = {}
        val_dict = {}

        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                if ";" not in linha:
                    continue
                e, t, v = linha.strip().split(";")
                e = int(e)
                t = float(t)
                v = float(v)

                # Agrupa os valores por época
                treino_dict.setdefault(e, []).append(t)
                val_dict.setdefault(e, []).append(v)

        # Ordena as épocas e calcula a média
        ep = sorted(treino_dict.keys())
        treino_mean = [np.mean(treino_dict[e]) for e in ep]
        val_mean = [np.mean(val_dict[e]) for e in ep]

        plt.figure()
        plt.plot(ep, treino_mean, label="Treino (média folds)")
        plt.plot(ep, val_mean, label="Validação (média folds)")
        plt.xlabel("Época")
        plt.ylabel("Erro")
        plt.title(f"{modelo.upper()} - {descritor} (Média por época)")
        plt.legend()
        plt.grid(True)

        nome_saida = f"Graficos/erro_{descritor}_{modelo}_mean.png"
        plt.savefig(nome_saida)
        plt.close()

        print(f"[OK] Gráfico salvo em {nome_saida}")
