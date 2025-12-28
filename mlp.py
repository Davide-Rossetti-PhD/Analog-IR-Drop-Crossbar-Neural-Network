import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import hashlib
from scipy import sparse
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from download_mnist_mlp import MLP_Crossbar
import csv
import os


# CACHE PER LE MATRICI ANALOGICHE (G_eff)
G_CACHE = {}  # verrà svuotata all'inizio di main()
STRUCT_CACHE = {}  # cache per strutture comuni in compute_Gtilde

def _hash_array(arr: np.ndarray) -> str:
    """Hash stabile del contenuto della matrice (per cache affidabile)."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def get_G_eff(G: np.ndarray, R: float, use_compensation: bool,
              layer_id: int, ti: int, to: int) -> np.ndarray:
    """
    Recupera G_eff dalla cache oppure la calcola (compute_Gtilde / compute_G4matrix).

    La chiave tiene conto di:
    - layer_id    : indice del layer Linear
    - ti, to      : indici di ingresso e uscita del sottoinsieme 'crossbar_dim' della matrice dei pesi
    - R           : resistenza delle linee
    - use_compensation : True/False (4-matrix o meno)
    - hash(G)     : contenuto reale dei pesi del crossbar
    """
    h = _hash_array(G)
    key = (layer_id, ti, to, R, use_compensation, h, G.shape)

    if key not in G_CACHE:
        if use_compensation:
            G_CACHE[key] = compute_G4matrix(G, R)
        else:
            G_CACHE[key] = compute_Gtilde(G, R)

    return G_CACHE[key]


def triang_inf_np(n: int):
    return np.tri(n, k=0, dtype=int)

def compute_Gtilde(G, R=0.35):
    """
      -Da G costruisco una matrice diagonale D (sparsa).
      -Costruisco le matrici sparse (LLt, LtL, Rw_base).
      -Costruisco il termine noto Rm = (1 x I_n) e lo salvo in formato denso.
      -Costruisco matrice H.
      -Risolvo in parallelo con CG n sistemi lineari H x = b con b colonna di Rm.
      -Riorganizzo le soluzioni in una matrice nxn tramite un'operazione di Kronecker finale (I x 1T).
      -Ritorno Gtilde.

    """
    global STRUCT_CACHE #uso una cache globale per evitare di ricostruire strutture pesanti quando n e R non cambiano.
    
    G = G.astype(np.float32, copy=False)
    n = G.shape[0]
    N = n * n
    gvec = G.T.reshape(-1, order='C')
    gvec = np.maximum(gvec, 1e-12)
    D = sparse.diags(1.0 / gvec, format='csr') # matrice diagonale sparsa (N×N)

    # Strutture comuni per (n, R)
    key = (n, R)
    if key not in STRUCT_CACHE:
        # Questi vengono precomputati una sola volta per (n, R)
        I_n = sparse.eye(n, format='csr')
        L = sparse.tril(np.ones((n, n)), format='csr')

        LLt = L @ L.T 
        LtL = L.T @ L

        Rw_base = sparse.kron(LLt, I_n, format='csr') + sparse.kron(I_n, LtL, format='csr') # (n2 x n2)

        # (1 x I_n), Rm vettore dei termini noti per CG
        one_col = np.ones((n, 1)) # (n x 1)
        Rm = sparse.kron(one_col, I_n, format='csr') # (n2 x n)
        Rm_dense = Rm.toarray()

        STRUCT_CACHE[key] = (Rw_base, Rm_dense)
    else:
        Rw_base, Rm_dense = STRUCT_CACHE[key]

    # Matrice H completa
    H = R * Rw_base + D

    #  CG parallelo sulle colonne
    def _solve_one_col(j):
        rhs = Rm_dense[:, j]
        sol, info = cg(H, rhs, rtol=1e-6, atol=0, maxiter=300)
        return sol

    # Tutti i core disponibili: full speed
    X = np.column_stack(
        Parallel(n_jobs=-1)(
            delayed(_solve_one_col)(j) for j in range(n)
        )
    ) #Parallel(n_jobs=-1) usa tutti i core della CPU,
    # delayed(_solve_one_col)(j) chiama _solve_one_col(j) per ogni colonna j,
    # il risultato è una lista di vettori soluzione sol (uno per colonna),
    # np.column_stack(...) mette tutti questi vettori come colonne di una matrice X.
    # Risultato: X è una matrice n2xn dove ognuna delle n colonne è la soluzione di un sistema lineare Hx=b


    # (I x 1T) X matrice (n×n)
    Lm = sparse.kron(sparse.eye(n, format='csr'),
                     np.ones((1, n)), format='csr')
    M = Lm @ X #collassa struttura (n x n x n x n) in (n x n)

    return M.T


# cache opzionale per non ricalcolare f* ogni volta
FSTAR_CACHE = {}  # fuori dalla funzione, a livello globale


def compute_G4matrix(G, R=0.35):
    """
    Compensazione 4-matrix con f* calibrato (vedi paper).
    """
    global FSTAR_CACHE

    n = G.shape[0]
    key = (n, R)  

    G1 = G
    G2 = np.flip(np.flip(G, 0), 1)
    G3 = np.flip(G, 1)
    G4 = np.flip(G, 0)

    Gt1 = compute_Gtilde(G1, R)
    Gt2 = compute_Gtilde(G2, R)
    Gt3 = compute_Gtilde(G3, R)
    Gt4 = compute_Gtilde(G4, R)

    Gt2 = np.flip(np.flip(Gt2, 0), 1)
    Gt3 = np.flip(Gt3, 1)
    Gt4 = np.flip(Gt4, 0)

    G_sum = Gt1 + Gt2 + Gt3 + Gt4  

    # calibrazione di f_star (una sola volta per (n, R)) 
    if key not in FSTAR_CACHE:
        f_star = calibrate_f_star(G_sum, G1, n_calib=256, f_min=1.0, f_max=6.0, n_f=50)
        FSTAR_CACHE[key] = f_star
    else:
        f_star = FSTAR_CACHE[key]

    return G_sum / f_star

def calibrate_f_star(G_sum, G_ideal, n_calib=256, f_min=1.0, f_max=6.0, n_f=50):
    """
    Calibra f_star per la 4-matrix dove v è campionato da una distribuzione semplice.
    G_sum   : Gt1 + Gt2 + Gt3 + Gt4   (n x n)
    G_ideal : G1, matrice ideale senza IR-drop 
    Restituisce:
      f_star (scalare)
    """
    n = G_ideal.shape[0]
    eps = 1e-8

    # campioni input v (qui +/- 1, shape: (n, n_calib))
    v = np.random.choice([-1.0, 1.0], size=(n, n_calib)).astype(np.float32)

    # proiezioni
    Hv = G_sum @ v        # (n, n_calib) crossbar compensato
    Gv = G_ideal @ v      # (n, n_calib) crossbar ideale

    # griglia di valori di f da provare
    f_candidates = np.linspace(f_min, f_max, n_f)

    best_f = f_candidates[0]
    best_loss = np.inf

    for f in f_candidates:
        y = Hv / f #applica un valore di prova di f
        diff = y - Gv                       # (n, n_calib) errore rispetto al modello ideale
        num = np.sum(np.abs(diff), axis=0)  # ||...||_1 per ogni v (errore assoluto)
        den = np.sum(np.abs(Gv), axis=0) + eps #normalizzazione
        rel = num / den                     # errore relativo per ogni v
        loss = np.mean(rel)                 # media su tutti i v

        if loss < best_loss: # trova il minimo
            best_loss = loss
            best_f = f

    return best_f


def analytical_forward_layer(x, W, crossbar_dim=256, R=0.35,
                             use_compensation=False, layer_id=0, Gmin=20e-6, Gmax=225e-6):
    """
    Forward analitico di un layer Linear con IR-drop e tiling.
    x: (fin, batch)
    W: (fout, fin)
    MW: limite massimo per i valori di conduttanza (clipping).
        Se None allora nessun limite.
    """
    fout, fin = W.shape
    batch = x.shape[1]

    # float32 per stabilità numerica
    x = x.astype(np.float64)
    W = W.astype(np.float64)

    # separazione pesi positivi/negativi
    Wp = np.maximum(W, 0.0)
    Wn = np.maximum(-W, 0.0)

    # mapping lineare W -- G
    Amax = W.max()
    Amin = W.min()
    # Calcolo gamma e delta 
    gamma = (Gmax - Gmin) / (Amax - Amin) #uS
    delta = Gmax - gamma * Amax #uS

    # Mapping corretto
    Gp = gamma * Wp + delta #uS
    Gn = gamma * Wn + delta #uS

    Wn = Gn # ora Wn sono conduttanze
    Wp = Gp # ora Wp sono conduttanze

    crossbar_dims_in = int(np.ceil(fin / crossbar_dim))
    crossbar_dims_out = int(np.ceil(fout / crossbar_dim))

    out = np.zeros((fout, batch), dtype=np.float32)

    for ti in range(crossbar_dims_in): #loop su crossbar_dim di ingresso

        # seleziono la porzione di input collegata a quel crossbar_dim
        c0 = ti * crossbar_dim
        cb = min((ti + 1) * crossbar_dim, fin)
        fin_crossbar_dim = cb - c0
        # preparo l’input espanso alla dimensione fisica del crossbar_dim (padding con zeri se necessario)
        xblk = x[c0:cb, :]
        xin = np.zeros((crossbar_dim, batch), dtype=np.float32)
        xin[:fin_crossbar_dim, :] = xblk

        for to in range(crossbar_dims_out): #loop su crossbar_dim di uscita
            r0 = to * crossbar_dim
            rb = min((to + 1) * crossbar_dim, fout)
            fout_crossbar_dim = rb - r0

            # blocchi di pesi
            Wp_blk = Wp[r0:rb, c0:cb]
            Wn_blk = Wn[r0:rb, c0:cb]

            # costruiamo Gp e Gn crossbar_dim × crossbar_dim (padding incluso)
            Gp = np.full((crossbar_dim, crossbar_dim), Gmin, dtype=np.float32)
            Gn = np.full((crossbar_dim, crossbar_dim), Gmin, dtype=np.float32)
            Gp[:fout_crossbar_dim, :fin_crossbar_dim] = Wp_blk
            Gn[:fout_crossbar_dim, :fin_crossbar_dim] = Wn_blk

            # parte positiva
            if Gp.max() > 0:
                G_eff_p = get_G_eff(Gp, R, use_compensation, layer_id, ti, to)
                y_p_full = G_eff_p @ xin #moltiplicazione matrice-vettore #ampere
                out[r0:rb, :] += y_p_full[:fout_crossbar_dim, :] 

            # parte negativa
            if Gn.max() > 0:
                G_eff_n = get_G_eff(Gn, R, use_compensation, layer_id, ti, to)
                y_n_full = G_eff_n @ xin
                out[r0:rb, :] -= y_n_full[:fout_crossbar_dim, :]
    out = out / gamma  # conversione ampere -> volt

    return out.astype(np.float32)  # ritorna l'output del layer (fout, batch)


def test_step(validation_data, model, criterion, device="cuda"):
    """
    calcolo loss e accuracy modello digitale
    """
    total_loss, correct, total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = total_loss / len(validation_data.dataset)
    test_acc = 100.0 * correct / total
    return test_loss, 100.0 - test_acc, test_acc

def show_image_2d(x, title=""):
    """
    Mostra un'immagine 2D (MNIST) con colormap in scala di grigi.
    x deve essere un array numpy 2D.
    """
    plt.figure(figsize=(3,3))
    plt.imshow(x, cmap="gray")
    plt.axis("off")
    plt.savefig(f"{title}.png")
    plt.close()

def logprobs_to_probs(logp):
    """
    Converte log-probabilità in probabilità (softmax).
    """
    lp = logp - np.max(logp, axis=0, keepdims=True)
    exp_lp = np.exp(lp)
    probs = exp_lp / np.sum(exp_lp, axis=0, keepdims=True)
    return probs

def forward_digital_layerwise(model, x_numpy):
    """
    Restituisce gli output digitali layer-per-layer esattamente come
      nella valutazione analogica.
    """
    x = x_numpy.copy()   # shape (fin, 1)
    outputs = {}

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            x = (W @ x) + b[:, None]
            outputs[i] = x.copy()

        elif isinstance(layer, nn.ReLU):
            x = np.maximum(x, 0)

        elif isinstance(layer, nn.LogSoftmax):
            m = np.max(x, axis=0, keepdims=True)
            x_shift = x - m
            logsumexp = m + np.log(np.sum(np.exp(x_shift), axis=0, keepdims=True))
            x = x - logsumexp

            outputs[i] = x.copy()

    return outputs


def compose_heatmap_histograms_three(vec_digital, vec_no, vec_comp,
                                     layer_id, outdir="."):
    """
    Serve a confrontare visivamente tre vettori numerici usando, per ciascuno, 
    una heatmap e un istogramma, il tutto organizzato in un'unica figura salvata su file.
    """


    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    titles = [
        "Digital MVM",
        "Analog MVM",
        "Analog MVM with four-matrix algorithm"
    ]

    vectors = [vec_digital, vec_no, vec_comp]

    for row, (title, v) in enumerate(zip(titles, vectors)):

        # heatmap
        side = int(np.ceil(np.sqrt(v.size)))
        canvas = np.zeros((side, side))
        canvas.flat[:v.size] = v

        axes[row, 0].imshow(canvas, cmap="viridis")
        axes[row, 0].set_title(f"{title} — heatmap")
        axes[row, 0].axis("off")

        # histogram
        axes[row, 1].hist(v, bins=50)
        axes[row, 1].set_title(f"{title} — histogram")

    fig.suptitle(f"Layer {layer_id}", fontsize=16)

    filename = os.path.join(outdir, f"layer_{layer_id}_FULL_COMPARISON.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def compose_probability_plots_three(probs_digital, probs_no, probs_comp, digit,
                                    outdir="."):
   

    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    titles = [
        "Digital MVM",
        "Analog MVM",
        "Analog MVM with four-matrix algorithm"
    ]

    vectors = [probs_digital, probs_no, probs_comp]

    class_labels = [f"Class {i}" for i in range(10)]

    save_probabilities_csv(
    probs_digital,
    probs_no,
    probs_comp,
    digit=digit,
    outdir=outdir
    )
   # Plot di ciascun pannello 
    for ax, title, v in zip(axes, titles, vectors):
        ax.bar(range(10), v * 100)   # convertiamo in percentuale
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks(range(10))
        ax.set_xticklabels(class_labels, rotation=45, fontsize=14, fontweight="bold")
        ax.set_ylabel("Class probability (%)", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 100)

    plt.tight_layout()

    filename = os.path.join(outdir, "final_probabilities.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()




def forward_analog_layerwise(model, x_numpy, R=0.35, crossbar_dim=256,
                             use_compensation=False, Gmin=20e-6, Gmax=225e-6):
    """
    Forward analogico layer-per-layer su UNA immagine, usando analytical_forward_layer.
    x_numpy: shape (N_in, 1), numpy array.
    R, crossbar_dim, use_compensation, Gmin, Gmax come nel resto del codice.

    Ritorna: dict layer_id , attivazioni (numpy array shape (N_out, 1))
    """
    # Copia per sicurezza, lavoriamo sempre in numpy
    x = x_numpy.copy().astype(np.float64)
    outputs = {}

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()

            # Forward analogico di UN layer
            y = analytical_forward_layer(
                x, W, crossbar_dim=crossbar_dim, R=R,
                use_compensation=use_compensation,
                layer_id=i, Gmin=Gmin, Gmax=Gmax
            )
            y += b[:, None]   # bias come nel digitale

            x = y
            outputs[i] = x.copy()

        elif isinstance(layer, nn.ReLU):
            x = np.maximum(x, 0)
            outputs[i] = x.copy()

        elif isinstance(layer, nn.LogSoftmax):
            m = np.max(x, axis=0, keepdims=True)
            x_shift = x - m
            logsumexp = m + np.log(np.sum(np.exp(x_shift), axis=0, keepdims=True))
            x = x - logsumexp
            outputs[i] = x.copy()

    return outputs




def save_probabilities_csv(probs_digital, probs_no, probs_comp,
                            digit, outdir):
    """
    Salva le probabilità finali in un CSV per una classe specifica.
    """
    filename = os.path.join(outdir, f"probabilities_class_{digit}.csv")

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "prob_digital",
            "prob_analog_no_comp",
            "prob_analog_comp"
        ])

        for k in range(10):
            writer.writerow([
                k,
                probs_digital[k],
                probs_no[k],
                probs_comp[k]
            ])

    print(f" Salvato CSV: {filename}")


def evaluate_analytical_accuracy(model, test_loader, R=0.35, crossbar_dim=256,
                                 Gmin=20e-6, Gmax=225e-6, save_plots=True):
    """
    Valuta l'accuracy analogica (senza e con compensazione).
    Se save_plots=True genera diagnostica grafica per UNA immagine randomica per ciascuna cifra 0–9.
    - Accuracy: calcolata su tutto il dataset (senza interruzioni).
    - Plot: per ogni cifra, quando si raggiunge l'indice random scelto,
            si salvano heatmap/istogrammi layer-wise e le probabilità finali.
    """
    model.eval()
    correct_no, correct_comp, total = 0, 0, 0

    # Costruisci lista di indici per ogni cifra
    digit_indices = {d: [] for d in range(10)}
    for idx in range(len(test_loader.dataset)):
        _, label = test_loader.dataset[idx]
        digit_indices[int(label)].append(idx)

    #  Scegli un indice randomico valido per ciascuna cifra
    chosen_index_for_digit = {d: random.choice(digit_indices[d]) for d in range(10) if digit_indices[d]}
    saved_digits = set()

    global_idx_start = 0  # indice globale nel dataset

    for images, labels in test_loader:
        batch_size = images.size(0)
        x = images.view(images.size(0), -1).T.numpy()

        # Forward analogico senza compensazione 
        x_no = x.copy()
        for i, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                y = analytical_forward_layer(
                    x_no, W, crossbar_dim=crossbar_dim, R=R,
                    use_compensation=False, layer_id=i,
                    Gmin=Gmin, Gmax=Gmax
                )
                y += b[:, None]
                x_no = y
            elif isinstance(layer, nn.ReLU):
                x_no = np.maximum(x_no, 0)
            elif isinstance(layer, nn.LogSoftmax):
                m = np.max(x_no, axis=0, keepdims=True)
                x_shift = x_no - m
                logsumexp = m + np.log(np.sum(np.exp(x_shift), axis=0, keepdims=True))
                x_no = x_no - logsumexp

        # Forward analogico con compensazione 
        x_comp = x.copy()
        for i, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()
                y = analytical_forward_layer(
                    x_comp, W, crossbar_dim=crossbar_dim, R=R,
                    use_compensation=True, layer_id=i,
                    Gmin=Gmin, Gmax=Gmax
                )
                y += b[:, None]
                x_comp = y
            elif isinstance(layer, nn.ReLU):
                x_comp = np.maximum(x_comp, 0)
            elif isinstance(layer, nn.LogSoftmax):
                m = np.max(x_comp, axis=0, keepdims=True)
                x_shift = x_comp - m
                logsumexp = m + np.log(np.sum(np.exp(x_shift), axis=0, keepdims=True))
                x_comp = x_comp - logsumexp

        # Accuracy 
        pred_no = np.argmax(x_no, axis=0)
        pred_comp = np.argmax(x_comp, axis=0)
        labels_np = labels.numpy()

        correct_no += (pred_no == labels_np).sum()
        correct_comp += (pred_comp == labels_np).sum()
        total += labels_np.size

        # Plot diagnostici: una immagine random per cifra
        if save_plots:
            for idx_in_batch, lbl in enumerate(labels_np):
                digit = int(lbl)
                if digit in saved_digits:
                    continue

                target_index = chosen_index_for_digit.get(digit, None)
                if target_index is None:
                    continue

                global_idx = global_idx_start + idx_in_batch
                if global_idx == target_index:
                    outdir = f"Digit {digit}"
                    os.makedirs(outdir, exist_ok=True)

                    # Immagine input
                    img0 = images[idx_in_batch].squeeze().numpy()
                    show_image_2d(img0, title=os.path.join(outdir, "input"))

                    # Input singolo per forward layer-wise
                    x0 = images[idx_in_batch].view(-1, 1).numpy()

                    # Forward digitali/analogici layer-wise
                    digital_outputs = forward_digital_layerwise(model, x0)
                    analog_no_outputs = forward_analog_layerwise(
                        model, x0, R=R, crossbar_dim=crossbar_dim,
                        use_compensation=False, Gmin=Gmin, Gmax=Gmax
                    )
                    analog_comp_outputs = forward_analog_layerwise(
                        model, x0, R=R, crossbar_dim=crossbar_dim,
                        use_compensation=True, Gmin=Gmin, Gmax=Gmax
                    )

                    # Heatmap + istogrammi layer-wise
                    for layer_id in digital_outputs.keys():
                        compose_heatmap_histograms_three(
                            digital_outputs[layer_id][:, 0],
                            analog_no_outputs[layer_id][:, 0],
                            analog_comp_outputs[layer_id][:, 0],
                            layer_id=layer_id,
                            outdir=outdir
                        )

                    # Probabilità finali
                    last_layer = max(digital_outputs.keys())
                    probs_digital = logprobs_to_probs(digital_outputs[last_layer][:, 0])
                    probs_no      = logprobs_to_probs(analog_no_outputs[last_layer][:, 0])
                    probs_comp    = logprobs_to_probs(analog_comp_outputs[last_layer][:, 0])

                    compose_probability_plots_three(
                        probs_digital, probs_no, probs_comp, digit,
                        outdir=outdir
                    )

                    saved_digits.add(digit)

        # aggiorna indice globale
        global_idx_start += batch_size

    acc_no = 100.0 * correct_no / total
    acc_comp = 100.0 * correct_comp / total
    return acc_no, acc_comp






def save_accuracy_csv(img_size, R, crossbar_dim, digital_acc, acc_no, acc_comp,
                      save_acc_csv=True):
    """
    Salva le accuracy in un CSV se save_acc_csv=True.
    """
    if not save_acc_csv:
        print("► Salvataggio CSV disabilitato")
        return

    csv_path = f"accuracy_{img_size}x{img_size}_R_{R}.csv"
    new_column = {
        "digital_acc": digital_acc,
        "analog_acc": acc_no,
        "analog_4matrix_acc": acc_comp,
    }
    crossbar_dim_label = f"crossbar_dim_{crossbar_dim}"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(index=["digital_acc", "analog_acc", "analog_4matrix_acc"])

    df[crossbar_dim_label] = pd.Series(new_column)
    df.to_csv(csv_path)
    print(f"\n Accuracy salvate in {csv_path}")




def main(
    IMG_SIZE=16,
    MODEL_PATH=None,
    R=0.35,
    crossbar_dim=256,
    GMIN=20e-6,
    GMAX=224.8e-6,
    save_acc_csv=True,
    save_plots=True
):

    global G_CACHE, STRUCT_CACHE, FSTAR_CACHE
    G_CACHE = {}
    STRUCT_CACHE = {}
    FSTAR_CACHE = {}

    torch.manual_seed(0)
    device = torch.device("cpu")
    print("\n Running inference on:", device)

    # Preprocessing coerente con IMG_SIZE
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),transforms.ToTensor()  
    ,transforms.Lambda(lambda t: t / (t.abs().max() * 1.2 + 1e-12))  
])

    test_set = datasets.MNIST("data", download=True, train=False, transform=transform)
    test_subset = Subset(test_set, list(range(0, 10000)))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # numero di input = IMG_SIZE × IMG_SIZE
    n_inputs = IMG_SIZE * IMG_SIZE

    model = MLP_Crossbar(n_inputs).to(device)

    if MODEL_PATH is None:
        MODEL_PATH = f"mnist_mlp_{n_inputs}.pth"

    print(f"Loading weights from: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)

    w_shape = state["layers.1.weight"].shape[1]
    if w_shape != n_inputs:
        raise RuntimeError(
            f"Mismatch: modello costruito per {n_inputs} input "
            f"ma checkpoint richiede {w_shape} input. "
        )

    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    print(f"\n R={R}, crossbar_dim={crossbar_dim}")
    
    # Accuracy digitale
    _, _, digital_acc = test_step(test_loader, model, criterion, device=device)
    print(f"\n Accuracy digitale: {digital_acc:.2f}%")

    # Accuracy IR-drop analitica
    acc_no, acc_comp = evaluate_analytical_accuracy(model, test_loader, R=R, crossbar_dim=crossbar_dim, Gmin=GMIN, Gmax=GMAX, save_plots=save_plots)
    print(f" Accuratezza analitica (senza compensazione): {acc_no:.2f}%")
    print(f" Accuratezza analitica (con compensazione): {acc_comp:.2f}%")
    
    # Salvataggio CSV
    save_accuracy_csv(IMG_SIZE, R, crossbar_dim, digital_acc, acc_no, acc_comp,
                    save_acc_csv=save_acc_csv)


if __name__ == "__main__":
    main()
