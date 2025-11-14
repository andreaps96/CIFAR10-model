# Classificazione CIFAR-10 con ResNet34 Adattato (94.21% Accuracy)

Questo repository contiene un'implementazione PyTorch per la classificazione di immagini sul dataset **CIFAR-10** utilizzando un'architettura **ResNet34** pre-addestrata e profondamente adattata.

L'obiettivo di questo script non è solo classificare, ma dimostrare una pipeline di training robusta che impiega tecniche avanzate per massimizzare l'accuratezza (raggiungendo il **94.21%** sul test set) quando si applica il transfer learning a immagini di piccole dimensioni (32x32).

---

## Architettura del Modello: Adattamento di ResNet34

Il modello base è un **ResNet34** (`ResNet34_Weights.IMAGENET1K_V1`) pre-addestrato su ImageNet (immagini 224x224). Per evitare una perdita di informazioni spaziali dovuta alle dimensioni ridotte delle immagini CIFAR-10 (32x32), l'architettura è stata modificata:

1.  **`model.conv1` (Layer Convoluzionale Iniziale):**
    * *Originale:* `Conv2d(7x7, stride=2)`
    * *Modificato:* `Conv2d(3x3, stride=1, padding=1)`
    * *Motivazione:* Un kernel 7x7 con stride 2 è troppo aggressivo per un input 32x32. Questa modifica preserva la dimensione spaziale (32x32) nel primo layer, mantenendo più informazioni.

2.  **`model.maxpool` (Layer di Pooling Iniziale):**
    * *Originale:* `MaxPool2d(3x3, stride=2)`
    * *Modificato:* `nn.Identity()` (Layer rimosso/bypassato)
    * *Motivazione:* Rimuovere il pooling iniziale previene un'ulteriore e prematura riduzione della dimensionalità.

3.  **`model.fc` (Classificatore Finale):**
    * *Originale:* `Linear(512, 1000)`
    * *Modificato:* `nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))`
    * *Motivazione:* Sostituzione della testa per adattarla alle 10 classi di CIFAR-10, con l'aggiunta di **Dropout** (p=0.2) per la regolarizzazione.

---

## Tecniche di Ottimizzazione e Training

L'addestramento è una strategia *fine-tuning* in due fasi progettata per stabilizzare il classificatore prima di addestrare l'intera rete.

### Fase 1: Warm-up del Classificatore (8 Epoche)

Prima di sbloccare l'intero *backbone*, viene addestrata solo la testa di classificazione (`fc`).
* **Layer Congelati:** Tutti i parametri (eccetto `model.fc`) sono impostati su `requires_grad = False`.
* **Ottimizzatore:** `SGD`
* **Learning Rate:** `0.01` (fisso)
* **Weight Decay (L2):** `5e-4`
* **Momentum:** `0.9`

### Fase 2: Fine-Tuning Completo (Fino a 100 Epoche)

L'intera rete viene sbloccata (`requires_grad = True`) e addestrata con un learning rate differenziato.
* **Ottimizzatore:** `SGD`
* **Weight Decay (L2):** `5e-4`
* **Momentum:** `0.9`
* **Learning Rate Differenziato:**
    * *Backbone* (layer pre-addestrati): `lr = 0.001` (basso, per non "distruggere" i pesi di ImageNet).
    * *Head* (classificatore `fc`): `lr = 0.01` (alto, per un adattamento rapido).
* **Scheduler:** `CosineAnnealingLR`
    * Il LR decade dolcemente su `T_max=100` epoche fino a `eta_min=1e-6`, aiutando il modello a convergere in un minimo stabile.

---

## Tecniche di Regolarizzazione

Per combattere l'overfitting e migliorare la generalizzazione, sono state implementate diverse tecniche:

* **Label Smoothing:**
    * La funzione di loss è `nn.CrossEntropyLoss(label_smoothing=0.1)`.
    * Questo impedisce al modello di diventare "troppo sicuro" delle sue previsioni (evitando probabilità di 1.0) e migliora la calibrazione.

* **Data Augmentation:**
    * `T.RandomCrop(32, padding=4)`
    * `T.RandomHorizontalFlip()`
    * *Motivazione:* Aumentano artificialmente la diversità del training set.

* **CutOut (via `RandomErasing`):**
    * `T.RandomErasing(p=0.25, scale=(0.02, 0.1))`
    * Una regione casuale dell'immagine viene oscurata. Questo costringe il modello a imparare da un contesto più ampio e a non fare affidamento su singole feature salienti.

* **Gradient Clipping:**
    * `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
    * Utilizzato in ogni step di training per prevenire l'esplosione dei gradienti e stabilizzare l'addestramento, specialmente nelle prime fasi del fine-tuning.

* **Early Stopping:**
    * Il training di Fase 2 viene monitorato sulla *validation loss* (`val_loss`) con una pazienza (`patience`) di 15 epoche.
    * L'addestramento si interrompe automaticamente se le prestazioni sulla validazione non migliorano, salvando il checkpoint migliore (`prova_cifar.pt`).

* **Weight Decay (L2) e Dropout:**
    * Come menzionato nelle sezioni precedenti, `weight_decay=5e-4` (L2) è applicato dall'ottimizzatore e `Dropout(p=0.2)` è inserito nella testa di classificazione.

## Performance e Metriche Finali

Dopo l'addestramento, il modello è stato valutato sul test set completo di CIFAR-10 (10.000 immagini).

* **Test Accuracy:** **94.21%**
* **Test Loss:** **0.6457**

### Report di Classificazione

La precisione e il richiamo sono elevati e bilanciati su quasi tutte le classi.

          precision    recall  f1-score   support

       0       0.94      0.96      0.95      1000
       1       0.96      0.97      0.97      1000
       2       0.95      0.91      0.93      1000
       3       0.88      0.88      0.88      1000
       4       0.94      0.95      0.94      1000
       5       0.91      0.90      0.90      1000
       6       0.96      0.97      0.97      1000
       7       0.96      0.96      0.96      1000
       8       0.96      0.97      0.96      1000
       9       0.96      0.95      0.95      1000

accuracy                           0.94     10000
macro avg      0.94      0.94      0.94     10000 
weighted avg   0.94      0.94      0.94     10000
---